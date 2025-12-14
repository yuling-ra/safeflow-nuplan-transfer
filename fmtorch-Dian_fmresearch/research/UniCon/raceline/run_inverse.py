# -*- coding: utf-8 -*-
"""
Inverse-only runner (staged-OCP + optional light MPPI)

策略：
- 前期(Stages 1~3)：强“贴线”吸附(Huber on lateral error)，安全权重≈0，仅做轻姿态/平滑，得到好起点
- 后期(Stages 4~6)：逐步增大安全与常规项，回到稳定可行解
- 曲率感知 v_ref，线段“中点安全”，方向分解误差(横>>纵)，方向盘变化/加速度jerk软罚
"""

import argparse
import numpy as np

from raceline_core import (
    MapData, OCPConfig, load_map, build_reference_from_raceline,
    build_sdf_from_grid, mppi_local_refine,
    plot_results, plot_mppi_results, sample_in_box, nearest_index,
    resample_polyline_to_N, simulate_vehicle_bc, unpack_controls, sample_sdf_bilinear
)

# -----------------------------
# Helpers
# -----------------------------
def box_midpoints(args):
    START_A = np.array(args.start_box[:2]); START_B = np.array(args.start_box[2:])
    END_A   = np.array(args.end_box[:2]);   END_B   = np.array(args.end_box[2:])
    start_mid = 0.5 * (START_A + START_B)
    end_mid   = 0.5 * (END_A   + END_B)
    return start_mid, end_mid

def robust_theta_from_polyline(poly: np.ndarray):
    dxy = np.diff(poly, axis=0, append=poly[-1:])
    if len(poly) >= 2:
        dxy[-1] = poly[-1] - poly[-2]
    theta = np.arctan2(dxy[:, 1], dxy[:, 0])
    return theta

def nonzero_accel_init(control_steps: int, a_fraction: float = 0.5):
    z = np.zeros((control_steps, 2), dtype=np.float64)
    z[: max(1, control_steps // 3), 0] = np.arctanh(np.clip(a_fraction, 1e-6, 1-1e-6))
    return z.reshape(-1)

# --------- 曲率感知 v_ref ----------
def curvature_capped_speed_profile(raceline_ref: np.ndarray,
                                   theta_ref: np.ndarray,
                                   cfg: OCPConfig,
                                   path_len_px: float,
                                   v_factor: float = 3.0,
                                   alpha_lat: float = 0.45):
    """生成曲率限速后的 v_ref 与 v_max（像素单位）"""
    control_steps = cfg.state_steps - 1
    total_time = control_steps * cfg.dt
    avg_speed = path_len_px / (total_time + 1e-12)  # px/s
    v_max_base = v_factor * avg_speed
    v_ref = np.clip(np.linspace(0, 2.0 * avg_speed, cfg.state_steps), 0, v_max_base)

    dtheta = np.diff(theta_ref, append=theta_ref[-1:])
    ds = np.maximum(1e-6, np.linalg.norm(np.diff(raceline_ref, axis=0, append=raceline_ref[-1:]), axis=1))
    kappa = dtheta / ds

    a_lat_max = alpha_lat * cfg.a_max
    v_cap_curve = np.sqrt(np.maximum(1e-6, a_lat_max / (np.abs(kappa) + 1e-6)))

    v_ref = np.minimum(v_ref, v_cap_curve)
    v_max = float(min(v_max_base, np.nanmax(v_cap_curve)))
    return v_ref, v_max

def build_full_inverse_reference(md: MapData, cfg: OCPConfig,
                                 use_box_midpoints: bool,
                                 start_mid: np.ndarray, end_mid: np.ndarray):
    raceline_full = md.raceline_grid.copy()
    if use_box_midpoints:
        print("[full] override endpoints with box centers")
        raceline_full[0]  = start_mid
        raceline_full[-1] = end_mid
    # 反向
    raceline_full = raceline_full[::-1].copy()
    raceline_ref, theta_ref, path_len_px = build_reference_from_raceline(
        raceline_full, cfg.state_steps
    )
    v_ref, v_max = curvature_capped_speed_profile(raceline_ref, theta_ref, cfg, path_len_px,
                                                  v_factor=3.0, alpha_lat=0.45)
    return raceline_ref, theta_ref, v_ref, v_max, path_len_px

# =========================
# 覆盖版 OCP：方向分解 + Huber吸附 + 中点安全 + 软罚
# =========================
def _huber(x, delta):
    # smooth L1
    absx = np.abs(x)
    return np.where(absx <= delta, 0.5 * x * x, delta * (absx - 0.5 * delta))

class OCPWeights:
    # 最终权重（各 stage 会覆盖）
    w_lat: float = 18.0
    w_lon: float = 2.0
    w_head: float = 6.0
    w_u: float = 1e-3
    w_smooth: float = 0.05
    w_v: float = 0.01
    w_terminal: float = 80.0
    w_drate: float = 6.0
    w_jerk: float = 2.0
    # 附加
    v_min_factor: float = 0.20
    w_vmin: float = 0.03
    w_snap: float = 0.0        # Huber 吸附权重
    snap_delta: float = 3.0

def ocp_cost_directional(z_flat,
                          x0, raceline_ref, theta_ref, v_ref,
                          dt, state_steps, control_steps,
                          a_max, delta_max, L_pix, v_max,
                          sdf_map=None, safety_margin_pix=8.0, w_safety=40.0,
                          delta_rate_max=2.5,
                          avg_speed_pixels: float = None,
                          w: OCPWeights = OCPWeights()):
    # 控制序列 & 动力学
    u = unpack_controls(z_flat, control_steps, a_max, delta_max)
    states = simulate_vehicle_bc(x0, u, dt, state_steps, L_pix=L_pix, v_max=v_max,
                                 delta_rate_max=delta_rate_max)

    # 方向分解误差
    pos_err = states[:, :2] - raceline_ref
    c = np.cos(theta_ref); s = np.sin(theta_ref)
    t = np.stack([c, s], axis=1); n = np.stack([-s, c], axis=1)
    e_lon = np.einsum('ij,ij->i', pos_err, t)
    e_lat = np.einsum('ij,ij->i', pos_err, n)

    # 贴线平方 + Huber 吸附（早期强吸附）
    J_path = w.w_lat * np.sum(e_lat**2) + w.w_lon * np.sum(e_lon**2)
    J_snap = w.w_snap * np.sum(_huber(e_lat, w.snap_delta))

    # 姿态
    dth = (states[:, 2] - theta_ref + np.pi) % (2*np.pi) - np.pi
    J_head = w.w_head * np.sum(dth**2)

    # 控制能量 + 平滑 + 软罚
    J_u = w.w_u * np.sum(u**2)
    Ju_smooth = w.w_smooth * np.sum(np.diff(u, axis=0)**2)
    J_drate = w.w_drate * np.sum(np.diff(u[:, 1])**2)
    J_jerk  = w.w_jerk  * np.sum(np.diff(u[:, 0])**2)

    # 速度跟踪 + 最小速度
    J_v = w.w_v * np.sum((states[:, 3] - v_ref)**2)
    if avg_speed_pixels is None:
        ds_tot = np.sum(np.linalg.norm(np.diff(raceline_ref, axis=0), axis=1))
        avg_speed_pixels = ds_tot / ((state_steps-1)*dt + 1e-12)
    v_min = w.v_min_factor * avg_speed_pixels
    J_vmin = w.w_vmin * np.sum(np.maximum(0.0, v_min - states[:, 3])**2)

    # 终端（位置+朝向）
    term_pos = np.sum((states[-1, :2] - raceline_ref[-1])**2)
    dtheta_T = (states[-1, 2] - theta_ref[-1] + np.pi) % (2*np.pi) - np.pi
    J_term = w.w_terminal * (term_pos + 0.1 * dtheta_T**2)

    # 安全（节点 + 中点）
    J_safe = 0.0
    if sdf_map is not None and w_safety > 0:
        sdf_nodes = sample_sdf_bilinear(sdf_map, states[:, :2])
        viol_nodes = np.maximum(0.0, safety_margin_pix - sdf_nodes)
        mid_xy = 0.5 * (states[:-1, :2] + states[1:, :2])
        sdf_mid = sample_sdf_bilinear(sdf_map, mid_xy)
        viol_mid = np.maximum(0.0, safety_margin_pix - sdf_mid)
        J_safe = np.sum(viol_nodes**2) + np.sum(viol_mid**2)


    J = (J_path + J_snap + J_head + J_u + Ju_smooth + J_drate + J_jerk
         + J_v + J_vmin + J_term + w_safety * J_safe)
    return float(J)

# =========================
# 续接求解（分阶段权重表）
# =========================
def continuation_solve_over(z0: np.ndarray,
                            x0: np.ndarray,
                            raceline_ref: np.ndarray,
                            theta_ref: np.ndarray,
                            v_ref: np.ndarray,
                            cfg: OCPConfig,
                            v_max: float,
                            L_pix: float,
                            sdf_map: np.ndarray):
    from scipy.optimize import minimize

    control_steps = cfg.state_steps - 1
    total_time = control_steps * cfg.dt
    path_len_px = np.sum(np.linalg.norm(np.diff(raceline_ref, axis=0), axis=1))
    avg_speed_pixels = path_len_px / (total_time + 1e-12)

    # 六个阶段：前 3 阶强吸附贴线；后 3 阶逐步加回安全与常规项
    stages = [
        dict(w_safety_scale=0.00, w_lat=80, w_lon=0.5, w_head=0.5,
             w_v=0.0, v_min_factor=0.05, w_snap=60.0, snap_delta=4.0,
             w_smooth=0.03, w_drate=1.5, w_jerk=1.0),
        dict(w_safety_scale=0.00, w_lat=60, w_lon=0.8, w_head=1.0,
             w_v=0.0, v_min_factor=0.08, w_snap=40.0, snap_delta=3.5,
             w_smooth=0.035, w_drate=2.0, w_jerk=1.5),
        dict(w_safety_scale=0.01, w_lat=40, w_lon=1.2, w_head=2.0,
             w_v=0.002, v_min_factor=0.12, w_snap=20.0, snap_delta=3.0,
             w_smooth=0.04, w_drate=3.0, w_jerk=2.0),
        dict(w_safety_scale=0.10, w_lat=28, w_lon=1.6, w_head=3.0,
             w_v=0.004, v_min_factor=0.16, w_snap=8.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=4.0, w_jerk=2.0),
        dict(w_safety_scale=0.50, w_lat=20, w_lon=2.0, w_head=4.0,
             w_v=0.006, v_min_factor=0.18, w_snap=0.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=5.0, w_jerk=2.0),
        dict(w_safety_scale=1.00, w_lat=18, w_lon=2.0, w_head=6.0,
             w_v=0.01, v_min_factor=0.20, w_snap=0.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=6.0, w_jerk=2.0),
    ]
    # stages = [
    #     dict(w_safety_scale=0.00, w_lat=80, w_lon=0.5, w_head=0.5,
    #          w_v=0.0, v_min_factor=0.05, w_snap=60.0, snap_delta=4.0,
    #          w_smooth=0.03, w_drate=1.5, w_jerk=1.0),
    #     dict(w_safety_scale=0.05, w_lat=60, w_lon=0.8, w_head=1.0,
    #          w_v=0.0, v_min_factor=0.08, w_snap=40.0, snap_delta=3.5,
    #          w_smooth=0.035, w_drate=2.0, w_jerk=1.5),
    #     dict(w_safety_scale=1.00, w_lat=40, w_lon=1.2, w_head=2.0,
    #          w_v=0.002, v_min_factor=0.12, w_snap=20.0, snap_delta=3.0,
    #          w_smooth=0.04, w_drate=3.0, w_jerk=2.0),
    # ]

    # 安全 schedule 与阶段对齐（多了截断，少了重复）
    sched = list(cfg.w_safety_schedule)
    if len(sched) < len(stages):
        sched += [sched[-1]]*(len(stages)-len(sched))
    elif len(sched) > len(stages):
        stages += [stages[-1]]*(len(sched)-len(stages))

    res_last = None
    for i, (w_s, st) in enumerate(zip(sched, stages), 1):
        w_now = OCPWeights()
        w_now.w_lat = st['w_lat']; w_now.w_lon = st['w_lon']; w_now.w_head = st['w_head']
        w_now.w_v   = st['w_v'];   w_now.v_min_factor = st['v_min_factor']
        w_now.w_snap = st['w_snap']; w_now.snap_delta = st['snap_delta']
        w_now.w_smooth = st['w_smooth']; w_now.w_drate = st['w_drate']; w_now.w_jerk = st['w_jerk']

        w_eff_safety = w_s * st['w_safety_scale']

        print(f"\n--- Stage {i}/{len(stages)}  w_safety_eff={w_eff_safety:.1f}  "
              f"w_lat={w_now.w_lat:.1f}  snap={w_now.w_snap:.1f}")

        res_last = minimize(
            ocp_cost_directional, z0,
            args=(x0, raceline_ref, theta_ref, v_ref, cfg.dt,
                  cfg.state_steps, control_steps, cfg.a_max, cfg.delta_max, L_pix, v_max,
                  sdf_map, cfg.safety_margin_pix, w_eff_safety, cfg.delta_rate_max,
                  avg_speed_pixels, w_now),
            method="L-BFGS-B",
            options={"maxiter": 300, "maxfun": 150000, "ftol": 1e-6, "gtol": 1e-5, "maxls": 50, "disp": True}
            #options={"maxiter": 200, "maxfun": 90000, "ftol": 3e-6, "gtol": 5e-5, "maxls": 35, "disp": True}

        )
        z0 = res_last.x

    u_opt = unpack_controls(res_last.x, control_steps, cfg.a_max, cfg.delta_max)
    states_opt = simulate_vehicle_bc(x0, u_opt, cfg.dt, cfg.state_steps,
                                     L_pix=L_pix, v_max=v_max, delta_rate_max=cfg.delta_rate_max)
    return u_opt, states_opt, {"result": res_last}

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Inverse-only OCP runner (+ optional MPPI)")
    ap.add_argument("--map", type=str, default="nuerburgring_segment_map.npz")
    ap.add_argument("--state_steps", type=int, default=101)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--a_max", type=float, default=35.0)
    ap.add_argument("--delta_max", type=float, default=1.0)
    ap.add_argument("--safety_pix", type=float, default=10.0)

    ap.add_argument("--do_full", action="store_true")
    ap.add_argument("--do_segment", action="store_true")
    ap.add_argument("--full_from_boxes", action="store_true")
    ap.add_argument("--run_mppi", action="store_true")
    ap.add_argument("--mppi_iters", type=int, default=30)
    ap.add_argument("--mppi_N", type=int, default=2048)

    ap.add_argument("--start_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[349.3, 572.1, 382.7, 561.9])
    ap.add_argument("--end_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[120.70, 153.80, 95.90, 178.50])

    ap.add_argument("--init_a_frac", type=float, default=0.5)

    ap.add_argument("--output_file", type=str, default="inverse_norm.npz", help="Path to save the output trajectory data.")

    args = ap.parse_args()
    if not args.do_full and not args.do_segment:
        args.do_segment = True

    md: MapData = load_map(args.map)
    cfg = OCPConfig(
        state_steps=args.state_steps,
        dt=args.dt,
        a_max=args.a_max,
        delta_max=args.delta_max,
        safety_margin_pix=args.safety_pix
    )
    control_steps = cfg.state_steps - 1

    sdf_map, _ = build_sdf_from_grid(md.grid_map)
    L_pix = cfg.wheelbase_m * md.resolution

    start_mid, end_mid = box_midpoints(args)

    # ===== FULL =====
    if args.do_full:
        print("\n[Full-Inverse] building reference...")
        raceline_ref, theta_ref, v_ref, v_max, path_len_px = build_full_inverse_reference(
            md, cfg, args.full_from_boxes, start_mid, end_mid
        )
        dx, dy = raceline_ref[1, 0] - raceline_ref[0, 0], raceline_ref[1, 1] - raceline_ref[0, 1]
        theta0 = np.arctan2(dy, dx)
        x0 = np.array([raceline_ref[0, 0], raceline_ref[0, 1], theta0, 0.0], dtype=np.float64)
        z0 = np.zeros(control_steps * 2, dtype=np.float64)

        print("[Full-Inverse] Solving OCP (staged directional cost)...")
        u_opt, states_opt, _ = continuation_solve_over(
            z0, x0, raceline_ref, theta_ref, v_ref, cfg, v_max, L_pix, sdf_map
        )
        plot_results(states_opt, u_opt, raceline_ref, v_ref,
                     md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max)

        final_states, final_controls = states_opt, u_opt
        if args.run_mppi:
            print("[Full-Inverse] MPPI polish (light)...")
            total_time = control_steps * cfg.dt
            avg_speed_pixels = (path_len_px / (total_time + 1e-12))
            u_mppi, states_mppi, _ = mppi_local_refine(
                u_opt, x0, cfg.dt, L_pix, v_max, cfg.delta_rate_max,
                raceline_ref, theta_ref, v_ref, sdf_map, cfg.safety_margin_pix,
                cfg.a_max, cfg.delta_max,
                iters=max(20, args.mppi_iters), N=min(1024, args.mppi_N),
                sigma_z_a=0.10, sigma_z_delta=0.06, rho=0.92, lam=7000.0,
                z_tr_clip=0.15, z_step_clip=0.06, step_size=0.6, smooth_passes=3,
                avg_speed_pixels=avg_speed_pixels
            )
            plot_mppi_results(states_opt, u_opt, states_mppi, u_mppi,
                              raceline_ref, v_ref, md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max)
            final_states, final_controls = states_mppi, u_mppi

        print(f"\n[Full-Inverse] Saving trajectory to {args.output_file}...")
        np.savez(
            args.output_file,
            states=final_states,
            controls=final_controls,
            raceline=raceline_ref,
            start_point=raceline_ref[0],
            end_point=raceline_ref[-1],
            x0=x0
        )

    # ===== SEGMENT =====
    if args.do_segment:
        print("\n[Segment-Inverse] sampling boxes and swapping start/end...")
        rng = np.random.default_rng(None)
        START_A = np.array(args.start_box[:2]); START_B = np.array(args.start_box[2:])
        END_A   = np.array(args.end_box[:2]);   END_B   = np.array(args.end_box[2:])
        start_sample = sample_in_box(START_A, START_B, rng)
        end_sample   = sample_in_box(END_A,   END_B,   rng)
        start_sample, end_sample = end_sample, start_sample
        print(f"[Segment-Inverse] start={start_sample}, end={end_sample}")

        idx_s = nearest_index(md.raceline_grid, start_sample)
        idx_e = nearest_index(md.raceline_grid, end_sample)
        if idx_s == idx_e:
            idx_e = min(idx_s + 1, len(md.raceline_grid) - 1)
        sub_poly = (md.raceline_grid[idx_s:idx_e+1]
                    if idx_s < idx_e else md.raceline_grid[idx_e:idx_s+1][::-1])

        raceline_seg, seg_len_px = resample_polyline_to_N(sub_poly, cfg.state_steps)
        raceline_seg[0]  = start_sample
        raceline_seg[-1] = end_sample
        theta_ref_seg = robust_theta_from_polyline(raceline_seg)

        v_ref_seg, v_max_seg = curvature_capped_speed_profile(
            raceline_seg, theta_ref_seg, cfg, seg_len_px, v_factor=3.0, alpha_lat=0.45
        )

        dx, dy = raceline_seg[1, 0] - raceline_seg[0, 0], raceline_seg[1, 1] - raceline_seg[0, 1]
        theta0_seg = np.arctan2(dy, dx)
        x0_seg = np.array([raceline_seg[0, 0], raceline_seg[0, 1], theta0_seg, 0.0], dtype=np.float64)
        z0_seg = nonzero_accel_init(control_steps, a_fraction=args.init_a_frac)

        print("[Segment-Inverse] Solving OCP (staged directional cost)...")
        u_seg, states_seg, _ = continuation_solve_over(
            z0_seg, x0_seg, raceline_seg, theta_ref_seg, v_ref_seg, cfg, v_max_seg, L_pix, sdf_map
        )
        plot_results(states_seg, u_seg, raceline_seg, v_ref_seg,
                     md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max_seg)

        final_states, final_controls = states_seg, u_seg
        if args.run_mppi:
            print("[Segment-Inverse] MPPI polish (light)...")
            total_time = control_steps * cfg.dt
            avg_speed_seg = seg_len_px / (total_time + 1e-12)
            u_mppi, states_mppi, _ = mppi_local_refine(
                u_seg, x0_seg, cfg.dt, L_pix, v_max_seg, cfg.delta_rate_max,
                raceline_seg, theta_ref_seg, v_ref_seg, sdf_map, cfg.safety_margin_pix,
                cfg.a_max, cfg.delta_max,
                iters=max(20, args.mppi_iters), N=min(1024, args.mppi_N),
                sigma_z_a=0.10, sigma_z_delta=0.06, rho=0.92, lam=7000.0,
                z_tr_clip=0.15, z_step_clip=0.06, step_size=0.6, smooth_passes=3,
                avg_speed_pixels=avg_speed_seg
            )
            plot_mppi_results(states_seg, u_seg, states_mppi, u_mppi,
                              raceline_seg, v_ref_seg, md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max_seg)
            final_states, final_controls = states_mppi, u_mppi

        print(f"\n[Segment-Inverse] Saving trajectory to {args.output_file}...")
        np.savez(
            args.output_file,
            states=final_states,
            controls=final_controls,
            raceline=raceline_seg,
            start_point=start_sample,
            end_point=end_sample,
            x0=x0_seg
        )

if __name__ == "__main__":
    main()

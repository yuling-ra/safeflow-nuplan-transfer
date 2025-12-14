# run_inverse_batch.py
# -*- coding: utf-8 -*-
"""
Batch inverse runner (staged-OCP, random start/end on segments, goal/tail attraction,
progress term, normalized losses, reference-tracker warm start; no MPPI)

关键改动：
- 参考跟踪器 warm start：Pure Pursuit + 曲率前馈 + a=(v_ref_{k+1}-v)/dt → u_init → z0
- 新增进度项 J_prog：用“末端在参考线上的索引”逼近终点索引
- 所有项按步数归一化（/N 或 /(N-1)），避免量级悬殊
- 安全权重设上限；仅后期升高；终点吸力仅后两阶段较大
"""

import argparse, os, time, math, uuid
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from raceline_core import (
    MapData, OCPConfig, load_map, build_sdf_from_grid,
    sample_sdf_bilinear, simulate_vehicle_bc, unpack_controls,
    nearest_index, resample_polyline_to_N
)

# -----------------------------
# 小工具
# -----------------------------
def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b

def sample_on_segment(A: np.ndarray, B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.0, 1.0)
    return lerp(A, B, t)

def robust_theta_from_polyline(poly: np.ndarray):
    dxy = np.diff(poly, axis=0, append=poly[-1:])
    if len(poly) >= 2:
        dxy[-1] = poly[-1] - poly[-2]
    return np.arctan2(dxy[:, 1], dxy[:, 0])

def ensure_min_nodes(raceline: np.ndarray, i: int, j: int, min_nodes: int) -> Tuple[int, int]:
    n = len(raceline)
    if i <= j:
        need = min_nodes - (j - i + 1)
        if need > 0:
            extra_left = min(i, need // 2 + need % 2)
            extra_right = min(n - 1 - j, need - extra_left)
            i -= extra_left; j += extra_right
    else:
        need = min_nodes - (i - j + 1)
        if need > 0:
            extra_right = min(n - 1 - i, need // 2 + need % 2)
            extra_left = min(j, need - extra_right)
            i += extra_right; j -= extra_left
    i = max(0, min(i, n - 1)); j = max(0, min(j, n - 1))
    if i == j: j = min(i + 1, n - 1)
    return i, j

# --------- 曲率感知 v_ref ----------
def curvature_capped_speed_profile(raceline_ref: np.ndarray,
                                   theta_ref: np.ndarray,
                                   cfg: OCPConfig,
                                   path_len_px: float,
                                   v_factor: float = 3.0,
                                   alpha_lat: float = 0.45):
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

# -----------------------------
# 参考跟踪器 Warm Start（生成可跑的 u_init）
# -----------------------------
def lowpass(u, passes=2):
    if passes <= 0: return u
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    out = u.copy()
    for _ in range(passes):
        for j in range(2):
            out[:, j] = np.convolve(out[:, j], k, mode='same')
    return out

def warmstart_controls(raceline_ref, theta_ref, v_ref, x0, cfg: OCPConfig, L_pix: float):
    T = cfg.state_steps - 1
    u = np.zeros((T, 2), dtype=np.float64)
    st = np.zeros((cfg.state_steps, 4), dtype=np.float64)
    st[0] = x0
    look = 6  # Pure Pursuit 预瞄步
    for k in range(T):
        # 目标速度简单前馈/反馈
        v_des = v_ref[min(k+1, len(v_ref)-1)]
        a = np.clip((v_des - st[k, 3]) / cfg.dt, -0.7*cfg.a_max, 0.7*cfg.a_max)

        # 曲率前馈 + PP 横向：delta ≈ atan(L * kappa)
        i_ff = min(k+1, len(theta_ref)-1)
        dth = theta_ref[i_ff] - theta_ref[max(i_ff-1, 0)]
        ds = np.linalg.norm(raceline_ref[i_ff] - raceline_ref[max(i_ff-1, 0)]) + 1e-6
        kappa = dth / ds
        delta_ff = np.arctan(L_pix * kappa)

        i_pp = min(k+look, len(raceline_ref)-1)
        vec_target = raceline_ref[i_pp] - st[k, :2]
        e_y = -np.sin(st[k, 2]) * vec_target[0] + np.cos(st[k, 2]) * vec_target[1]
        delta_pp = np.arctan2(2*L_pix*e_y, max(1e-3, st[k, 3]*st[k, 3]))

        delta = np.clip(0.6*delta_ff + 0.4*delta_pp, -cfg.delta_max, cfg.delta_max)
        u[k] = [a, delta]

        # 前滚
        v_next = np.clip(st[k, 3] + cfg.dt * a, 0.0, np.inf)
        th_next = st[k, 2] + cfg.dt * (v_next / (L_pix + 1e-12)) * np.tan(delta)
        x_next = st[k, 0] + cfg.dt * v_next * np.cos(st[k, 2])
        y_next = st[k, 1] + cfg.dt * v_next * np.sin(st[k, 2])
        st[k+1] = [x_next, y_next, th_next, v_next]

    u = lowpass(u, passes=2)
    u[:, 0] = np.clip(u[:, 0], -cfg.a_max, cfg.a_max)
    u[:, 1] = np.clip(u[:, 1], -cfg.delta_max, cfg.delta_max)

    z0 = np.zeros_like(u)
    z0[:, 0] = np.arctanh(np.clip(u[:, 0] / cfg.a_max, -0.999999, 0.999999))
    z0[:, 1] = np.arctanh(np.clip(u[:, 1] / cfg.delta_max, -0.999999, 0.999999))
    return z0.reshape(-1)

# =========================
# OCP 代价（归一化 + 进度 + 目标吸力）
# =========================
def _huber(x, d):  # smooth L1
    ax = np.abs(x)
    return np.where(ax <= d, 0.5*x*x, d*(ax - 0.5*d))

@dataclass
class OCPWeights:
    w_lat: float = 18.0
    w_lon: float = 2.0
    w_head: float = 6.0
    w_u: float = 1e-3
    w_smooth: float = 0.05
    w_v: float = 0.01
    w_terminal: float = 80.0
    w_drate: float = 6.0
    w_jerk: float = 2.0
    v_min_factor: float = 0.20
    w_vmin: float = 0.03
    w_snap: float = 0.0
    snap_delta: float = 3.0
    w_goal: float = 0.0
    w_goal_tail: float = 0.0
    K_tail: int = 12
    w_prog: float = 50.0

def ocp_cost(z_flat, x0, raceline_ref, theta_ref, v_ref,
             dt, state_steps, control_steps,
             a_max, delta_max, L_pix, v_max,
             sdf_map=None, safety_margin_pix=8.0, w_safety=40.0,
             delta_rate_max=2.5,
             avg_speed_pixels: float = None,
             w: OCPWeights = OCPWeights(),
             goal_xy: np.ndarray = None):

    u = unpack_controls(z_flat, control_steps, a_max, delta_max)
    states = simulate_vehicle_bc(x0, u, dt, state_steps, L_pix=L_pix, v_max=v_max,
                                 delta_rate_max=delta_rate_max)
    N = state_steps
    Nu = control_steps

    pos_err = states[:, :2] - raceline_ref
    c = np.cos(theta_ref); s = np.sin(theta_ref)
    t = np.stack([c, s], axis=1); n = np.stack([-s, c], axis=1)
    e_lon = np.einsum('ij,ij->i', pos_err, t)
    e_lat = np.einsum('ij,ij->i', pos_err, n)

    J_path = (w.w_lat * np.sum(e_lat**2) + w.w_lon * np.sum(e_lon**2)) / N
    J_snap = (w.w_snap * np.sum(_huber(e_lat, w.snap_delta))) / N

    dth = (states[:, 2] - theta_ref + np.pi) % (2*np.pi) - np.pi
    J_head = (w.w_head * np.sum(dth**2)) / N

    J_u = (w.w_u * np.sum(u**2)) / Nu
    Ju_smooth = (w.w_smooth * np.sum(np.diff(u, axis=0)**2)) / max(1, Nu-1)
    J_drate = (w.w_drate * np.sum(np.diff(u[:, 1])**2)) / max(1, Nu-1)
    J_jerk  = (w.w_jerk  * np.sum(np.diff(u[:, 0])**2)) / max(1, Nu-1)

    J_v = (w.w_v * np.sum((states[:, 3] - v_ref)**2)) / N
    if avg_speed_pixels is None:
        ds_tot = np.sum(np.linalg.norm(np.diff(raceline_ref, axis=0), axis=1))
        avg_speed_pixels = ds_tot / ((state_steps-1)*dt + 1e-12)
    v_min = w.v_min_factor * avg_speed_pixels
    J_vmin = (w.w_vmin * np.sum(np.maximum(0.0, v_min - states[:, 3])**2)) / N

    # 终端“走廊”
    pT = raceline_ref[-1]
    tT = np.array([np.cos(theta_ref[-1]), np.sin(theta_ref[-1])])
    nT = np.array([-tT[1], tT[0]])
    eT = states[-1, :2] - pT
    eT_lat = np.dot(eT, nT); eT_lon = np.dot(eT, tT)
    J_term = w.w_terminal * (eT_lat**2 + 0.3*np.maximum(0.0, -eT_lon)**2 +
                             0.1 * ((states[-1, 2] - theta_ref[-1] + np.pi) % (2*np.pi) - np.pi)**2)  # 不归一化留强度

    # 终点吸力（末端 & 尾段）
    J_goal = 0.0
    if goal_xy is not None:
        if w.w_goal > 0.0:
            J_goal += w.w_goal * np.sum((states[-1, :2] - goal_xy)**2)
        if w.w_goal_tail > 0.0 and w.K_tail > 0:
            K = min(w.K_tail, N)
            idx0 = N - K
            weights = np.linspace(0.2, 1.0, K)
            diffs = states[idx0:, :2] - goal_xy[None, :]
            J_goal += (w.w_goal_tail * np.sum(weights[:, None] * diffs**2)) / K

    # 进度项：末端在参考线的最近索引越接近结尾越好
    idx = np.argmin(np.sum((raceline_ref - states[-1, :2])**2, axis=1))
    J_prog = w.w_prog * ((len(raceline_ref) - 1 - idx)**2) / (len(raceline_ref)**2)

    # 安全：节点 + 中点
    J_safe = 0.0
    if sdf_map is not None and w_safety > 0:
        sdf_nodes = sample_sdf_bilinear(sdf_map, states[:, :2])
        viol_nodes = np.maximum(0.0, safety_margin_pix - sdf_nodes)
        mid_xy = 0.5 * (states[:-1, :2] + states[1:, :2])
        sdf_mid = sample_sdf_bilinear(sdf_map, mid_xy)
        viol_mid = np.maximum(0.0, safety_margin_pix - sdf_mid)
        J_safe = (np.sum(viol_nodes**2) + np.sum(viol_mid**2)) / (N + (N-1))

    J = (J_path + J_snap + J_head + J_u + Ju_smooth + J_drate + J_jerk
         + J_v + J_vmin + J_term + J_goal + J_prog + w_safety * J_safe)
    return float(J)

# -----------------------------
# 续接（分阶段）
# -----------------------------
def continuation_solve_staged(z0: np.ndarray,
                              x0: np.ndarray,
                              raceline_ref: np.ndarray,
                              theta_ref: np.ndarray,
                              v_ref: np.ndarray,
                              cfg: OCPConfig,
                              v_max: float,
                              L_pix: float,
                              sdf_map: np.ndarray,
                              goal_xy: np.ndarray):
    from scipy.optimize import minimize

    control_steps = cfg.state_steps - 1
    total_time = control_steps * cfg.dt
    path_len_px = np.sum(np.linalg.norm(np.diff(raceline_ref, axis=0), axis=1))
    avg_speed_pixels = path_len_px / (total_time + 1e-12)

    # 限制安全权重的“有效上限”，避免过早压死
    SAFETY_CAP = 3.0e5

    stages = [
        dict(w_safety_scale=0.00, w_lat=80, w_lon=0.6, w_head=0.6,
             w_v=0.0, v_min_factor=0.05, w_snap=60.0, snap_delta=4.0,
             w_smooth=0.03, w_drate=1.5, w_jerk=1.0, w_goal=2.0,  w_goal_tail=0.0, K_tail=0,  w_prog=80),
        dict(w_safety_scale=0.02, w_lat=60, w_lon=0.9, w_head=1.2,
             w_v=0.0, v_min_factor=0.08, w_snap=40.0, snap_delta=3.5,
             w_smooth=0.035, w_drate=2.0, w_jerk=1.5, w_goal=6.0,  w_goal_tail=0.0, K_tail=0,  w_prog=80),
        dict(w_safety_scale=0.05, w_lat=42, w_lon=1.2, w_head=2.0,
             w_v=0.002, v_min_factor=0.12, w_snap=20.0, snap_delta=3.0,
             w_smooth=0.04, w_drate=3.0, w_jerk=2.0, w_goal=12.0, w_goal_tail=4.0, K_tail=8,  w_prog=60),
        dict(w_safety_scale=0.15, w_lat=30, w_lon=1.5, w_head=3.0,
             w_v=0.004, v_min_factor=0.16, w_snap=8.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=4.0, w_jerk=2.0, w_goal=28.0, w_goal_tail=10.0, K_tail=10, w_prog=45),
        dict(w_safety_scale=0.45, w_lat=22, w_lon=2.0, w_head=4.0,
             w_v=0.006, v_min_factor=0.18, w_snap=0.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=5.0, w_jerk=2.0, w_goal=70.0, w_goal_tail=18.0, K_tail=12, w_prog=35),
        dict(w_safety_scale=1.00, w_lat=18, w_lon=2.0, w_head=6.0,
             w_v=0.01, v_min_factor=0.20, w_snap=0.0,  snap_delta=3.0,
             w_smooth=0.05, w_drate=6.0, w_jerk=2.0, w_goal=120.0, w_goal_tail=30.0, K_tail=14, w_prog=30),
    ]

    sched = list(cfg.w_safety_schedule)
    if len(sched) < len(stages):
        sched += [sched[-1]]*(len(stages)-len(sched))
    elif len(sched) > len(stages):
        stages += [stages[-1]]*(len(sched)-len(stages))

    res_last = None
    for i, (w_s_base, st) in enumerate(zip(sched, stages), 1):
        w_now = OCPWeights(
            w_lat=st['w_lat'], w_lon=st['w_lon'], w_head=st['w_head'],
            w_u=1e-3, w_smooth=st['w_smooth'], w_v=st['w_v'], w_terminal=80.0,
            w_drate=st['w_drate'], w_jerk=st['w_jerk'],
            v_min_factor=st['v_min_factor'], w_vmin=0.03,
            w_snap=st['w_snap'], snap_delta=st['snap_delta'],
            w_goal=st['w_goal'], w_goal_tail=st['w_goal_tail'], K_tail=st['K_tail'],
            w_prog=st['w_prog']
        )
        w_eff_safety = min(SAFETY_CAP, w_s_base * st['w_safety_scale'])

        print(f"\n--- Stage {i}/{len(stages)}  w_safety_eff={w_eff_safety:.1f}  "
              f"w_lat={w_now.w_lat:.1f}  snap={w_now.w_snap:.1f}  "
              f"goal={w_now.w_goal:.1f} tail={w_now.w_goal_tail:.1f}/{w_now.K_tail}")

        res_last = minimize(
            ocp_cost, z0,
            args=(x0, raceline_ref, theta_ref, v_ref, cfg.dt,
                  cfg.state_steps, control_steps, cfg.a_max, cfg.delta_max, L_pix, v_max,
                  sdf_map, cfg.safety_margin_pix, w_eff_safety, cfg.delta_rate_max,
                  avg_speed_pixels, w_now, goal_xy),
            method="L-BFGS-B",
            options={"maxiter": 300, "maxfun": 150000, "ftol": 1e-6, "gtol": 1e-5, "maxls": 50, "disp": True}
        )
        z0 = res_last.x

    u_opt = unpack_controls(res_last.x, control_steps, cfg.a_max, cfg.delta_max)
    states_opt = simulate_vehicle_bc(x0, u_opt, cfg.dt, cfg.state_steps,
                                     L_pix=L_pix, v_max=v_max, delta_rate_max=cfg.delta_rate_max)
    return u_opt, states_opt

# -----------------------------
# 增量式存储
# -----------------------------
def save_npz(path: str, buf: list, meta: dict):
    if not buf: return
    actions = np.stack([b["u"] for b in buf], axis=0).astype(np.float64)
    states  = np.stack([b["states"] for b in buf], axis=0).astype(np.float64)
    x0s     = np.stack([b["x0"] for b in buf], axis=0).astype(np.float64)
    starts  = np.stack([b["start"] for b in buf], axis=0).astype(np.float64)
    ends    = np.stack([b["goal"] for b in buf], axis=0).astype(np.float64)
    refs    = np.stack([b["raceline"] for b in buf], axis=0).astype(np.float64)
    vmaxs   = np.array([b["v_max"] for b in buf], dtype=np.float64)

    if os.path.isfile(path):
        try:
            old = np.load(path, allow_pickle=False)
            if "actions" in old:
                actions = np.concatenate([old["actions"], actions], axis=0)
                states  = np.concatenate([old["states"],  states],  axis=0)
                x0s     = np.concatenate([old["x0s"],     x0s],     axis=0)
                starts  = np.concatenate([old["starts"],  starts],  axis=0)
                ends    = np.concatenate([old["ends"],    ends],    axis=0)
                refs    = np.concatenate([old["refs"],    refs],    axis=0)
                vmaxs   = np.concatenate([old["vmaxs"],   vmaxs],   axis=0)
            old.close()
        except Exception as e:
            print(f"[warn] Could not load existing dataset at {path}, creating a new one. Error: {e}")

    np.savez_compressed(
        path,
        actions=actions, states=states, x0s=x0s, starts=starts, ends=ends, refs=refs, vmaxs=vmaxs,
        **meta
    )


# -----------------------------
# 单样本（供并行）
# -----------------------------
def run_one_sample(args_pack: Tuple) -> Dict[str, Any]:
    (seed, md, cfg, sdf_map, L_pix, start_A, start_B, end_A, end_B, min_nodes) = args_pack
    rng = np.random.default_rng(seed)
    start_xy = sample_on_segment(start_A, start_B, rng)
    goal_xy  = sample_on_segment(end_A, end_B, rng)

    idx_s = nearest_index(md.raceline_grid, start_xy)
    idx_g = nearest_index(md.raceline_grid, goal_xy)
    idx_s, idx_g = ensure_min_nodes(md.raceline_grid, idx_s, idx_g, min_nodes=min_nodes)
    sub_poly = (md.raceline_grid[idx_s:idx_g+1]
                if idx_s <= idx_g else md.raceline_grid[idx_g:idx_s+1][::-1])

    raceline_seg, seg_len_px = resample_polyline_to_N(sub_poly, cfg.state_steps)
    raceline_seg[0] = start_xy  # 锁起点
    theta_ref = robust_theta_from_polyline(raceline_seg)

    control_steps = cfg.state_steps - 1
    v_ref, v_max = curvature_capped_speed_profile(raceline_seg, theta_ref, cfg, seg_len_px,
                                                  v_factor=3.0, alpha_lat=0.45)
    # 初始状态
    dx, dy = raceline_seg[1, 0] - raceline_seg[0, 0], raceline_seg[1, 1] - raceline_seg[0, 1]
    theta0 = math.atan2(dy, dx)
    x0 = np.array([start_xy[0], start_xy[1], theta0, 0.0], dtype=np.float64)

    # Warm start（关键！）
    z0 = warmstart_controls(raceline_seg, theta_ref, v_ref, x0, cfg, L_pix)

    # 续接 OCP
    u_opt, states_opt = continuation_solve_staged(
        z0, x0, raceline_seg, theta_ref, v_ref, cfg, v_max, L_pix, sdf_map, goal_xy
    )
    return dict(
        start=start_xy, goal=goal_xy, x0=x0,
        raceline=raceline_seg, theta_ref=theta_ref,
        u=u_opt, states=states_opt, v_ref=v_ref, v_max=v_max
    )

# -----------------------------
# 主程序
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Batch inverse OCP runner (random start/end on segments)")
    ap.add_argument("--map", type=str, default="nuerburgring_segment_map.npz")
    ap.add_argument("--state_steps", type=int, default=101)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--a_max", type=float, default=35.0)
    ap.add_argument("--delta_max", type=float, default=1.0)
    ap.add_argument("--safety_pix", type=float, default=10.0)

    ap.add_argument("--start", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[120.70, 153.80, 95.90, 178.50])   # 起点段（锁定）
    ap.add_argument("--end",   type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[349.3, 572.1, 382.7, 561.9])     # 终点段（吸力目标）

    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--workers", type=str, default="auto")
    ap.add_argument("--plot_all", action="store_true")
    ap.add_argument("--min_nodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible experiments")

    random_id = str(uuid.uuid4())[:8]
    default_save_path = f"./outputs/ocp_dataset_inverse_{random_id}.npz"
    ap.add_argument("--save", type=str, default=default_save_path, help="Path to save the dataset")
    ap.add_argument("--save_every", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    out_path = args.save

    md: MapData = load_map(args.map)
    cfg = OCPConfig(
        state_steps=args.state_steps,
        dt=args.dt,
        a_max=args.a_max,
        delta_max=args.delta_max,
        safety_margin_pix=args.safety_pix
    )
    sdf_map, _ = build_sdf_from_grid(md.grid_map)
    L_pix = cfg.wheelbase_m * md.resolution

    control_steps = cfg.state_steps - 1
    meta = dict(
        dt=np.float64(cfg.dt),
        state_steps=np.int64(cfg.state_steps),
        control_steps=np.int64(control_steps),
        a_max=np.float64(cfg.a_max),
        delta_max=np.float64(cfg.delta_max),
        resolution=np.float64(md.resolution),
        safety_margin_pix=np.float64(cfg.safety_margin_pix),
        map_shape=np.array(md.grid_map.shape, dtype=np.int32),
        map_grid=md.grid_map.astype(np.uint8),
    )

    start_A = np.array(args.start[:2], dtype=float)
    start_B = np.array(args.start[2:], dtype=float)
    end_A   = np.array(args.end[:2], dtype=float)
    end_B   = np.array(args.end[2:], dtype=float)

    if args.workers == "auto":
        n_workers = max(1, cpu_count())
    else:
        n_workers = max(1, int(args.workers))

    # 新：支持可复现实验的 --seed；没有就用时间戳^pid 混合
    base_ss = np.random.SeedSequence(
        args.seed if hasattr(args, "seed") and args.seed is not None
        else (int(time.time() * 1e6) ^ os.getpid())
    )
    children = base_ss.spawn(args.samples)

    # 为每个 child 生成真正独立的 32-bit 种子
    unique_seeds = [int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children]

    job_args = [(s, md, cfg, sdf_map, L_pix, start_A, start_B, end_A, end_B, args.min_nodes)
                for s in unique_seeds]

    results: List[Dict[str, Any]] = []
    last_saved_count = 0

    print(f"Starting inverse OCP generation for {args.samples} samples with {n_workers} worker(s)...")
    print(f"Dataset will be saved to: {out_path}")
    t_start_gen = time.time()

    if n_workers == 1:
        for i, ja in enumerate(job_args, 1):
            results.append(run_one_sample(ja))
            if len(results) - last_saved_count >= args.save_every:
                save_npz(out_path, results[last_saved_count:], meta)
                print(f"[save] total={len(results)}/{args.samples} file={out_path} (+{len(results)-last_saved_count} saved)")
                last_saved_count = len(results)
    else:
        with Pool(processes=n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_one_sample, job_args), 1):
                results.append(res)
                if len(results) - last_saved_count >= args.save_every:
                    save_npz(out_path, results[last_saved_count:], meta)
                    print(f"[save] total={len(results)}/{args.samples} file={out_path} (+{len(results)-last_saved_count} saved)")
                    last_saved_count = len(results)

    if len(results) > last_saved_count:
        save_npz(out_path, results[last_saved_count:], meta)
        print(f"[save] total={len(results)}/{args.samples} file={out_path} (+{len(results)-last_saved_count} saved)")

    t_end_gen = time.time()
    total_gen_time = t_end_gen - t_start_gen
    n_gen = len(results)
    if n_gen > 0:
        avg_time = total_gen_time / n_gen
        print(f"\nFinished generating {n_gen} trajectories.")
        print(f"Total generation time: {total_gen_time:.2f} s")
        print(f"Average time per trajectory: {avg_time:.2f} s/traj")

    if args.plot_all:
        fig = plt.figure(figsize=(18, 12))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(md.grid_map, cmap='gray', origin='lower',
                  extent=[0, md.grid_map.shape[1], 0, md.grid_map.shape[0]])
        ax.plot(md.raceline_grid[:, 0], md.raceline_grid[:, 1], 'r--', lw=1.3, alpha=0.7, label='Reference')
        ax.plot([start_A[0], start_B[0]], [start_A[1], start_B[1]], 'g-', lw=3, alpha=0.6, label='Start segment')
        ax.plot([end_A[0], end_B[0]],     [end_A[1], end_B[1]],     'c-', lw=3, alpha=0.6, label='End segment')
        for r in results:
            ax.plot(r["states"][:, 0], r["states"][:, 1], '-', lw=2)
            ax.plot(r["start"][0], r["start"][1], 'g^', ms=7)
            ax.plot(r["goal"][0],  r["goal"][1],  'ro', ms=6, alpha=0.8)
        ax.legend(loc='upper left'); ax.set_aspect('equal'); ax.grid(True)
        ax.set_title(f"Batch inverse trajectories (N={len(results)})")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

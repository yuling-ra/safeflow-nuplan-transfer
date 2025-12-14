# raceline_core.py
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, map_coordinates
import matplotlib.pyplot as plt

# =========================
# 基础数据结构
# =========================
@dataclass
class MapData:
    grid_map: np.ndarray
    raceline_grid: np.ndarray
    resolution: float               # pixels per meter
    translation_offset: np.ndarray  # (2,)

@dataclass
class OCPConfig:
    state_steps: int = 101
    dt: float = 0.2
    a_max: float = 35.0
    delta_max: float = 1.0
    safety_margin_pix: float = 10.0
    #w_safety_schedule: Tuple[float, ...] = (100.0, 800.0, 3200.0, 12800.0, 51200.0)
    w_safety_schedule: Tuple[float, ...] = (1000.0, 4000.0, 16000.0, 64000.0, 256000.0, 1000000.0)
    wheelbase_m: float = 2.7  # 车辆轴距（米）
    delta_rate_max: Optional[float] = 2.5

# =========================
# IO 与参考构造
# =========================
def load_map(map_path: str) -> MapData:
    if not os.path.exists(map_path):
        print(f"[warn] Map file not found: {map_path}. Using dummy map.")
        grid_map = np.zeros((100, 100), dtype=np.uint8)
        raceline_grid = np.vstack([np.linspace(10, 90, 100), np.linspace(10, 90, 100)]).T
        resolution = 1.0
        translation_offset = np.array([0.0, 0.0])
    else:
        data = np.load(map_path, allow_pickle=False)
        grid_map = data["grid_map"]
        raceline_grid = data["raceline_grid"]
        resolution = float(data["resolution"])
        translation_offset = data["translation_offset"]
        data.close()
        print(f"[ok] Map loaded: grid={grid_map.shape}, raceline_pts={len(raceline_grid)}")
    return MapData(grid_map, raceline_grid, resolution, translation_offset)

def build_reference_from_raceline(raceline_grid: np.ndarray, N: int):
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(raceline_grid, axis=0), axis=1))))
    fx = interp1d(s, raceline_grid[:, 0], kind='cubic')
    fy = interp1d(s, raceline_grid[:, 1], kind='cubic')
    s_new = np.linspace(0.0, s[-1], N)
    ref = np.vstack([fx(s_new), fy(s_new)]).T
    dxy = np.diff(ref, axis=0, append=ref[-1:])
    theta_ref = np.arctan2(dxy[:, 1], dxy[:, 0])
    return ref, theta_ref, float(s[-1])

# =========================
# 几何/采样工具
# =========================
def resample_polyline_to_N(pts: np.ndarray, N: int):
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    fx = interp1d(s, pts[:, 0], kind='cubic')
    fy = interp1d(s, pts[:, 1], kind='cubic')
    s_new = np.linspace(0.0, s[-1], N)
    return np.vstack([fx(s_new), fy(s_new)]).T, float(s[-1])

def nearest_index(points: np.ndarray, xy: np.ndarray) -> int:
    return int(np.argmin(np.sum((points - xy[None, :])**2, axis=1)))

def sample_in_box(P: np.ndarray, Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = np.minimum(P, Q), np.maximum(P, Q)
    return rng.uniform(lo, hi)

# =========================
# SDF
# =========================
def build_sdf_from_grid(grid: np.ndarray):
    g = grid.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    inside = g < 0.5
    d_in = distance_transform_edt(inside)
    d_out = distance_transform_edt(~inside)
    sdf = d_in - d_out
    return sdf, inside

def sample_sdf_bilinear(sdf: np.ndarray, xy: np.ndarray) -> np.ndarray:
    coords = np.vstack([xy[:, 1], xy[:, 0]])
    return map_coordinates(sdf, coords, order=1, mode='nearest')

# =========================
# 车辆动力学与控制参数化
# =========================
def simulate_vehicle_bc(x0: np.ndarray, u_seq: np.ndarray, dt: float, state_steps: int,
                        L_pix: float, v_max: Optional[float] = None,
                        delta_rate_max: Optional[float] = None) -> np.ndarray:
    states = np.zeros((state_steps, 4), dtype=np.float64)
    states[0] = x0[:4]
    last_delta = 0.0
    for k in range(state_steps - 1):
        x, y, th, v = states[k]
        a, delta = u_seq[k]
        if delta_rate_max is not None:
            delta = np.clip(delta, last_delta - delta_rate_max * dt, last_delta + delta_rate_max * dt)
        last_delta = delta
        v_next = v + dt * a
        if v_max is not None:
            v_next = np.clip(v_next, 0.0, v_max)
        th_next = th + dt * (v_next / (L_pix + 1e-12)) * np.tan(delta)
        x_next = x + dt * v_next * np.cos(th)
        y_next = y + dt * v_next * np.sin(th)
        states[k+1] = [x_next, y_next, th_next, v_next]
    return states

def unpack_controls(z_flat: np.ndarray, control_steps: int, a_max: float, delta_max: float) -> np.ndarray:
    z = z_flat.reshape(control_steps, 2)
    a = a_max * np.tanh(z[:, 0])
    d = delta_max * np.tanh(z[:, 1])
    return np.column_stack([a, d])

# =========================
# OCP 代价
# =========================
def cost_function_reparam(z_flat, x0, raceline_ref, theta_ref, v_ref, dt,
                          state_steps, control_steps, a_max, delta_max, L_pix, v_max,
                          sdf_map=None, safety_margin_pix=8.0, w_safety=40.0,
                          delta_rate_max: Optional[float] = 2.5):
    u = unpack_controls(z_flat, control_steps, a_max, delta_max)
    states = simulate_vehicle_bc(x0, u, dt, state_steps, L_pix=L_pix, v_max=v_max, delta_rate_max=delta_rate_max)

    # Weights（与 notebook 保持）
    ##forward
    #w_path, w_u, w_smooth, w_v, w_terminal = 5.0, 1e-3, 0.05, 0.02, 30.0
    #inverse
    w_path, w_u, w_smooth, w_v, w_terminal = 18.0, 1e-3, 0.05, 0.01, 80.0
    pos_err = states[:, :2] - raceline_ref
    J_path = np.sum(np.einsum('ij,ij->i', pos_err, pos_err))
    J_u = np.sum(u**2)
    J_smooth = np.sum(np.diff(u, axis=0)**2)
    J_v = np.sum((states[:, 3] - v_ref)**2)
    term_pos = np.sum((states[-1, :2] - raceline_ref[-1])**2)
    dtheta = (states[-1, 2] - theta_ref[-1] + np.pi) % (2*np.pi) - np.pi
    J_term = term_pos + 0.1 * dtheta**2

    J_safe = 0.0
    if sdf_map is not None:
        sdf_vals = sample_sdf_bilinear(sdf_map, states[:, :2])
        clear = sdf_vals - safety_margin_pix
        phi = 1.0 / np.maximum(clear, 1.0)**2
        J_safe = np.sum(phi)

    J = (w_path*J_path + w_u*J_u + w_smooth*J_smooth + w_v*J_v + w_terminal*J_term + w_safety*J_safe)
    return float(J)

# =========================
# 连续化求解（续接法）
# =========================
def continuation_solve(z0: np.ndarray,
                       x0: np.ndarray,
                       raceline_ref: np.ndarray,
                       theta_ref: np.ndarray,
                       v_ref: np.ndarray,
                       cfg: OCPConfig,
                       v_max: float,
                       L_pix: float,
                       sdf_map: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    control_steps = cfg.state_steps - 1
    res_last = None
    for i, w_s in enumerate(cfg.w_safety_schedule, 1):
        print(f"\n--- Continuation {i}/{len(cfg.w_safety_schedule)}  w_safety={w_s:.1f}")
        res_last = minimize(
            cost_function_reparam, z0,
            args=(x0, raceline_ref, theta_ref, v_ref, cfg.dt,
                  cfg.state_steps, control_steps, cfg.a_max, cfg.delta_max, L_pix, v_max,
                  sdf_map, cfg.safety_margin_pix, w_s, cfg.delta_rate_max),
            method="L-BFGS-B",
            options={"maxiter": 400, "maxfun": 200000, "ftol": 1e-6, "gtol": 1e-5, "maxls": 50, "disp": True}
        )
        z0 = res_last.x
    u_opt = unpack_controls(res_last.x, control_steps, cfg.a_max, cfg.delta_max)
    states_opt = simulate_vehicle_bc(x0, u_opt, cfg.dt, cfg.state_steps, L_pix=L_pix, v_max=v_max,
                                     delta_rate_max=cfg.delta_rate_max)
    return u_opt, states_opt, {"result": res_last}

# =========================
# MPPI（本地抛光）
# =========================
def atanh_clip(x, eps=1e-6):
    x = np.clip(x, -1+eps, 1-eps)
    return np.arctanh(x)

def lowpass_controls(u, passes=2):
    if passes <= 0: return u
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    out = u.copy()
    for _ in range(passes):
        for j in range(2):
            out[:, j] = np.convolve(out[:, j], k, mode='same')
    return out

def rollout_cost_ocp_aligned(states, u_seq, raceline_ref, theta_ref, v_ref,
                             w_path=5.0, w_u=1e-3, w_smooth=0.05,
                             w_v=0.12, w_terminal=30.0,
                             sdf_map=None, safety_margin_pix=10.0, w_safety=3.5e4,
                             v_min_factor=0.35, w_vmin=0.08, w_prog=250.0,
                             avg_speed_pixels: Optional[float]=None):
    pos_err = states[:, :2] - raceline_ref
    J_path = np.sum(np.einsum('ij,ij->i', pos_err, pos_err))
    J_u = np.sum(u_seq**2)
    J_smooth = np.sum(np.diff(u_seq, axis=0)**2)
    J_v = np.sum((states[:, 3] - v_ref)**2)
    term_pos = np.sum((states[-1, :2] - raceline_ref[-1])**2)
    dtheta = (states[-1, 2] - theta_ref[-1] + np.pi) % (2*np.pi) - np.pi
    J_term = term_pos + 0.1 * dtheta**2
    J_safe = 0.0
    if sdf_map is not None:
        # print("saturation safety")
        # sdf_vals = sample_sdf_bilinear(sdf_map, states[:, :2])
        # clear = sdf_vals - safety_margin_pix
        # J_safe = np.sum(1.0 / np.maximum(clear, 1.0)**2)
        print("hinge2 safety")
        sdf_vals = sample_sdf_bilinear(sdf_map, states[:, :2])
        viol = np.maximum(0.0, safety_margin_pix - sdf_vals)  # <0安全, >0违规/不足余量
        J_safe = np.sum(viol * viol)

    v_min = (v_min_factor * (avg_speed_pixels if avg_speed_pixels is not None else 0.0))
    J_vmin = np.sum(np.maximum(0.0, v_min - states[:, 3])**2)
    idx = np.argmin(np.sum((raceline_ref - states[-1, :2])**2, axis=1))
    J_prog = (len(raceline_ref) - 1 - idx)**2
    return float(w_path*J_path + w_u*J_u + w_smooth*J_smooth + w_v*J_v + w_terminal*J_term +
                 w_safety*J_safe + w_vmin*J_vmin + w_prog*J_prog)

def mppi_local_refine(u_base, x0,
                      dt, L_pix, v_max, delta_rate_max,
                      raceline_ref, theta_ref, v_ref, sdf_map, safety_margin_pix,
                      a_max, delta_max,
                      iters=30, N=2048, sigma_z_a=0.18, sigma_z_delta=0.10, rho=0.85,
                      z_tr_clip=0.35, z_step_clip=0.20, lam=5000.0, step_size=0.9,
                      smooth_passes=2, avg_speed_pixels: Optional[float]=None):
    T = u_base.shape[0]
    z0 = np.zeros_like(u_base)
    z0[:, 0] = atanh_clip(u_base[:, 0] / a_max)
    z0[:, 1] = atanh_clip(u_base[:, 1] / delta_max)
    z_mean = z0.copy()

    states_best = simulate_vehicle_bc(x0, u_base, dt, T+1, L_pix=L_pix, v_max=v_max, delta_rate_max=delta_rate_max)
    J_best = rollout_cost_ocp_aligned(states_best, u_base, raceline_ref, theta_ref, v_ref,
                                      sdf_map=sdf_map, safety_margin_pix=safety_margin_pix,
                                      avg_speed_pixels=avg_speed_pixels)
    u_best = u_base.copy()

    for it in range(iters):
        eps = np.random.randn(N, T, 2)
        for k in range(1, T):
            eps[:, k, :] = rho * eps[:, k-1, :] + np.sqrt(1-rho**2) * eps[:, k, :]
        eps[:, :, 0] *= sigma_z_a
        eps[:, :, 1] *= sigma_z_delta

        z_samp = z_mean[None, :, :] + eps
        z_samp = z0[None, :, :] + np.clip(z_samp - z0[None, :, :], -z_tr_clip, z_tr_clip)

        u_samp = np.empty_like(z_samp)
        u_samp[:, :, 0] = a_max   * np.tanh(z_samp[:, :, 0])
        u_samp[:, :, 1] = delta_max * np.tanh(z_samp[:, :, 1])
        if smooth_passes > 0:
            for i in range(N):
                u_samp[i] = lowpass_controls(u_samp[i], passes=smooth_passes)
        u_samp[:, :, 0] = np.clip(u_samp[:, :, 0], -a_max, a_max)
        u_samp[:, :, 1] = np.clip(u_samp[:, :, 1], -delta_max, delta_max)

        costs = np.empty(N, dtype=np.float64)
        for i in range(N):
            st = simulate_vehicle_bc(x0, u_samp[i], dt, T+1, L_pix=L_pix, v_max=v_max, delta_rate_max=delta_rate_max)
            costs[i] = rollout_cost_ocp_aligned(st, u_samp[i], raceline_ref, theta_ref, v_ref,
                                                sdf_map=sdf_map, safety_margin_pix=safety_margin_pix,
                                                avg_speed_pixels=avg_speed_pixels)
        Jmin = np.min(costs)
        w = np.exp(-(costs - Jmin) / lam)
        S = np.sum(w) + 1e-12
        dz = (w[:, None, None] * (z_samp - z_mean[None, :, :])).sum(axis=0) / S
        dz = np.clip(dz, -z_step_clip, +z_step_clip)
        z_mean = z_mean + step_size * dz
        z_mean = z0 + np.clip(z_mean - z0, -z_tr_clip, z_tr_clip)

        i_best = np.argmin(costs)
        if costs[i_best] < J_best:
            J_best = float(costs[i_best])
            u_best = u_samp[i_best].copy()
            states_best = simulate_vehicle_bc(x0, u_best, dt, T+1, L_pix=L_pix, v_max=v_max, delta_rate_max=delta_rate_max)

        if (it+1) % 6 == 0:
            sigma_z_a *= 0.85
            sigma_z_delta *= 0.85
        if (it+1) % 5 == 0:
            print(f"[MPPI] iter {it+1:02d}/{iters}  J_best={J_best:.3e}  sig=({sigma_z_a:.3f},{sigma_z_delta:.3f})")

    u_best = lowpass_controls(u_best, passes=1)
    u_best[:, 0] = np.clip(u_best[:, 0], -a_max, a_max)
    u_best[:, 1] = np.clip(u_best[:, 1], -delta_max, delta_max)
    states_best = simulate_vehicle_bc(x0, u_best, dt, T+1, L_pix=L_pix, v_max=v_max, delta_rate_max=delta_rate_max)
    return u_best, states_best, J_best

# =========================
# 可视化
# =========================
def plot_results(states, u_seq, raceline_ref, v_ref, grid_map, dt, a_max, delta_max, v_max):
    t_s = np.arange(states.shape[0]) * dt
    t_u = np.arange(u_seq.shape[0]) * dt
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(grid_map, cmap='gray', origin='lower',
               extent=[0, grid_map.shape[1], 0, grid_map.shape[0]])
    ax1.plot(raceline_ref[:, 0], raceline_ref[:, 1], 'r--', lw=2, label='Reference')
    ax1.plot(states[:, 0], states[:, 1], 'b-', lw=2.5, label='Optimized')
    ax1.plot(states[0, 0], states[0, 1], 'g^', markersize=10, label='Start')
    ax1.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
    ax1.axis('equal'); ax1.grid(True); ax1.legend(); ax1.set_title('Trajectory')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_u, u_seq[:, 0], 'b-', label='a')
    ax2.axhline(a_max, color='b', ls='--'); ax2.axhline(-a_max, color='b', ls='--')
    ax2.plot(t_u, u_seq[:, 1], 'g-', label='δ')
    ax2.axhline(delta_max, color='g', ls='--'); ax2.axhline(-delta_max, color='g', ls='--')
    ax2.grid(True); ax2.legend(); ax2.set_title('Controls')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_s, v_ref, 'r--', label='v_ref')
    ax3.plot(t_s, states[:, 3], 'b-', label='v')
    ax3.axhline(v_max, color='b', ls=':', label='v_max')
    ax3.grid(True); ax3.legend(); ax3.set_title('Velocity')

    plt.tight_layout(); plt.show()

def draw_box(ax, P, Q, label, color='y'):
    lo = np.minimum(P, Q); hi = np.maximum(P, Q)
    xs = [lo[0], hi[0], hi[0], lo[0], lo[0]]
    ys = [lo[1], lo[1], hi[1], hi[1], lo[1]]
    ax.plot(xs, ys, color=color, lw=2)
    ax.text(lo[0], hi[1] + 6, label, color=color)

def plot_mppi_results(states_pre, u_pre, states_post, u_post,
                      raceline_ref, v_ref, grid_map, dt, a_max, delta_max, v_max):
    t_s = np.arange(states_post.shape[0]) * dt
    t_u = np.arange(u_post.shape[0]) * dt
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(grid_map, cmap='gray', origin='lower',
               extent=[0, grid_map.shape[1], 0, grid_map.shape[0]])
    ax1.plot(raceline_ref[:,0], raceline_ref[:,1], 'r--', lw=2, label='Reference')
    ax1.plot(states_pre[:,0],  states_pre[:,1],  'b--', lw=2.0, label='Before (OCP)')
    ax1.plot(states_post[:,0], states_post[:,1], 'c-',  lw=2.5, label='After (MPPI)')
    ax1.plot(states_pre[0, 0], states_pre[0, 1], '^', color='b', markersize=10)
    ax1.plot(states_pre[-1, 0], states_pre[-1, 1], 'o', color='b', markersize=10)
    ax1.plot(states_post[0, 0], states_post[0, 1], '^', color='c', markersize=10)
    ax1.plot(states_post[-1, 0], states_post[-1, 1], 'o', color='c', markersize=10)
    ax1.axis('equal'); ax1.grid(True); ax1.legend(); ax1.set_title('Trajectory')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t_u, u_pre[:,0],  'b--', alpha=0.6, label='a (before)')
    ax2.plot(t_u, u_post[:,0], 'b-',  label='a (after)')
    ax2.axhline(a_max, color='b', ls=':'); ax2.axhline(-a_max, color='b', ls=':')
    ax2.plot(t_u, u_pre[:,1],  'g--', alpha=0.6, label='δ (before)')
    ax2.plot(t_u, u_post[:,1], 'g-',  label='δ (after)')
    ax2.axhline(delta_max, color='g', ls=':'); ax2.axhline(-delta_max, color='g', ls=':')
    ax2.grid(True); ax2.legend(); ax2.set_title('Controls')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_s, v_ref, 'r--', label='v_ref')
    ax3.plot(t_s, states_pre[:,3],  'b--', alpha=0.6, label='v before')
    ax3.plot(t_s, states_post[:,3], 'b-',  label='v after')
    ax3.axhline(v_max, color='b', ls=':', label='v_max')
    ax3.grid(True); ax3.legend(); ax3.set_title('Velocity')

    plt.tight_layout(); plt.show()

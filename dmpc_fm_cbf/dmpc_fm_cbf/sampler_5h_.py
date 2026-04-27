# dmpc_fm_cbf/sampler_5h.py
# ===============================================================
# Piecewise FM sampler (5-channel) with XY-only SGFM-CBF projection
# - x_norm: (1,5,T)
# - v_pred: (1,5,T) from model(x,t,cond=None)
# - CBF: modifies ONLY v_pred[:,0:2,:] inside RHS
# - Piecewise integration: integrate [t0,t1] segment-by-segment
# - Output: x_fm_norm (5,T), states_norm (T,3), controls_norm (T,2)
# ===============================================================

from __future__ import annotations

import numpy as np
import torch
import torchdiffeq
import importlib

from dmpc_fm_cbf import cbf_project_velocity_sgfm


def _to_ellipse_list(ellipses):
    """
    Normalize ellipses/circles spec to:
      [{"center": np.array([cx,cy],float32), "radius": float}, ...]
    """
    if ellipses is None:
        return []
    out = []
    for e in ellipses:
        c = np.asarray(e["center"], dtype=np.float32).reshape(2)
        r = float(e.get("radius", e.get("r", 0.3)))
        out.append({"center": c, "radius": r})
    return out

# ★ Terminal Safety Filter
# 在 ODE 积分开始之前，把初始噪声 x_0 的 XY 通道里
# 落在障碍物椭圆内部的点沿径向推到椭圆边界外。
# 保证 h(x_0) >= 0，使 CBF 的初始可行性条件得到满足。
# 只修改 channel 0,1（XY），不动 theta/a/delta_rate 通道。
# 只在 use_cbf=True 且 ellipses 非空时调用。
# ===============================================================
def _terminal_safety_filter(
    x_seg: torch.Tensor,
    ellipses_list: list,
    dev: torch.device,
) -> torch.Tensor:
    """
    Args:
        x_seg:         (1, 5, T) 初始噪声 tensor
        ellipses_list: [{"center": np.array([cx,cy]), "radius": float}, ...]
        dev:           torch device
    Returns:
        x_safe: (1, 5, T) 投影后满足 h(x_0)>=0 的初始状态
    """
    x_safe = x_seg.clone()
    xy = x_safe[0, 0:2, :]   # (2, T)

    for ellipse in ellipses_list:
        center = torch.tensor(
            ellipse["center"], dtype=xy.dtype, device=dev
        )  # (2,)
        radius = float(ellipse["radius"])

        # 每个时间步到障碍物中心的向量和距离
        diff = xy - center.unsqueeze(1)        # (2, T)
        dist = torch.norm(diff, dim=0)          # (T,)

        # 找到落在椭圆内部的时间步（违反安全约束）
        unsafe_mask = dist < radius             # (T,) bool

        if unsafe_mask.any():
            # 处理点正好在中心的情况（避免除零，给默认方向）
            near_center = dist < 1e-6
            default_dir = torch.zeros_like(diff)
            default_dir[0, :] = 1.0             # 默认沿 x 轴正方向推出

            safe_diff = torch.where(
                near_center.unsqueeze(0).expand(2, -1),
                default_dir,
                diff
            )  # (2, T)

            # 单位径向方向
            direction = safe_diff / (dist.unsqueeze(0) + 1e-9)  # (2, T)

            # 把不安全的点推到椭圆边界上
            xy[:, unsafe_mask] = (
                center.unsqueeze(1) + direction[:, unsafe_mask] * radius
            )

    x_safe[0, 0:2, :] = xy
    return x_safe

def _resolve_cbf_projector():
    """
    Backward-compatible resolver expected by older notebook checks.
    Returns a callable SGFM-CBF projector even if imports yielded a module.
    """
    # common case: direct function export
    if callable(cbf_project_velocity_sgfm):
        return cbf_project_velocity_sgfm

    # sometimes package-level import may expose the module instead of function
    if hasattr(cbf_project_velocity_sgfm, "cbf_project_velocity_sgfm"):
        fn = getattr(cbf_project_velocity_sgfm, "cbf_project_velocity_sgfm")
        if callable(fn):
            return fn

    # final fallback: import concrete module and fetch function
    mod = importlib.import_module("dmpc_fm_cbf.cbf_project_velocity_sgfm")
    fn = getattr(mod, "cbf_project_velocity_sgfm", None)
    if callable(fn):
        return fn

    raise TypeError("Could not resolve callable cbf_project_velocity_sgfm projector.")


def _prepare_cond_tensor(cond, *, device: torch.device, dtype: torch.dtype):
    if cond is None:
        return None
    if torch.is_tensor(cond):
        c = cond.to(device=device, dtype=dtype)
    else:
        c = torch.as_tensor(np.asarray(cond, dtype=np.float32), device=device, dtype=dtype)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.ndim != 2 or c.shape[0] != 1:
        raise ValueError(f"cond must be shape (C,) or (1,C). Got {tuple(c.shape)}")
    return c


def _prepare_xy_norm(xy_norm, xy_world, scaler, *, device: torch.device, dtype: torch.dtype, name: str):
    if xy_norm is not None:
        arr = np.asarray(xy_norm, dtype=np.float32).reshape(2)
        return torch.as_tensor(arr, device=device, dtype=dtype)
    if xy_world is None:
        return None
    if scaler is None:
        raise ValueError(f"{name} provided in world coordinates but scaler is None.")
    arr = np.asarray(xy_world, dtype=np.float32).reshape(1, 2)
    arr_n = np.asarray(scaler.normalize(arr), dtype=np.float32).reshape(2)
    return torch.as_tensor(arr_n, device=device, dtype=dtype)


@torch.no_grad()
def piecewise_sample_fm_5ch_with_cbf(
    *,
    model_5ch,                 # trained FM model: v = model(x, t, cond=None)
    T_steps: int,              # length T
    device: str = "cuda",
    # ---- ODE settings ----
    total_time: float = 1.0,
    num_segments: int = 20,
    ode_method: str = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    # ---- init ----
    noise_scale: float = 1.0,
    x0_noise: torch.Tensor | None = None,   # (1,5,T)
    # ---- conditioning / guidance ----
    cond=None,                 # optional conditioning vector (C,) or (1,C)
    start_xy=None,             # world start XY (2,)
    start_xy_norm=None,        # normalized start XY (2,)
    start_guidance_weight: float = 0.0,
    lock_start: bool = False,
    goal_xy=None,              # world goal XY (2,)
    goal_xy_norm=None,         # normalized goal XY (2,)
    goal_guidance_weight: float = 0.0,
    goal_guidance_mode: str = "endpoint",  # endpoint | tail | all
    # ---- CBF ----
    use_cbf: bool = True,
    scaler=None,               # must have denormalize() used inside cbf_project_velocity_sgfm
    ellipses=None,             # list of {"center":(2,), "radius":float}
    # ---- SGFM schedule knobs ----
    tau0_hf: float = 0.7,      # HF starts after tau>=tau0_hf
    p_hf: float = 2.0,         # HF ramp power
    lf_on: float = 1.0,        # LF gate multiplier (constant)
    hf_max: float = 1.0,       # HF gate max
    # ---- extra cbf hyperparams ----
    cbf_kwargs: dict | None = None,
):
    """
    Returns:
      x_fm_norm: (5,T) numpy float32
      states_norm:  (T,3) numpy float32   [x,y,theta] in normalized space
      controls_norm:(T,2) numpy float32   [a,delta_rate] in normalized space
    """
    dev = torch.device(device)
    model_5ch = model_5ch.to(dev).eval()
    cbf_projector = _resolve_cbf_projector()

    ellipses_list = _to_ellipse_list(ellipses)
    cbf_kwargs = {} if cbf_kwargs is None else dict(cbf_kwargs)
    cond_t = _prepare_cond_tensor(cond, device=dev, dtype=torch.float32)
    start_xy_norm_t = _prepare_xy_norm(
        start_xy_norm, start_xy, scaler, device=dev, dtype=torch.float32, name="start_xy"
    )
    goal_xy_norm_t = _prepare_xy_norm(
        goal_xy_norm, goal_xy, scaler, device=dev, dtype=torch.float32, name="goal_xy"
    )

    # init x0
    if x0_noise is None:
        x_seg = torch.randn((1, 5, T_steps), device=dev) * float(noise_scale)
    else:
        x_seg = x0_noise.to(dev)
        if tuple(x_seg.shape) != (1, 5, T_steps):
            raise ValueError(f"x0_noise must be (1,5,T). Got {tuple(x_seg.shape)}")
    if lock_start and start_xy_norm_t is not None:
        x_seg[:, 0:2, 0] = start_xy_norm_t.to(dtype=x_seg.dtype).view(1, 2)

    # ★ Terminal Safety Filter：保证 h(x_0) >= 0
    # 在 ODE 积分之前执行，把初始噪声里落在障碍物椭圆内的 XY 点推到边界外。
    # 触发条件：use_cbf=True 且有障碍物（远距离时 use_cbf=False，不触发）。
    if use_cbf and len(ellipses_list) > 0:
        x_seg = _terminal_safety_filter(x_seg, ellipses_list, dev)


    # piecewise times
    times = torch.linspace(0.0, float(total_time), steps=int(num_segments) + 1, device=dev)

    def hf_gate(t_scalar: float) -> float:
        tau = float(t_scalar) / float(total_time)
        if tau < tau0_hf:
            return 0.0
        s = (tau - tau0_hf) / max(1e-9, (1.0 - tau0_hf))
        return float(hf_max) * float(s ** p_hf)

    def lf_gate(_t_scalar: float) -> float:
        return float(lf_on)

    # RHS
    def fm_rhs(t, x_flat):
        x_norm = x_flat.view(1, 5, T_steps)  # (1,5,T)
        t_t = torch.tensor([t], device=dev, dtype=x_norm.dtype)

        # model predicts full 5ch velocity field (conditional if cond provided)
        cond_now = None if cond_t is None else cond_t.to(dtype=x_norm.dtype)
        v_pred = model_5ch(x_norm, t_t.view(-1), cond=cond_now)  # (1,5,T)

        # optional start/goal guidance in normalized XY
        tau = float(t) / max(1e-12, float(total_time))
        w_t = 0.25 + 0.75 * tau
        if start_xy_norm_t is not None and float(start_guidance_weight) > 0.0:
            start_now = start_xy_norm_t.to(dtype=x_norm.dtype).view(1, 2, 1)
            w_start = float(start_guidance_weight) * w_t
            v_pred[:, 0:2, 0:1] = v_pred[:, 0:2, 0:1] + w_start * (start_now - x_norm[:, 0:2, 0:1])

        if goal_xy_norm_t is not None and float(goal_guidance_weight) > 0.0:
            goal_now = goal_xy_norm_t.to(dtype=x_norm.dtype).view(1, 2, 1)
            w_goal = float(goal_guidance_weight) * w_t
            mode = str(goal_guidance_mode).lower()
            if mode == "endpoint":
                v_pred[:, 0:2, -1:] = v_pred[:, 0:2, -1:] + w_goal * (goal_now - x_norm[:, 0:2, -1:])
            elif mode == "tail":
                idx = torch.linspace(0.0, 1.0, T_steps, device=dev, dtype=x_norm.dtype).view(1, 1, T_steps)
                ramp = torch.clamp((idx - 0.5) / 0.5, min=0.0, max=1.0)
                v_pred[:, 0:2, :] = v_pred[:, 0:2, :] + (w_goal * ramp) * (goal_now - x_norm[:, 0:2, :])
            else:  # all
                v_pred[:, 0:2, :] = v_pred[:, 0:2, :] + w_goal * (goal_now - x_norm[:, 0:2, :])

        if (not use_cbf) or (len(ellipses_list) == 0):
            return v_pred.reshape(-1)

        # CBF only on XY
        v_xy_nom = v_pred[:, 0:2, :]  # (1,2,T)
        g_lf = lf_gate(float(t))
        g_hf = hf_gate(float(t))

        v_xy_safe = cbf_projector(
            x_norm_1x2T=x_norm[:, 0:2, :],
            v_nom_1x2T=v_xy_nom,
            t_scalar=float(t),
            T_scalar=float(total_time),
            scaler=scaler,
            ellipses=ellipses_list,
            gate_lf=float(g_lf),
            gate_hf=float(g_hf),
            **cbf_kwargs,
        )  # (1,2,T)

        v_out = v_pred.clone()
        v_out[:, 0:2, :] = v_xy_safe
        return v_out.reshape(-1)

    # piecewise integrate
    for i in range(int(num_segments)):
        t0 = times[i]
        t1 = times[i + 1]
        t_span = torch.stack([t0, t1])
        sol = torchdiffeq.odeint(
            fm_rhs,
            x_seg.reshape(-1),
            t_span,
            method=ode_method,
            atol=atol,
            rtol=rtol,
        )
        x_seg = sol[-1].view(1, 5, T_steps)
        if lock_start and start_xy_norm_t is not None:
            x_seg[:, 0:2, 0] = start_xy_norm_t.to(dtype=x_seg.dtype).view(1, 2)

    x_fm_norm = x_seg[0].detach().cpu().numpy().astype(np.float32)  # (5,T)
    states_norm = x_fm_norm[:3, :].T.astype(np.float32)             # (T,3)
    controls_norm = x_fm_norm[3:5, :].T.astype(np.float32)          # (T,2)

    return x_fm_norm, states_norm, controls_norm


@torch.no_grad()
def split_and_denormalize_5ch(
    x_fm_norm_5T,
    *,
    state_scaler=None,
    control_scaler=None,
):
    """
    x_fm_norm_5T: (5,T) numpy
    Returns:
      states (T,3), controls (T,2) in physical space if scalers provided;
      else normalized arrays.
    """
    x_fm_norm_5T = np.asarray(x_fm_norm_5T, dtype=np.float32)
    states_norm = x_fm_norm_5T[:3, :].T
    controls_norm = x_fm_norm_5T[3:5, :].T

    if (state_scaler is None) or (control_scaler is None):
        return states_norm, controls_norm

    states = state_scaler.denormalize(states_norm)
    controls = control_scaler.denormalize(controls_norm)
    return states, controls

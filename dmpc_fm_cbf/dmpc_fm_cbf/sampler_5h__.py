# dmpc_fm_cbf/sampler_5h.py
# ===============================================================
# Piecewise FM sampler (5-channel) with XY-only CBF projection
#
# 简化版（导师建议）：
#   - 去掉 Terminal Safety Filter
#   - 去掉 LF/HF 双门控
#   - CBF gate = time_gate(tau) × proximity_gate(x, ellipses)
#       time_gate:      前期为 0，tau >= tau0 后平滑增大（sigmoid-like）
#       proximity_gate: 离障碍物越近越大（势能场思路）
#   两者相乘，自然实现"前期不开，靠近障碍物时才逐渐开强"
# ===============================================================

from __future__ import annotations

import numpy as np
import torch
import torchdiffeq
import importlib

from dmpc_fm_cbf import cbf_project_velocity_sgfm


# ================================================================
# 工具函数
# ================================================================

def _to_ellipse_list(ellipses):
    if ellipses is None:
        return []
    out = []
    for e in ellipses:
        c = np.asarray(e["center"], dtype=np.float32).reshape(2)
        r = float(e.get("radius", e.get("r", 0.3)))
        out.append({"center": c, "radius": r})
    return out


def _resolve_cbf_projector():
    if callable(cbf_project_velocity_sgfm):
        return cbf_project_velocity_sgfm
    if hasattr(cbf_project_velocity_sgfm, "cbf_project_velocity_sgfm"):
        fn = getattr(cbf_project_velocity_sgfm, "cbf_project_velocity_sgfm")
        if callable(fn):
            return fn
    mod = importlib.import_module("dmpc_fm_cbf.cbf_project_velocity_sgfm")
    fn  = getattr(mod, "cbf_project_velocity_sgfm", None)
    if callable(fn):
        return fn
    raise TypeError("Could not resolve callable cbf_project_velocity_sgfm projector.")


def _prepare_cond_tensor(cond, *, device, dtype):
    if cond is None:
        return None
    if torch.is_tensor(cond):
        c = cond.to(device=device, dtype=dtype)
    else:
        c = torch.as_tensor(np.asarray(cond, dtype=np.float32),
                             device=device, dtype=dtype)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.ndim != 2 or c.shape[0] != 1:
        raise ValueError(f"cond must be (C,) or (1,C). Got {tuple(c.shape)}")
    return c


def _prepare_xy_norm(xy_norm, xy_world, scaler, *, device, dtype, name):
    if xy_norm is not None:
        return torch.as_tensor(
            np.asarray(xy_norm, dtype=np.float32).reshape(2),
            device=device, dtype=dtype
        )
    if xy_world is None:
        return None
    if scaler is None:
        raise ValueError(f"{name} given in world coords but scaler is None.")
    arr_n = np.asarray(
        scaler.normalize(np.asarray(xy_world, np.float32).reshape(1, 2)),
        dtype=np.float32
    ).reshape(2)
    return torch.as_tensor(arr_n, device=device, dtype=dtype)


# ================================================================
# ★ CBF gate 函数
#
# gate(tau, x_norm, ellipses) = time_gate(tau) × proximity_gate(x, ellipses)
#
# time_gate(tau):
#   tau < tau0          → 0               （前期完全不开）
#   tau0 <= tau <= tau1 → smooth ramp      （平滑增大）
#   tau > tau1          → 1               （后期全开）
#   使用 smoothstep：3t²-2t³，比线性更平滑，无突变
#
# proximity_gate(x, ellipses):
#   对每个椭圆计算势能 P = exp(-k * max(h, 0))
#   h = dist - radius（CBF 函数值，h>0 安全，h<0 不安全）
#   离障碍物越近 h 越小，势能越大，gate 越大
#   取所有椭圆的最大势能（最危险的那个障碍物决定强度）
#   范围 (0, 1]，远离障碍物时趋近于 0，进入危险区域时趋近于 1
# ================================================================

def _time_gate(tau: float, tau0: float, tau1: float) -> float:
    """
    smoothstep ramp：
      tau < tau0  → 0
      tau > tau1  → 1
      中间        → 3t²-2t³  (t = (tau-tau0)/(tau1-tau0))
    """
    if tau <= tau0:
        return 0.0
    if tau >= tau1:
        return 1.0
    t = (tau - tau0) / (tau1 - tau0)
    return float(3 * t**2 - 2 * t**3)


def _proximity_gate(
    x_norm_1x2T: torch.Tensor,   # (1, 2, T)
    ellipses_list: list,
    k: float = 3.0,               # 势能衰减系数，越大越陡
) -> float:
    """
    势能场门控：
      对每个障碍物椭圆计算 h = dist - radius
      P = 1 - exp(-k * max(-h, 0))  即 h < 0（在障碍物内）时 P→1
      等价写法：P = 1 - exp(-k * relu(-h))
      取所有椭圆的最大 P

    h > 0（安全区）：P → 0，CBF gate 贡献小
    h < 0（危险区）：P → 1，CBF gate 贡献大
    """
    if len(ellipses_list) == 0:
        return 0.0

    xy = x_norm_1x2T[0]  # (2, T)
    max_p = 0.0

    for e in ellipses_list:
        center = torch.tensor(e["center"], dtype=xy.dtype, device=xy.device)
        radius = float(e["radius"])

        dist = torch.norm(xy - center.unsqueeze(1), dim=0)  # (T,)
        h    = dist - radius                                  # (T,)

        # relu(-h)：只有在障碍物内部或边界处才有值
        penalty = torch.relu(-h)                              # (T,)
        P       = float(1.0 - torch.exp(-k * penalty).min()) # 最危险的时间步
        max_p   = max(max_p, P)

    return float(np.clip(max_p, 0.0, 1.0))


# ================================================================
# 主采样函数
# ================================================================

@torch.no_grad()
def piecewise_sample_fm_5ch_with_cbf(
    *,
    model_5ch,
    T_steps: int,
    device: str = "cuda",
    # ---- ODE settings ----
    total_time: float = 1.0,
    num_segments: int = 20,
    ode_method: str = "dopri5",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    # ---- init ----
    noise_scale: float = 1.0,
    x0_noise: torch.Tensor | None = None,
    # ---- conditioning / guidance ----
    cond=None,
    start_xy=None,
    start_xy_norm=None,
    start_guidance_weight: float = 0.0,
    lock_start: bool = False,
    goal_xy=None,
    goal_xy_norm=None,
    goal_guidance_weight: float = 0.0,
    goal_guidance_mode: str = "endpoint",
    # ---- CBF ----
    use_cbf: bool = True,
    scaler=None,
    ellipses=None,
    tau0: float = 0.5,    # flow time 低于此值 CBF 完全关闭
    tau1: float = 0.9,    # flow time 高于此值 CBF 全开
    prox_k: float = 3.0,  # 势能场衰减系数
    # ---- extra cbf hyperparams ----
    cbf_kwargs: dict | None = None,
):
    """
    Returns:
      x_fm_norm:     (5, T) numpy float32
      states_norm:   (T, 3) numpy float32   [x, y, theta]
      controls_norm: (T, 2) numpy float32   [a, delta_rate]
    """
    dev           = torch.device(device)
    model_5ch     = model_5ch.to(dev).eval()
    cbf_projector = _resolve_cbf_projector()

    ellipses_list = _to_ellipse_list(ellipses)
    cbf_kwargs    = {} if cbf_kwargs is None else dict(cbf_kwargs)
    cond_t        = _prepare_cond_tensor(cond, device=dev, dtype=torch.float32)
    start_xy_norm_t = _prepare_xy_norm(
        start_xy_norm, start_xy, scaler,
        device=dev, dtype=torch.float32, name="start_xy"
    )
    goal_xy_norm_t = _prepare_xy_norm(
        goal_xy_norm, goal_xy, scaler,
        device=dev, dtype=torch.float32, name="goal_xy"
    )

    # ── init x0 ──────────────────────────────────────────────
    if x0_noise is None:
        x_seg = torch.randn((1, 5, T_steps), device=dev) * float(noise_scale)
    else:
        x_seg = x0_noise.to(dev)
        if tuple(x_seg.shape) != (1, 5, T_steps):
            raise ValueError(f"x0_noise must be (1,5,T). Got {tuple(x_seg.shape)}")

    if lock_start and start_xy_norm_t is not None:
        x_seg[:, 0:2, 0] = start_xy_norm_t.to(dtype=x_seg.dtype).view(1, 2)

    # ── piecewise times ───────────────────────────────────────
    times = torch.linspace(
        0.0, float(total_time),
        steps=int(num_segments) + 1, device=dev
    )

    # ── ODE RHS ──────────────────────────────────────────────
    def fm_ode(t, x_flat):
        x_norm = x_flat.view(1, 5, T_steps)                        # (1, 5, T)
        t_t    = torch.tensor([t], device=dev, dtype=x_norm.dtype)

        # model forward
        cond_now = None if cond_t is None else cond_t.to(dtype=x_norm.dtype)
        v_pred   = model_5ch(x_norm, t_t.view(-1), cond=cond_now)  # (1, 5, T)

        # ── guidance ─────────────────────────────────────────
        tau = float(t) / max(1e-12, float(total_time))
        w_t = 0.25 + 0.75 * tau                                     # ramp weight

        if start_xy_norm_t is not None and float(start_guidance_weight) > 0.0:
            start_now = start_xy_norm_t.to(dtype=x_norm.dtype).view(1, 2, 1)
            w_s = float(start_guidance_weight) * w_t
            v_pred[:, 0:2, 0:1] += w_s * (start_now - x_norm[:, 0:2, 0:1])

        if goal_xy_norm_t is not None and float(goal_guidance_weight) > 0.0:
            goal_now = goal_xy_norm_t.to(dtype=x_norm.dtype).view(1, 2, 1)
            w_g  = float(goal_guidance_weight) * w_t
            mode = str(goal_guidance_mode).lower()
            if mode == "endpoint":
                v_pred[:, 0:2, -1:] += w_g * (goal_now - x_norm[:, 0:2, -1:])
            elif mode == "tail":
                idx  = torch.linspace(0.0, 1.0, T_steps,
                                      device=dev, dtype=x_norm.dtype).view(1, 1, T_steps)
                ramp = torch.clamp((idx - 0.5) / 0.5, min=0.0, max=1.0)
                v_pred[:, 0:2, :] += (w_g * ramp) * (goal_now - x_norm[:, 0:2, :])
            else:  # "all"
                v_pred[:, 0:2, :] += w_g * (goal_now - x_norm[:, 0:2, :])

        # ── CBF correction ────────────────────────────────────
        # 只修正 XY 两个通道，theta/a/delta_rate 通道不变
        # gate = time_gate(tau) × proximity_gate(x, ellipses)
        # projector 负责: v_safe = v_nom + gate * (v_projected - v_nom)
        if use_cbf and len(ellipses_list) > 0:
            g_time = _time_gate(tau, tau0, tau1)
            g_prox = _proximity_gate(x_norm[:, 0:2, :], ellipses_list, k=prox_k)
            gate   = float(g_time * g_prox)

            if gate >= 1e-6:                          # gate 为 0 时跳过，省计算
                v_xy_safe = cbf_projector(
                    x_norm_1x2T=x_norm[:, 0:2, :],   # (1, 2, T)
                    v_nom_1x2T=v_pred[:, 0:2, :],     # (1, 2, T)
                    ellipses=ellipses_list,
                    gate_scalar=gate,                  # 唯一 gate
                    **cbf_kwargs,
                )                                      # (1, 2, T)

                v_pred = v_pred.clone()
                v_pred[:, 0:2, :] = v_xy_safe         # 写回 XY，其余通道不变

        return v_pred.reshape(-1)

    # ── piecewise integrate ───────────────────────────────────
    for i in range(int(num_segments)):
        t_span = torch.stack([times[i], times[i + 1]])
        sol    = torchdiffeq.odeint(
            fm_ode,
            x_seg.reshape(-1),
            t_span,
            method=ode_method,
            atol=atol,
            rtol=rtol,
        )
        x_seg = sol[-1].view(1, 5, T_steps)

        if lock_start and start_xy_norm_t is not None:
            x_seg[:, 0:2, 0] = start_xy_norm_t.to(dtype=x_seg.dtype).view(1, 2)

    # ── unpack outputs ────────────────────────────────────────
    x_fm_norm     = x_seg[0].detach().cpu().numpy().astype(np.float32)  # (5, T)
    states_norm   = x_fm_norm[:3, :].T.astype(np.float32)               # (T, 3)
    controls_norm = x_fm_norm[3:5, :].T.astype(np.float32)              # (T, 2)

    return x_fm_norm, states_norm, controls_norm


@torch.no_grad()
def split_and_denormalize_5ch(
    x_fm_norm_5T,
    *,
    state_scaler=None,
    control_scaler=None,
):
    x_fm_norm_5T  = np.asarray(x_fm_norm_5T, dtype=np.float32)
    states_norm   = x_fm_norm_5T[:3, :].T
    controls_norm = x_fm_norm_5T[3:5, :].T

    if (state_scaler is None) or (control_scaler is None):
        return states_norm, controls_norm

    states   = state_scaler.denormalize(states_norm)
    controls = control_scaler.denormalize(controls_norm)
    return states, controls
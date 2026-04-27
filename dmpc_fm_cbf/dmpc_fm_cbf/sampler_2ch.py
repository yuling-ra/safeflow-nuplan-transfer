# dmpc_fm_cbf/sampler_2ch.py

from __future__ import annotations
import numpy as np
import torch
import torchdiffeq
import importlib
from dmpc_fm_cbf import cbf_project_velocity_sgfm


class BicycleDynamics:
    def __init__(self, L=0.8, dt=0.1,
                 delta_max=0.524, a_max=0.8,
                 delta_rate_max=2.09,
                 v_min=0.1, v_max=2.0):
        self.L              = L
        self.dt             = dt
        self.delta_max      = delta_max
        self.a_max          = a_max
        self.delta_rate_max = delta_rate_max
        self.v_min          = v_min
        self.v_max          = v_max

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x, y, theta, v, delta = state
        a, delta_rate = control
        a          = np.clip(a,          -self.a_max,          self.a_max)
        delta_rate = np.clip(delta_rate, -self.delta_rate_max, self.delta_rate_max)
        delta_new  = np.clip(delta + delta_rate * self.dt, -self.delta_max, self.delta_max)
        v_new      = np.clip(v + a * self.dt, self.v_min, self.v_max)
        x_new      = x + v * np.cos(theta) * self.dt
        y_new      = y + v * np.sin(theta) * self.dt
        theta_new  = theta + (v / self.L) * np.tan(delta) * self.dt
        return np.array([x_new, y_new, theta_new, v_new, delta_new], dtype=np.float32)

    def rollout(self, state0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = [state0.copy()]
        state  = state0.copy()
        for u in controls:
            state = self.step(state, u)
            states.append(state)
        return np.array(states, dtype=np.float32)


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
    raise TypeError("Could not resolve callable cbf_project_velocity_sgfm.")


def _prepare_cond_tensor(cond, *, device, dtype):
    if cond is None:
        return None
    if torch.is_tensor(cond):
        c = cond.to(device=device, dtype=dtype)
    else:
        c = torch.as_tensor(np.asarray(cond, dtype=np.float32), device=device, dtype=dtype)
    if c.ndim == 1:
        c = c.unsqueeze(0)
    if c.ndim != 2 or c.shape[0] != 1:
        raise ValueError(f"cond must be (C,) or (1,C). Got {tuple(c.shape)}")
    return c


def _normalize_cond(cond_raw: np.ndarray, norm_params: dict) -> np.ndarray:
    c          = cond_raw.copy().astype(np.float32)
    cond_mean  = norm_params["cond_mean"]
    cond_scale = norm_params["cond_scale"]
    c[0] = (c[0] - cond_mean[0]) / cond_scale[0]
    c[1] = (c[1] - cond_mean[1]) / cond_scale[1]
    c[4] = (c[4] - cond_mean[4]) / cond_scale[4]
    return c


def _denormalize_controls(ctrl_norm: np.ndarray, norm_params: dict) -> np.ndarray:
    ctrl_mean = norm_params["ctrl_mean"]  # (1, 2, 1)
    ctrl_std  = norm_params["ctrl_std"]   # (1, 2, 1)
    return ctrl_norm * ctrl_std[0] + ctrl_mean[0]  # (2, T)


@torch.no_grad()
def piecewise_sample_safefm_2ch(
    *,
    model_safefm,
    T_steps: int,
    state0: np.ndarray,           # (5,) [x, y, theta, v, delta]
    norm_params: dict,
    device: str        = "cuda",
    total_time: float  = 1.0,
    num_segments: int  = 20,
    ode_method: str    = "dopri5",
    atol: float        = 1e-5,
    rtol: float        = 1e-5,
    noise_scale: float = 1.0,
    x0_noise: torch.Tensor | None = None,
    cond               = None,
    use_cbf: bool      = True,
    scaler             = None,
    ellipses           = None,
    tau0_hf: float     = 0.7,
    p_hf: float        = 2.0,
    lf_on: float       = 1.0,
    hf_max: float      = 1.0,
    cbf_kwargs: dict | None = None,
    bicycle_params: dict | None = None,
):
    """
    SafeFM 2ch sampler with CBF inside ODE:

    ODE state = [u_flat (2*T), xy_current (2)]
    在每个 ODE step 里：
      1. model 预测控制速度场 v_u (2, T)
      2. 反归一化当前控制量 → 得到当前时刻的 xy 速度
      3. 对 xy 速度做 CBF 投影
      4. 把 CBF 修正映射回控制速度场

    Returns:
        controls:  (T, 2)   物理控制量
        states:    (T+1, 5) 物理状态
        ctrl_norm: (2, T)   归一化控制量（调试用）
    """
    dev           = torch.device(device)
    model_safefm  = model_safefm.to(dev).eval()
    ellipses_list = _to_ellipse_list(ellipses)
    cbf_kwargs    = {} if cbf_kwargs is None else dict(cbf_kwargs)
    bp            = bicycle_params or {}
    dynamics      = BicycleDynamics(**bp)

    # 归一化 cond
    cond_norm = _normalize_cond(np.asarray(cond, dtype=np.float32), norm_params)
    cond_t    = _prepare_cond_tensor(cond_norm, device=dev, dtype=torch.float32)

    # 初始噪声 (1, 2, T)
    if x0_noise is None:
        u_seg = torch.randn((1, 2, T_steps), device=dev) * float(noise_scale)
    else:
        u_seg = x0_noise.to(dev)
        if tuple(u_seg.shape) != (1, 2, T_steps):
            raise ValueError(f"x0_noise must be (1,2,T). Got {tuple(u_seg.shape)}")

    # 初始物理状态
    s0      = np.asarray(state0, dtype=np.float32).reshape(5)
    xy_curr = torch.tensor(s0[0:2], dtype=torch.float32, device=dev)  # (2,)

    times = torch.linspace(0.0, float(total_time),
                           steps=int(num_segments) + 1, device=dev)

    def hf_gate(tau: float) -> float:
        if tau < tau0_hf:
            return 0.0
        s = (tau - tau0_hf) / max(1e-9, 1.0 - tau0_hf)
        return float(hf_max) * float(s ** p_hf)

    def lf_gate(_tau: float) -> float:
        return float(lf_on)

    cbf_projector = _resolve_cbf_projector() if (use_cbf and len(ellipses_list) > 0) else None

    # ---------------------------------------------------------------
    # ODE RHS：控制空间 + CBF
    # ---------------------------------------------------------------
    def fm_rhs(t, u_flat):
        tau    = float(t) / max(1e-12, float(total_time))
        u_norm = u_flat.view(1, 2, T_steps)
        t_t    = torch.tensor([float(t)], device=dev, dtype=u_norm.dtype)
        cond_now = None if cond_t is None else cond_t.to(dtype=u_norm.dtype)

        # FM 预测控制速度场
        v_u = model_safefm(u_norm, t_t.view(-1), cond=cond_now)  # (1, 2, T)

        if cbf_projector is None:
            return v_u.reshape(-1)

        # ---- CBF 修正 ----
        # 用当前归一化控制量估计每个时间步的 xy 位置
        # 策略：用 xy_curr（当前已知位置）+ 控制量积分偏移
        ctrl_now = _denormalize_controls(
            u_norm[0].detach().cpu().numpy(), norm_params
        )  # (2, T)

        # 简单估计：从 xy_curr 出发，用当前控制量滚动得到各时间步 xy
        state_tmp = np.array([
            xy_curr[0].item(), xy_curr[1].item(),
            s0[2], s0[3], s0[4]
        ], dtype=np.float32)
        xy_traj = np.zeros((2, T_steps), dtype=np.float32)
        for k in range(T_steps):
            state_tmp    = dynamics.step(state_tmp, ctrl_now[:, k])
            xy_traj[:, k] = state_tmp[0:2]

        # xy 位置 tensor (1, 2, T)
        xy_t = torch.tensor(xy_traj[None], dtype=torch.float32, device=dev)

        # 对应 xy 速度：从控制量反推（v_x = v*cos(θ), v_y = v*sin(θ)）
        # 用相邻位置差近似
        xy_prev      = torch.tensor(
            np.concatenate([
                xy_curr.cpu().numpy().reshape(2, 1),
                xy_traj[:, :-1]
            ], axis=1)[None], dtype=torch.float32, device=dev
        )  # (1, 2, T)
        v_xy_nom = xy_t - xy_prev   # (1, 2, T) 位置增量 ≈ xy 速度

        g_lf = lf_gate(tau)
        g_hf = hf_gate(tau)

        v_xy_safe = cbf_projector(
            x_norm_1x2T = xy_t,
            v_nom_1x2T  = v_xy_nom,
            t_scalar    = float(t),
            T_scalar    = float(total_time),
            scaler      = scaler,
            ellipses    = ellipses_list,
            gate_lf     = float(g_lf),
            gate_hf     = float(g_hf),
            **cbf_kwargs,
        )  # (1, 2, T)

        # CBF 修正量映射回控制速度场
        # delta_xy = v_xy_safe - v_xy_nom → 对应的控制修正
        # 简单策略：把 xy 修正比例映射到 delta_rate 通道
        # delta_v_xy: (1, 2, T)
        delta_v_xy = v_xy_safe - v_xy_nom  # (1, 2, T)

        # 修正幅度（标量）per time step
        corr_scale = torch.norm(delta_v_xy, dim=1, keepdim=True)  # (1, 1, T)

        # 把修正加到 v_u 的 delta_rate 通道（channel 1）
        # 因为 delta_rate 控制转向，和 xy 偏移最相关
        v_u_safe      = v_u.clone()
        v_u_safe[:, 1:2, :] = v_u[:, 1:2, :] + corr_scale * torch.sign(
            delta_v_xy[:, 1:2, :]
        )

        return v_u_safe.reshape(-1)

    # ---------------------------------------------------------------
    # Piecewise integrate
    # ---------------------------------------------------------------
    for i in range(int(num_segments)):
        t0     = times[i]
        t1     = times[i + 1]
        t_span = torch.stack([t0, t1])
        sol    = torchdiffeq.odeint(
            fm_rhs,
            u_seg.reshape(-1),
            t_span,
            method=ode_method,
            atol=atol,
            rtol=rtol,
        )
        u_seg = sol[-1].view(1, 2, T_steps)

        # 更新 xy_curr：用当前段结束时的控制量做一步积分
        ctrl_seg = _denormalize_controls(
            u_seg[0].cpu().numpy(), norm_params
        )  # (2, T)
        # 取第一个时间步的控制量更新当前位置
        state_upd = np.array([
            xy_curr[0].item(), xy_curr[1].item(),
            s0[2], s0[3], s0[4]
        ], dtype=np.float32)
        state_upd = dynamics.step(state_upd, ctrl_seg[:, 0])
        xy_curr   = torch.tensor(state_upd[0:2], dtype=torch.float32, device=dev)

    # ---------------------------------------------------------------
    # 最终反归一化 + 动力学 rollout
    # ---------------------------------------------------------------
    ctrl_norm_np = u_seg[0].cpu().numpy().astype(np.float32)  # (2, T)
    ctrl_phys_np = _denormalize_controls(ctrl_norm_np, norm_params)
    controls     = ctrl_phys_np.T   # (T, 2)
    states       = dynamics.rollout(s0, controls)  # (T+1, 5)

    return controls, states, ctrl_norm_np


def load_norm_params(ckpt_path: str, device: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "norm_params" not in ckpt:
        raise KeyError(f"norm_params not found in {ckpt_path}")
    return ckpt["norm_params"]


#**CBF 在 ODE 里的逻辑：**

#每个 ODE step:
#   1. model → v_u (控制速度场)
#   2. 用当前控制量估计各时间步 xy 轨迹
#   3. CBF 投影 xy 速度 → v_xy_safe
#   4. 计算修正量 delta_v_xy
#   5. 把修正映射回 v_u 的 delta_rate 通道
#   6. 返回修正后的 v_u_safe
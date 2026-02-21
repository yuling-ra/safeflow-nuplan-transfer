# ===============================================================
# NEW CELL (NO-VIRTUAL-OBS): Rollout with model_vel (CBF inside ODE)
#   + EMA smoothing + Goal Pull
#   + AUTO-ADAPT cond_dim (expects 5 by default) to match your trained model
#
# Key change:
#   - conditioning is GOAL-only (no virtual obstacle, no obs_x/obs_y in cond)
#   - REAL obstacles are still used by CBF INSIDE the ODE
# ===============================================================

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchdiffeq

# -------------------------
# Utils
# -------------------------
def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)

def _rot2(a: float) -> np.ndarray:
    ca, sa = float(np.cos(a)), float(np.sin(a))
    return np.array([[ca, -sa], [sa, ca]], dtype=np.float64)

def _circle_list_to_ellipses(circles):
    ell = []
    for c in circles:
        if isinstance(c, dict):
            center = np.asarray(c["center"], dtype=np.float64).reshape(2)
            radius = float(c["radius"])
        else:
            cx, cy, r = c
            center = np.asarray([cx, cy], dtype=np.float64)
            radius = float(r)
        ell.append({"center": center, "radius": radius})
    return ell

def _infer_cond_dim(model):
    # Try model.cond_dim first
    if hasattr(model, "cond_dim") and model.cond_dim is not None:
        return int(model.cond_dim)
    # Try to infer from cond_mlp first Linear layer
    cond_mlp = getattr(model, "cond_mlp", None)
    if cond_mlp is not None:
        for layer in cond_mlp:
            if hasattr(layer, "in_features"):
                return int(layer.in_features)
    return None

# -------------------------
# GOAL-only canonicalization (NO virtual obstacle)
#   - define canonical frame by rotating so goal is on +x
#   - optional reflect so goal is on +y (consistent sign convention)
# -------------------------
def canonicalize_runtime_goal_cond(
    xy_world,
    goal_xy,
    *,
    theta0=0.0,
    v0=0.3,
    delta0=0.0,
    cond_dim=5,
):
    """
    Returns:
      cond: (cond_dim,)
      meta: {phi, reflect}
    """
    p = np.asarray(xy_world, dtype=np.float64).reshape(2)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    g0 = g - p
    goal_dist = float(np.linalg.norm(g0) + 1e-12)

    phi = float(np.arctan2(g0[1], g0[0]))
    R = _rot2(-phi)

    g_can = (R @ g0.reshape(2, 1)).reshape(2)

    # reflect rule WITHOUT obstacle: keep goal in +y half-plane (optional but makes sign consistent)
    reflect = False
    if float(g_can[1]) < 0.0:
        reflect = True
        g_can[1] *= -1.0

    theta0_can = _wrap_pi(float(theta0) - phi)
    delta0_can = float(delta0)
    if reflect:
        theta0_can = -theta0_can
        delta0_can = -delta0_can

    # ---- cond construction ----
    if int(cond_dim) == 5:
        # [goal_dist, v0, cos(theta0_can), sin(theta0_can), delta0_can]
        cond = np.array(
            [
                goal_dist,
                float(v0),
                float(np.cos(theta0_can)),
                float(np.sin(theta0_can)),
                float(delta0_can),
            ],
            dtype=np.float32,
        )
    else:
        raise ValueError(
            f"Unsupported cond_dim={cond_dim} for NO-OBS rollout. "
            f"Expected 5 (goal-only). Your model_vel cond_dim={cond_dim}."
        )

    meta = {"phi": phi, "reflect": reflect}
    return cond, meta

def decanonicalize_vxy(vxy_can_2, meta):
    v = np.asarray(vxy_can_2, dtype=np.float64).reshape(2).copy()
    if bool(meta.get("reflect", False)):
        v[1] *= -1.0
    Rinv = _rot2(float(meta["phi"]))
    return (Rinv @ v.reshape(2, 1)).reshape(2)

# -------------------------
# ODE sampling with CBF inside
#   - FM-ODE time is still [0,1] via T_SPAN (unchanged)
#   - We ONLY use dt (world step) to roll x_sim during safety check
#     (this does NOT change FM-ODE time semantics)
# -------------------------
@torch.no_grad()
def sample_v_seq_safe_odeint(
    *,
    model,
    cond,
    seq_len: int,
    T_SPAN: torch.Tensor,
    solver_method="euler",
    solver_tol=1e-5,
    noise_scale=1.0,
    device="cpu",
    x0_world_xy,
    ellipses_true,
    safety_margin=0.12,
    gamma_min=6.0,
    max_corr_norm=10.0,
    dt=0.1,  # world step for simulating x during CBF checking
    phi_for_world=0.0,
    reflect_for_world=False,
    cbf_project_velocity_world,
):
    dev = torch.device(device)
    model.eval()

    cond_t = torch.as_tensor(cond, dtype=torch.float32, device=dev).view(1, -1)
    x0 = torch.randn(1, 2, int(seq_len), device=dev) * float(noise_scale)

    t_span = T_SPAN.to(dev)
    dt_ode = float((t_span[1] - t_span[0]).item())

    x0_world_xy = np.asarray(x0_world_xy, dtype=np.float64).reshape(2)
    phi = float(phi_for_world)
    reflect = bool(reflect_for_world)

    Rinv = _rot2(phi)     # canonical -> world
    Rcan = _rot2(-phi)    # world -> canonical

    def fm_ode_func(t, x_flat):
        v_can = x_flat.view(1, 2, int(seq_len))
        t_tensor = torch.tensor([t], device=v_can.device, dtype=v_can.dtype)

        # FM velocity field in sequence-space
        v_nom = model(v_can, t_tensor, cond=cond_t)  # (1,2,H)

        # one Euler step in ODE space -> candidate sequence
        v_can_next = (v_can + dt_ode * v_nom)
        v_seq_can = v_can_next.detach().cpu().numpy()[0]  # (2,H)

        # canonical -> world for CBF checking
        v_seq_tmp = v_seq_can.copy()
        if reflect:
            v_seq_tmp[1, :] *= -1.0
        v_seq_world = (Rinv @ v_seq_tmp)  # (2,H)

        # project step-by-step in world with CBF
        x_world = x0_world_xy.copy()
        v_safe_world = np.zeros_like(v_seq_world)

        for k in range(int(seq_len)):
            v_nom_k = v_seq_world[:, k].copy()

            x_traj = torch.tensor(x_world, dtype=torch.float32, device=dev).view(1, 2, 1)
            v_traj = torch.tensor(v_nom_k, dtype=torch.float32, device=dev).view(1, 2, 1)

            v_safe_k = cbf_project_velocity_world(
                x_traj,
                v_traj,
                ellipses=ellipses_true,
                extra_margin=float(safety_margin),
                max_corr_norm=float(max_corr_norm),
                passes=5,
                other_agents_traj=None,
                agent_safe_radius=0.5,
                gamma_min=float(gamma_min),
                w_p=0.0,
                press_k=1.0,
                press_n=2.0,
            ).detach().cpu().numpy().reshape(2)

            v_safe_world[:, k] = v_safe_k
            x_world = x_world + float(dt) * v_safe_k  # world step for prediction

        # world -> canonical for ODE state target
        v_safe_tmp = (Rcan @ v_safe_world)  # (2,H)
        if reflect:
            v_safe_tmp[1, :] *= -1.0

        v_safe_can_t = torch.as_tensor(v_safe_tmp, dtype=v_can.dtype, device=v_can.device).unsqueeze(0)
        dv_safe = (v_safe_can_t - v_can) / max(dt_ode, 1e-9)
        return dv_safe.view(-1)

    sol = torchdiffeq.odeint(
        fm_ode_func,
        x0.view(-1),
        t_span,
        atol=float(solver_tol),
        rtol=float(solver_tol),
        method=str(solver_method),
    )
    v_final_can = sol[-1].view(1, 2, int(seq_len))
    return v_final_can[0].detach().cpu().numpy().T.astype(np.float64)  # (H,2)

# -------------------------
# Outer rollout (execute only v0) + EMA + goal pull
# -------------------------
def rollout_model_vel_with_cbf_inside_ode(
    x0_xy,
    goal_xy,
    *,
    model,
    cbf_project_velocity_world,
    circles,
    dt=0.1,
    v_max=1.4,
    max_total_steps=300,
    goal_tol=0.15,
    H=20,
    n_steps_sample=60,
    noise_scale=1.0,
    device="cpu",
    solver_method="euler",
    solver_tol=1e-5,
    safety_margin=0.12,
    gamma_min=6.0,
    max_corr_norm=10.0,
    # smoothing / goal pull
    ema_beta=0.88,
    goal_pull=0.20,
    goal_pull_near=0.55,
    goal_pull_radius=1.2,
    # optional runtime state info (if you have it)
    theta0_runtime=0.0,
    v0_runtime=0.3,
    delta0_runtime=0.0,
    cond_dim_model,
):
    x = np.asarray(x0_xy, dtype=np.float64).reshape(2)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    ellipses_true = _circle_list_to_ellipses(circles)
    xs = [x.copy()]

    v_exec_prev = np.zeros(2, dtype=np.float64)
    T_SPAN = torch.linspace(0.0, 1.0, int(n_steps_sample) + 1)

    for _ in range(int(max_total_steps)):
        dist = float(np.linalg.norm(x - g))
        if dist <= float(goal_tol):
            break

        # GOAL-only cond (no obstacle in cond)
        cond, meta = canonicalize_runtime_goal_cond(
            x, g,
            theta0=theta0_runtime,
            v0=v0_runtime,
            delta0=delta0_runtime,
            cond_dim=cond_dim_model,
        )

        # sample v_seq_can with CBF inside ODE (CBF uses REAL obstacles)
        v_seq_can = sample_v_seq_safe_odeint(
            model=model,
            cond=cond,
            seq_len=int(H),
            T_SPAN=T_SPAN,
            solver_method=solver_method,
            solver_tol=solver_tol,
            noise_scale=noise_scale,
            device=device,
            x0_world_xy=x,
            ellipses_true=ellipses_true,
            safety_margin=float(safety_margin),
            gamma_min=float(gamma_min),
            max_corr_norm=float(max_corr_norm),
            dt=float(dt),
            phi_for_world=float(meta["phi"]),
            reflect_for_world=bool(meta["reflect"]),
            cbf_project_velocity_world=cbf_project_velocity_world,
        )

        # execute first velocity
        v0_can = v_seq_can[0]
        v0_world = decanonicalize_vxy(v0_can, meta)
        v0_world = np.clip(v0_world, -float(v_max), float(v_max))

        # --- goal attraction ---
        to_goal = (g - x).astype(np.float64)
        d = float(np.linalg.norm(to_goal) + 1e-12)
        dir_goal = to_goal / d
        s = 1.0 - min(1.0, d / float(goal_pull_radius))
        pull = float(goal_pull) + float(goal_pull_near - goal_pull) * s

        speed = float(np.linalg.norm(v0_world) + 1e-12)
        v_goal = speed * dir_goal
        v0_world = (1.0 - pull) * v0_world + pull * v_goal

        # --- EMA smoothing ---
        v0_world = float(ema_beta) * v_exec_prev + (1.0 - float(ema_beta)) * v0_world
        v_exec_prev = v0_world.copy()

        v0_world = np.clip(v0_world, -float(v_max), float(v_max))

        # step
        x = x + float(dt) * v0_world
        xs.append(x.copy())

    return np.asarray(xs, dtype=np.float64), ellipses_true

def run_rollout_static(
    *,
    model_vel,
    cbf_project_velocity_world,
    x0_xy=None,
    goal_xy=None,
    circles=None,
    dt=0.1,
    v_max=1.0,
    max_total_steps=350,
    goal_tol=0.15,
    H=20,
    n_steps_sample=60,
    noise_scale=1.0,
    safety_margin=0.12,
    gamma_min=12.0,
    max_corr_norm=25.0,
    device="cpu",
    solver_method="euler",
    solver_tol=1e-5,
    ema_beta=0.9,
    goal_pull=0.20,
    goal_pull_near=0.55,
    goal_pull_radius=1.2,
    theta0_runtime=0.0,
    v0_runtime=0.3,
    delta0_runtime=0.0,
):
    cond_dim_model = _infer_cond_dim(model_vel)
    if cond_dim_model is None:
        raise RuntimeError("Could not infer cond_dim from model_vel.")
    print(f"[rollout] model_vel expects cond_dim = {cond_dim_model}")

    if x0_xy is None:
        x0_xy = np.array([0.0, 0.0], dtype=np.float64)
    if goal_xy is None:
        goal_xy = np.array([1.5, 2.0], dtype=np.float64)
    if circles is None:
        mid = x0_xy + 0.55 * (goal_xy - x0_xy)
        circles = [{"center": mid + np.array([0.20, 0.0], dtype=np.float64), "radius": 0.35}]

    xs_mpc, circles_true = rollout_model_vel_with_cbf_inside_ode(
        x0_xy,
        goal_xy,
        model=model_vel,
        cbf_project_velocity_world=cbf_project_velocity_world,
        circles=circles,
        dt=float(dt),
        v_max=float(v_max),
        max_total_steps=int(max_total_steps),
        goal_tol=float(goal_tol),
        H=int(H),
        n_steps_sample=int(n_steps_sample),
        noise_scale=float(noise_scale),
        safety_margin=float(safety_margin),
        gamma_min=float(gamma_min),
        max_corr_norm=float(max_corr_norm),
        device=str(device),
        solver_method=str(solver_method),
        solver_tol=float(solver_tol),
        ema_beta=float(ema_beta),
        goal_pull=float(goal_pull),
        goal_pull_near=float(goal_pull_near),
        goal_pull_radius=float(goal_pull_radius),
        theta0_runtime=float(theta0_runtime),
        v0_runtime=float(v0_runtime),
        delta0_runtime=float(delta0_runtime),
        cond_dim_model=int(cond_dim_model),
    )

    return xs_mpc, circles_true


def plot_rollout_static(xs_mpc, goal_xy, circles_true, *, safety_margin=0.12):
    th_grid = np.linspace(0.0, 2 * np.pi, 360)
    plt.figure(figsize=(7, 7))
    plt.plot(xs_mpc[:, 0], xs_mpc[:, 1], "-", linewidth=2.6, label="model_vel (CBF inside ODE) + EMA + goalpull")
    plt.scatter([xs_mpc[0, 0]], [xs_mpc[0, 1]], s=90, label="start")
    plt.scatter([goal_xy[0]], [goal_xy[1]], s=180, marker="*", label="goal")

    margin = float(safety_margin)
    for e in circles_true:
        c = np.asarray(e["center"], dtype=np.float64).reshape(2)
        R = float(e["radius"])
        plt.plot(c[0] + R * np.cos(th_grid), c[1] + R * np.sin(th_grid), "r-", linewidth=2.0, label="obstacle (true)")
        plt.fill(c[0] + R * np.cos(th_grid), c[1] + R * np.sin(th_grid), color="red", alpha=0.10)
        R_safe = R + margin
        plt.plot(c[0] + R_safe * np.cos(th_grid), c[1] + R_safe * np.sin(th_grid), "r--", linewidth=1.8, label="safety boundary")

    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title("Rollout with model_vel (NO virtual obs cond) + CBF inside FM-ODE + EMA + goal pull")
    plt.legend()
    plt.show()

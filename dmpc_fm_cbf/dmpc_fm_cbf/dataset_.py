import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# =========================
# Utils
# =========================
def _wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# =========================
# 1) Vehicle rollout (bicycle)
# =========================
def simulate_vehicle_bicycle(
    x0,            # [x,y,theta,v,delta]
    u_seq,         # (T-1,2) = [a, delta_rate]
    dt,
    L,    # 轴距
    delta_max,
    v_min=0.0, # 不允许倒车 负速度
):
    u_seq = np.asarray(u_seq, dtype=np.float64)  # 转换成 numpy
    Tm1 = u_seq.shape[0]
    T = Tm1 + 1 #T 个状态，只需要 T−1 个控制

    states = np.zeros((T, 5), dtype=np.float64) 
    states[0] = np.asarray(x0, dtype=np.float64)

    for k in range(Tm1):
        x, y, th, v, delta = states[k]
        a, delta_rate = u_seq[k]

        # steering integration
        delta_next = delta + dt * delta_rate
        delta_next = np.clip(delta_next, -delta_max, delta_max)

        # speed integration
        v_next = v + dt * a
        v_next = max(v_min, v_next)

        # kinematic bicycle
        th_next = _wrap_to_pi(th + dt * (v / L) * np.tan(delta))
        x_next = x + dt * v * np.cos(th)
        y_next = y + dt * v * np.sin(th)

        states[k+1] = [x_next, y_next, th_next, v_next, delta_next]

    return states

# =========================
# 2) Baseline reference (the "goal-attracted" curve you like)
# =========================
def make_baseline_reference_bicycle(
    start_xy, theta0, goal_xy,
    T, dt, L, delta_max, delta_rate_max,
    v_ref,
    turn_scale=3.0,
    k_theta=3.0,
):
    x, y = float(start_xy[0]), float(start_xy[1])
    theta = float(theta0)
    delta = 0.0

    dir_init = np.array([np.cos(theta0), np.sin(theta0)], dtype=np.float64)
    ref_xy = np.zeros((T, 2), dtype=np.float64)
    ref_xy[0] = [x, y]

    for t in range(1, T):
        vec_goal = goal_xy - np.array([x, y], dtype=np.float64)
        dist = float(np.linalg.norm(vec_goal) + 1e-8)
        dir_goal = vec_goal / dist

        w = np.exp(-t / (T / turn_scale))
        dir_mix = w * dir_init + (1.0 - w) * dir_goal
        dir_mix /= np.linalg.norm(dir_mix) + 1e-8

        theta_ref = float(np.arctan2(dir_mix[1], dir_mix[0]))

        # heading P-control -> yaw_rate_des -> delta*
        e_theta = _wrap_to_pi(theta_ref - theta)
        yaw_rate_des = k_theta * e_theta
        delta_star = np.arctan2(L * yaw_rate_des, max(v_ref, 1e-6))
        delta_star = np.clip(delta_star, -delta_max, delta_max)

        # delta rate limit
        max_delta_step = delta_rate_max * dt
        delta += np.clip(delta_star - delta, -max_delta_step, max_delta_step)
        delta = np.clip(delta, -delta_max, delta_max)

        # bicycle rollout with v_ref
        x += v_ref * np.cos(theta) * dt
        y += v_ref * np.sin(theta) * dt
        theta = _wrap_to_pi(theta + (v_ref / L) * np.tan(delta) * dt)

        ref_xy[t] = [x, y]

    return ref_xy

# =========================
# 3) Tracking optimal control cost
# =========================
def make_cost_function_tracking(
    dt, L, delta_max,
    ref_xy, goal_xy,
    tau_track=15.0,
    w_track=80.0,
    w_goal_terminal=1200.0,
    w_v_terminal=120.0,
    w_u=0.02,
    w_delta_stage=0.1,
):
    T = ref_xy.shape[0]
    ts = np.arange(T, dtype=np.float64)
    wt = np.exp(-ts / tau_track)  # early strong, later weak

    def cost(u_flat, x0, theta_goal_unused, control_steps):
        u_seq = u_flat.reshape((control_steps, 2))
        states = simulate_vehicle_bicycle(
            x0=x0, u_seq=u_seq, dt=dt, L=L, delta_max=delta_max, v_min=0.0
        )

        x = states[:, 0]
        y = states[:, 1]
        v = states[:, 3]
        delta = states[:, 4]

        # tracking (position)
        ex = x - ref_xy[:, 0]
        ey = y - ref_xy[:, 1]
        track_err2 = ex*ex + ey*ey
        J_track = w_track * np.sum(wt * track_err2)

        # terminal goal + stop
        xf, yf, _, vf, _ = states[-1]
        pos_goal = (xf - goal_xy[0])**2 + (yf - goal_xy[1])**2
        J_terminal = w_goal_terminal * pos_goal + w_v_terminal * (vf*vf)

        # control effort + steering magnitude
        J_u = w_u * np.sum(u_seq*u_seq)
        J_delta = w_delta_stage * np.sum(delta*delta)

        return J_track + J_terminal + J_u + J_delta

    return cost

# =========================
# 4) Dataset generation (tracking OC)
# =========================
def make_center_to_ring_dataset_tracking_oc(
    N=100,
    T=60,
    start=(0.0, 0.0),
    goal_center=(0.0, 0.0),
    radius=3.0,
    seed=42,
    save_filename="car_ring_track_oc_dataset.npy",
    LOAD_SAVED_DATA=False,

    # vehicle
    dt=0.1,
    L=0.8,
    delta_max=np.deg2rad(30.0),
    a_max=0.8,
    delta_rate_max=np.deg2rad(120.0),

    # initial speed range in x0 (important)
    v0_min=0.15,
    v0_max=0.6,

    # baseline ref params
    turn_scale=3.0,
    k_theta=3.0,

    # tracking cost params
    tau_track_ratio=0.8,   # tau_track = ratio*T
    w_track=160,
    w_goal_terminal=3500,
    w_v_terminal=120.0,
    w_u=0.02,
    w_delta_stage=0.1,

    # optimizer
    maxiter=600,
):
    rng = np.random.default_rng(seed)

    if LOAD_SAVED_DATA and os.path.exists(save_filename):
        print("Loading:", save_filename)
        return np.load(save_filename, allow_pickle=True).item()

    # goals on ring
    gc = np.array(goal_center, dtype=np.float64)
    phi = np.linspace(-np.pi, np.pi, N, endpoint=False).astype(np.float64)
    goals = gc + radius * np.stack([np.cos(phi), np.sin(phi)], axis=1)

    # initial headings
    thetas0 = rng.uniform(-np.pi, np.pi, size=N).astype(np.float64)

    control_steps = T - 1
    bounds = []
    for _ in range(control_steps):
        bounds.append((-a_max, a_max))
        bounds.append((-delta_rate_max, delta_rate_max))

    traj_xy = np.zeros((N, 2, T), dtype=np.float64)
    states_all = np.zeros((N, T, 5), dtype=np.float64)
    controls_all = np.zeros((N, T, 2), dtype=np.float64)
    refs_all = np.zeros((N, T, 2), dtype=np.float64)

    x0_global, y0_global = float(start[0]), float(start[1])

    print("Generating tracking optimal-control ring dataset ...")
    for i in range(N):
        goal = goals[i]
        theta0 = float(thetas0[i])

        # initial state
        v0_i = float(rng.uniform(float(v0_min), float(v0_max)))
        x0 = np.array([x0_global, y0_global, theta0, v0_i, 0.0], dtype=np.float64)

        # baseline reference speed: use the same v0_i
        v_ref = float(x0[3])

        # baseline ref trajectory (the shape you like)
        ref_xy = make_baseline_reference_bicycle(
            start_xy=np.array([x0_global, y0_global], dtype=np.float64),
            theta0=theta0,
            goal_xy=goal,
            T=T,
            dt=dt,
            L=L,
            delta_max=delta_max,
            delta_rate_max=delta_rate_max,
            v_ref=v_ref,
            turn_scale=turn_scale,
            k_theta=k_theta,
        )
        refs_all[i] = ref_xy

        # tracking cost for this trajectory
        cost_fn = make_cost_function_tracking(
            dt=dt, L=L, delta_max=delta_max,
            ref_xy=ref_xy, goal_xy=goal,
            tau_track=tau_track_ratio * T,
            w_track=w_track,
            w_goal_terminal=w_goal_terminal,
            w_v_terminal=w_v_terminal,
            w_u=w_u,
            w_delta_stage=w_delta_stage,
        )

        # initial guess
        u0 = np.zeros(control_steps * 2, dtype=np.float64)

        # optimize
        res = minimize(
            fun=cost_fn,
            x0=u0,
            args=(x0, 0.0, control_steps),  # theta_goal unused
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-6, "disp": False},
        )

        u_opt = res.x.reshape((control_steps, 2))
        states = simulate_vehicle_bicycle(
            x0=x0, u_seq=u_opt, dt=dt, L=L, delta_max=delta_max, v_min=0.0
        )

        states_all[i] = states
        traj_xy[i, 0, :] = states[:, 0]
        traj_xy[i, 1, :] = states[:, 1]

        controls_padded = np.zeros((T, 2), dtype=np.float64)
        controls_padded[:-1] = u_opt
        controls_all[i] = controls_padded

        if (i + 1) % max(1, N // 10) == 0:
            print(f"{(i+1)/N*100:.0f}% ({i+1}/{N}) done. last success={res.success}")

    data = {
        "traj_xy": traj_xy.astype(np.float32),
        "states": states_all.astype(np.float32),
        "controls": controls_all.astype(np.float32),
        "refs_xy": refs_all.astype(np.float32),
        "goals": goals.astype(np.float32),
        "theta0": thetas0.astype(np.float32),
        "params": {
            "dt": float(dt), "L": float(L), "delta_max": float(delta_max),
            "a_max": float(a_max), "delta_rate_max": float(delta_rate_max),
            "T": int(T), "N": int(N), "radius": float(radius),
            "start": start, "goal_center": goal_center,
            "v0_min": float(v0_min),
            "v0_max": float(v0_max),
            "turn_scale": float(turn_scale), "k_theta": float(k_theta),
            "tau_track_ratio": float(tau_track_ratio),
            "w_track": float(w_track),
            "w_goal_terminal": float(w_goal_terminal),
            "w_v_terminal": float(w_v_terminal),
            "w_u": float(w_u),
            "w_delta_stage": float(w_delta_stage),
            "save_filename": save_filename,
        }
    }

    np.save(save_filename, data, allow_pickle=True)
    print("Saved:", save_filename)
    return data

# =========================
# 5) Plotting cells
# =========================
def plot_all_trajs_timecolor(
    traj_xy,
    goals,
    title,
    theta0=None,
    n_show=16,
    arrow_every=4,
    show_circle=True,
    align_start_to_right=True,
):
    """Dataset overview plot in the style you sketched.

    Key visualization idea:
    - translate start -> origin
    - (optionally) rotate each trajectory so its initial heading points to +x (right)
      so the plot shows: "go straight right first, then turn by different angles".

    Notes:
    - This is ONLY a plotting transform. Simulation/control logic is unchanged.

    Args:
      traj_xy: (N,2,T)
      goals:   (N,2)
      theta0:  (N,) optional. If given and align_start_to_right=True, we rotate by -theta0.
      n_show:  how many trajectories to display (spread over angles)
    """
    traj_xy = np.asarray(traj_xy, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)

    N, _, T = traj_xy.shape

    def _rot2(a):
        ca, sa = float(np.cos(a)), float(np.sin(a))
        return np.array([[ca, -sa], [sa, ca]], dtype=np.float64)

    # choose an ordering angle (for selecting a representative subset)
    if theta0 is not None:
        base_angles = np.asarray(theta0, dtype=np.float64)
    else:
        base_angles = np.arctan2(goals[:, 1], goals[:, 0]).astype(np.float64)

    n_show = int(min(max(1, n_show), N))
    order = np.argsort(base_angles)
    pick = order[np.linspace(0, N - 1, n_show).astype(int)]

    fig, ax = plt.subplots(figsize=(7, 7))

    # dashed ring (goal radius)
    if show_circle:
        r = float(np.mean(np.linalg.norm(goals, axis=1)))
        tt = np.linspace(0.0, 2 * np.pi, 256)
        ax.plot(r * np.cos(tt), r * np.sin(tt), "k--", alpha=0.35, linewidth=1.2, label="goal ring")

    # start point is always origin after alignment
    ax.scatter([0.0], [0.0], s=90, c="red", label="start")

    cmap = plt.cm.hsv
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    # If theta0 is NOT provided, DO NOT infer per-trajectory heading from the first segment.
    # That inference tends to rotate each trajectory so its goal points to +x,
    # collapsing the whole dataset to the right (exactly the issue you saw).
    if align_start_to_right and theta0 is None:
        print("[plot_all_trajs_timecolor] WARNING: theta0 is None; alignment disabled to avoid collapsing the plot.")

    # plot trajectories (aligned frame)
    aligned_goals = []
    for k, i in enumerate(pick):
        xy = traj_xy[i].T  # (T,2)
        start = xy[0].copy()
        goal = goals[i].copy()

        # translate
        xy0 = xy - start[None, :]
        g0 = goal - start

        # rotate so initial heading is +x (only if theta0 is provided)
        if align_start_to_right and (theta0 is not None):
            a = float(theta0[i])
            R = _rot2(-a)
            xy0 = (R @ xy0.T).T
            g0 = (R @ g0.reshape(2, 1)).reshape(2)

        aligned_goals.append(g0.copy())

        # color by goal direction (after alignment)
        ang = float(np.arctan2(g0[1], g0[0]))
        c = cmap(norm(ang))

        ax.plot(xy0[:, 0], xy0[:, 1], color=c, alpha=0.9, linewidth=1.8)

        # arrow head occasionally
        if arrow_every > 0 and (k % int(arrow_every) == 0) and T >= 2:
            ax.annotate(
                "",
                xy=(float(xy0[-1, 0]), float(xy0[-1, 1])),
                xytext=(float(xy0[-2, 0]), float(xy0[-2, 1])),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2, alpha=0.9),
            )

    # show ALL goals (aligned in the same frame) for context
    aligned_goals = np.asarray(aligned_goals, dtype=np.float64)
    ax.scatter(aligned_goals[:, 0], aligned_goals[:, 1], s=30, marker="x", c="black", alpha=0.55, label="goals")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("goal angle (aligned frame, rad)")

    ax.legend(loc="best")
    plt.show()


def plot_one_compare(data, idx=None):
    traj_xy = data["traj_xy"]
    refs_xy = data["refs_xy"]
    goals = data["goals"]
    controls = data["controls"]
    dt = data["params"]["dt"]

    N, _, T = traj_xy.shape
    if idx is None:
        idx = np.random.randint(N)

    xy = traj_xy[idx].T    # (T,2)
    ref = refs_xy[idx]     # (T,2)
    goal = goals[idx]
    u = controls[idx]      # (T,2)

    # 1) XY compare
    plt.figure(figsize=(6,6))
    plt.plot(ref[:,0], ref[:,1], "k--", linewidth=2, label="baseline ref (heuristic)")
    plt.plot(xy[:,0], xy[:,1], "-o", markersize=3, label="optimized rollout")
    plt.scatter(xy[0,0], xy[0,1], c="green", s=80, label="start")
    plt.scatter(goal[0], goal[1], c="red", s=80, label="goal")
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"Trajectory #{idx}: ref vs optimized")
    plt.legend()
    plt.show()

    # 2) distance to goal over time
    dist = np.linalg.norm(xy - goal[None,:], axis=1)
    plt.figure(figsize=(8,3))
    plt.plot(np.arange(T)*dt, dist)
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel("distance to goal")
    plt.title("Distance-to-goal (should decrease and converge)")
    plt.show()

    # 3) controls
    t = np.arange(T)*dt
    fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    axs[0].plot(t, u[:,0])
    axs[0].set_ylabel("a [m/s^2]")
    axs[0].grid(True)
    axs[0].set_title("Controls")

    axs[1].plot(t, u[:,1])
    axs[1].set_ylabel("delta_rate [rad/s]")
    axs[1].set_xlabel("time [s]")
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()


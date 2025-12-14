#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pinocchio + MeshCat — Panda 7-DoF (with stick) — stable YZ tracking (circle/figure-8)

Features:
- Resolved-rate IK (6D) with damped pseudo-inverse, SVD-based speed scaling, joint-velocity limiting.
- Inner loop: computed torque (RNEA) with acc-level PD on joint space: ddq_cmd = Kp*(q_ref - q) + Kv*(qdot_ref - dq).
- Gentle startup: initial hold + amplitude/frequency ramp.
- Torque spike suppression: low-pass on qdot_ref & tau, and torque slew-rate limit.
- Plots include: joint trajectories/torques AND end-effector YZ path (actual vs desired).

Install:
  uv pip install "pinocchio[meshcat]>=2.7.0" meshcat numpy matplotlib
"""

import os, time, numpy as np
import argparse
import pinocchio as pin

# Try official MeshcatVisualizer
PinMeshcatVis = None
try:
    from pinocchio.visualize import MeshcatVisualizer as PinMeshcatVis
except Exception:
    pass

# Raw meshcat fallback
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    HAS_MESHCAT = True
except Exception:
    HAS_MESHCAT = False
    print("[warn] meshcat not found; running headless. `pip/uv pip install meshcat` to enable viz.")

# ---------- helpers ----------
def so3_log(R):
    tr = np.trace(R); c = np.clip((tr-1.0)*0.5, -1.0, 1.0); th = np.arccos(c)
    if th < 1e-8:
        return 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    w = (1.0/(2.0*np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return th*w

def ema(prev, new, alpha):
    """Exponential moving average: alpha in (0,1], closer to 1 = stronger smoothing."""
    return alpha*prev + (1.0-alpha)*new

def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
    """在 [0, 2π/ω) 里采样找最近相位（只用位置），返回 phase0。"""
    thetas = np.linspace(0.0, 2.0*np.pi/omega, K, endpoint=False)
    y  = cy + Ay*np.sin(omega*thetas)
    z  = cz + Az*np.sin(2.0*omega*thetas + phi)
    i  = np.argmin((y - y0)**2 + (z - z0)**2)
    return float(thetas[i])

def land_to_phase(model, data, ee_fid, q, dq,
                  x_plane, cy, cz, Ay, Az, omega, phi,
                  phase0, dt,
                  kp_pos, kp_ori, damp, qdot_lim, rate_alpha,
                  Kp_j, Kv_j,
                  iters=400, tol=1e-4, viz_cb=None):
    """把 (q,dq) 移动到给定相位 phase0 的落点（位置+可选姿态）。返回新 (q,dq)。"""
    # 相位对应的期望
    y  = cy + Ay*np.sin(omega*phase0)
    z  = cz + Az*np.sin(2.0*omega*phase0 + phi)
    vy = Ay*omega*np.cos(omega*phase0)
    vz = 2.0*Az*omega*np.cos(2.0*omega*phase0 + phi)
    x_des = np.array([x_plane, y, z], dtype=float)
    v_des = np.array([0.0, vy, vz], dtype=float)

    # 姿态：水平 + 切向（若速度很小则退化为水平）
    ez = np.array([1.,0.,0.])
    tang = np.array([0.0, vy, vz])
    if np.linalg.norm(tang) < 1e-8:
        # 纯水平
        up = np.array([0.,0.,1.]); ex = np.cross(up, ez); ex /= np.linalg.norm(ex); ey = np.cross(ez, ex)
        R_des = np.column_stack([ex, ey, ez])
    else:
        ey = tang/np.linalg.norm(tang); ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
        R_des = np.column_stack([ex, ey, ez])

    # 迭代到位：解析速度 → 关节级加速度 PD → RNEA → 半隐式欧拉
    for it in range(iters):
        q_ref, qdot_ref, _, _ = resolved_rate_step6(
            model, data, ee_fid, q, x_des, v_des, R_des,
            kp_pos=kp_pos, kp_ori=kp_ori, base_damp=damp,
            qdot_limit=qdot_lim, rate_alpha=rate_alpha, dt=dt
        )
        e  = pin.difference(model, q, q_ref)
        ed = (qdot_ref - dq)
        ddq_cmd = Kp_j * e + Kv_j * ed
        tau = pin.rnea(model, data, q, dq, ddq_cmd)
        M = pin.crba(model, data, q); h = pin.nle(model, data, q, dq)
        ddq = np.linalg.solve(M, tau - h)
        dq  = dq + ddq*dt
        q   = pin.integrate(model, q, dq*dt)

        if viz_cb is not None and it % 5 == 0: viz_cb(q)

        # 收敛判据：末端位置误差
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        p = data.oMf[ee_fid].translation
        if np.linalg.norm(p - x_des) < tol: break

    # 最后再"对齐相位"：以当前 q 计算的实际 y,z 再做一次 nearest_phase，更新 phase0
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    p = data.oMf[ee_fid].translation
    phase0 = nearest_phase(p[1], p[2], cy, cz, Ay, Az, omega, phi)
    return q, dq, phase0

def resolved_rate_step6(model, data, frame_id, q,
                        x_des, v_des, R_des,
                        kp_pos=6.0, kp_ori=0.0,
                        base_damp=0.10,
                        qdot_limit=1.8,
                        rate_alpha=0.5,
                        dt=0.002):
    """One IK step -> (q_ref_next, qdot_ref, smin, pos_err)"""
    # FK & placements
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[frame_id]; p = oMf.translation; R = oMf.rotation

    # Cartesian command
    v_cmd = v_des + kp_pos*(x_des - p)
    w_cmd = kp_ori * so3_log(R_des @ R.T) if kp_ori > 0 else np.zeros(3)

    # Frame Jacobian (local-world-aligned) 6xnv
    J6 = pin.computeFrameJacobian(model, data, q, frame_id,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # Conditioning info
    s = np.linalg.svd(J6, compute_uv=False)
    smin = float(s[-1]) if s.size else 0.0

    # SVD-based speed scaling (near singularities → slower, but less conservative)
    thr = 0.005
    speed_scale = 1.0 if smin > thr else np.clip(smin/thr, 0.6, 1.0)
    v_cmd *= speed_scale
    w_cmd *= speed_scale

    y = np.hstack([v_cmd, w_cmd])

    lam = max(base_damp, 0.05)
    JJt_reg = J6 @ J6.T + (lam**2)*np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt_reg, y)

    # Joint velocity limit
    lim = np.full(model.nv, qdot_limit, dtype=float)
    scale = np.max(np.abs(qdot)/lim) if lim.size else 0.0
    if scale > 1.0:
        qdot /= scale

    q_next = pin.integrate(model, q, rate_alpha * qdot * dt)
    return q_next, qdot, smin, np.linalg.norm(x_des - p)

# ---------- fallback skeleton visualizer ----------
class SimpleChainViz:
    def __init__(self, model):
        assert HAS_MESHCAT, "meshcat not available"
        self.viz = meshcat.Visualizer()
        for j in range(1, model.njoints):
            self.viz[f"joints/j{j}"].set_object(g.Sphere(0.015))
        self.viz["ee/axis"].set_object(g.AxisHelper(0.08))
        self.viz["ee/tip"].set_object(g.Sphere(0.01))
    def display(self, model, data, ee_SE3):
        for j in range(1, model.njoints):
            T = data.oMi[j]
            M = np.eye(4); M[:3,:3] = T.rotation; M[:3,3] = T.translation
            self.viz[f"joints/j{j}"].set_transform(M)
        Mee = np.eye(4); Mee[:3,:3] = ee_SE3.rotation; Mee[:3,3] = ee_SE3.translation
        self.viz["ee/axis"].set_transform(Mee)
        self.viz["ee/tip"].set_transform(Mee)
    @property
    def url(self): return self.viz.url()

def main():
    # === Parse arguments ===
    parser = argparse.ArgumentParser(description='Panda FR3 draw figure-8 with Pinocchio+MeshCat')
    parser.add_argument('--cyc', type=float, default=1.5,
                        help='Number of cycles to draw (default: 1.5)')
    parser.add_argument('--speed', type=float, default=2.5,
                        help='Visualization slowdown factor (default: 2.5, higher=slower)')
    parser.add_argument('--replay', action='store_true',
                        help='Replay trajectory after simulation and compute state differences')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for initial position (default: None for random)')
    parser.add_argument('--init-steady-state', type=str, default=None,
                        help='Path to steady-state cycle file (.npz) for initialization')
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[init] Using random seed: {args.seed}")
    
    # === Load steady-state cycle if provided ===
    steady_state_data = None
    if args.init_steady_state is not None:
        steady_state_path = args.init_steady_state
        if not os.path.isabs(steady_state_path):
            steady_state_path = os.path.join(os.path.dirname(__file__), steady_state_path)
        
        if os.path.isfile(steady_state_path):
            steady_state_data = np.load(steady_state_path)
            print(f"[init] Loaded steady-state cycle from: {steady_state_path}")
            print(f"[init]   - Frames: {steady_state_data['n_frames']}")
            print(f"[init]   - Cycle duration: {steady_state_data['cycle_duration']:.4f}s")
            print(f"[init]   - Closure error: q={steady_state_data['q_error']:.6f}, dq={steady_state_data['dq_error']:.6f}")
        else:
            print(f"[warn] Steady-state file not found: {steady_state_path}")
            print(f"[warn] Will use default initialization")
            steady_state_data = None
    
    # === Load model ===
    URDF_PATH = os.path.join(os.path.dirname(__file__),
                             "panda_description", "urdf", "panda_stick.urdf")
    assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"
    PACKAGE_DIRS = [os.path.join(os.path.dirname(__file__), "panda_description")]

    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, PACKAGE_DIRS)
    data  = model.createData()
    vdata = visual_model.createData()
    print(f"[pin] nq={model.nq}, nv={model.nv}")
    assert model.nv == 7, "Expect 7-DoF arm. Check URDF (no floating base)!"

    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    assert ee_fid < len(model.frames), f"EE frame not found: {EE_NAME}"

    # Initial state (will be overridden if using steady-state)
    initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105], dtype=np.float64)
    q  = initial_qpos.copy()
    dq = np.zeros(model.nv)

    # --- Workspace & trajectory parameters ---
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    p0_ref = data.oMf[ee_fid].translation.copy()

    # Figure-8 parameters (may be overridden by steady-state data)
    if steady_state_data is not None:
        # Use parameters from steady-state recording
        x_plane = float(steady_state_data['x_plane'])
        cy = float(steady_state_data['cy'])
        cz = float(steady_state_data['cz'])
        Ay = float(steady_state_data['Ay'])
        Az = float(steady_state_data['Az'])
        omega = float(steady_state_data['omega'])
        phi = float(steady_state_data['phi'])
        dt = float(steady_state_data['dt'])
        print(f"[init] Using trajectory parameters from steady-state file")
    else:
        # Default parameters
        x_plane = p0_ref[0] + 0.25
        cy, cz = p0_ref[1], p0_ref[2] + 0.12
        Ay, Az = 0.10, 0.05
        omega, phi = 0.6, 0.0
        dt = 0.002
    
    # Initialization depends on whether we have steady-state data
    if steady_state_data is None:
        # Original random initialization + landing
        # Randomize initial EE position on the same x-plane, in a larger area than the figure-8
        # Figure-8 range: Y in [cy-Ay, cy+Ay], Z in [cz-Az, cz+Az]
        # Randomize in a slightly larger range (1.5x)
        random_scale = 1.5
        y_rand = cy + random_scale * Ay * (2.0 * np.random.rand() - 1.0)  # cy ± 1.5*Ay
        z_rand = cz + random_scale * Az * (2.0 * np.random.rand() - 1.0)  # cz ± 1.5*Az
        target_ee_init = np.array([x_plane, y_rand, z_rand])
        
        print(f"[init] Randomized target EE position: Y={y_rand:.4f}, Z={z_rand:.4f}")
        print(f"[init] Figure-8 center: Y={cy:.4f}, Z={cz:.4f}, range: Ay={Ay:.4f}, Az={Az:.4f}")
        
        # Use IK to find joint configuration for randomized EE position
        # Simple iterative IK to reach the randomized position
        print("[init] Solving IK for randomized initial position...")
        
        # Target orientation: horizontal with tool z pointing to world x
        ez = np.array([1.,0.,0.])
        up = np.array([0.,0.,1.])
        ey = np.cross(up, ez)
        ey /= np.linalg.norm(ey)
        ex = np.cross(ey, ez)
        R_init = np.column_stack([ex, ey, ez])
    else:
        # Steady-state initialization: pick random frame from cycle
        n_frames = int(steady_state_data['n_frames'])
        random_frame_idx = np.random.randint(0, n_frames)
        q = steady_state_data['q_cycle'][random_frame_idx].copy()
        dq = steady_state_data['dq_cycle'][random_frame_idx].copy()
        phase0 = float(steady_state_data['phase_cycle'][random_frame_idx])
        
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p0 = data.oMf[ee_fid].translation.copy()
        
        print(f"[init] Initialized from steady-state frame {random_frame_idx}/{n_frames}")
        print(f"[init]   - Phase: {phase0:.4f} rad")
        print(f"[init]   - EE position: Y={p0[1]:.4f}, Z={p0[2]:.4f}")
        print(f"[init] Skipping landing phase - starting directly in periodic motion!")
        
        # Skip the rest of initialization
        R_init = None  # Not needed

    TRAJ_MODE = "eight"       # "circle" or "eight"
    RAMP_T = 1.2              # ramp-in duration

    # --- Gains ---
    # IK parameters (tuned for high bandwidth tracking with phase lock)
    kp_pos, kp_ori = 24.0, 0.5  # High position gain, weak orientation constraint
    damp = 0.08                 # Lower damping for faster response
    qdot_lim = 3.5              # Higher joint velocity limit
    rate_alpha = 0.90           # Faster IK integration

    # Joint-space computed-torque gains (acc-level) - high bandwidth
    Kp_j = np.array([120,120,100,100,80,70,60], dtype=float)  # Higher stiffness for tracking
    Kv_j = 2.0 * np.sqrt(Kp_j) * 1.0  # Critical damping
    
    # Solve IK for randomized initial position (only if not using steady-state)
    if steady_state_data is None:
        for _ in range(300):
            q_ref, qdot_ref, _, pos_err = resolved_rate_step6(
                model, data, ee_fid, q, target_ee_init, np.zeros(3), R_init,
                kp_pos=kp_pos, kp_ori=kp_ori, base_damp=damp,
                qdot_limit=qdot_lim, rate_alpha=rate_alpha, dt=dt
            )
            e = pin.difference(model, q, q_ref)
            ddq_cmd = Kp_j * e + Kv_j * (qdot_ref - dq)
            tau = pin.rnea(model, data, q, dq, ddq_cmd)
            M = pin.crba(model, data, q)
            h = pin.nle(model, data, q, dq)
            ddq = np.linalg.solve(M, tau - h)
            dq = dq + ddq * dt
            q = pin.integrate(model, q, dq * dt)
            
            if pos_err < 1e-4:
                break
        
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p0 = data.oMf[ee_fid].translation.copy()
        print(f"[init] Reached initial position: Y={p0[1]:.4f}, Z={p0[2]:.4f}, error={pos_err:.6f}")

    # ---------- Visualization (early init for landing phase) ----------
    full_viz = None
    simple_viz = None
    if HAS_MESHCAT and PinMeshcatVis is not None:
        full_viz = PinMeshcatVis(model, collision_model, visual_model)
        try:
            full_viz.initViewer(open=True)
        except Exception:
            full_viz.initViewer(open=False)
        full_viz.loadViewerModel()
        full_viz.display(q)
        print("[viz] pinocchio.visualize.MeshcatVisualizer ON")
        print("URL:", full_viz.viewer.url())
    elif HAS_MESHCAT:
        simple_viz = SimpleChainViz(model)
        print("[viz] fallback skeleton viz ON")
        print("URL:", simple_viz.url)
    else:
        print("[viz] headless (no meshcat)")

    # ---------- Landing to nearest phase (only if not using steady-state) ----------
    if steady_state_data is None:
        print("[init] Finding nearest phase on figure-8...")
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p0 = data.oMf[ee_fid].translation.copy()
        phase0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, omega, phi)
        print(f"[init] Nearest phase estimate: {phase0_guess:.4f} rad")
        
        def _viz_cb(q_now):
            if full_viz is not None:
                full_viz.display(q_now)
            elif simple_viz is not None:
                pin.forwardKinematics(model, data, q_now)
                pin.updateFramePlacements(model, data)
                simple_viz.display(model, data, data.oMf[ee_fid])
        
        print("[init] Landing to phase point...")
        q, dq, phase0 = land_to_phase(
            model, data, ee_fid, q, dq,
            x_plane, cy, cz, Ay, Az, omega, phi,
            phase0_guess, dt,
            kp_pos, kp_ori, damp, qdot_lim, rate_alpha,
            Kp_j, Kv_j,
            iters=500, tol=5e-5,
            viz_cb=_viz_cb if HAS_MESHCAT else None
        )
        print(f"[init] Landed on phase0={phase0:.4f} rad (8-curve)")
    # else: phase0 already set from steady_state_data

    # Store initial state for replay (after landing)
    q_init = q.copy()
    dq_init = dq.copy()

    # Calculate total time based on number of cycles (no INITIAL_HOLD needed)
    T_CYCLE = 2.0 * np.pi / omega
    NUM_CYCLES = args.cyc
    T_TOTAL = NUM_CYCLES * T_CYCLE
    
    # Trajectory function using phase state (for phase-lock control)
    def desired_pose_from_phase(s):
        """
        Generate desired pose from phase state s.
        Returns: x, v, R, tang_vec (tangent velocity vector for phase correction)
        """
        r = 1.0
        Ay_eff, Az_eff = r * Ay, r * Az
        omega_eff = omega
        
        if TRAJ_MODE == "circle":
            y  = cy + Ay_eff*np.sin(omega_eff*s)
            z  = cz + Az_eff*np.cos(omega_eff*s)
            vy = Ay_eff*omega_eff*np.cos(omega_eff*s)
            vz = -Az_eff*omega_eff*np.sin(omega_eff*s)
        else:  # figure-8 (Lissajous 1:2)
            y  = cy + Ay_eff*np.sin(omega_eff*s)
            z  = cz + Az_eff*np.sin(2.0*omega_eff*s + phi)
            vy = Ay_eff*omega_eff*np.cos(omega_eff*s)
            vz = 2.0*Az_eff*omega_eff*np.cos(2.0*omega_eff*s + phi)
        
        x = np.array([x_plane, y, z])
        v = np.array([0.0, vy, vz])
        tang_vec = np.array([0.0, vy, vz])  # Tangent velocity for phase correction
        
        # Orientation: use tangential direction as tool y-axis; tool z-axis -> world x
        ez = np.array([1.,0.,0.])
        tang = np.array([0.0, vy, vz])
        if np.linalg.norm(tang) < 1e-8:
            ey = np.array([0.,1.,0.])
        else:
            ey = tang / np.linalg.norm(tang)
        ex = np.cross(ey, ez)
        n = np.linalg.norm(ex)
        if n < 1e-8:
            up = np.array([0.,0.,1.])
            ey = np.cross(up, ez); ey /= np.linalg.norm(ey)
            ex = np.cross(ey, ez)
        else:
            ex /= n
        
        R = np.column_stack([ex, ey, ez])
        return x, v, R, tang_vec

    # --- Timing ---
    T = T_TOTAL
    SLOWDOWN = args.speed  # wall-clock slowdown for clearer meshcat playback

    # Torque safety / smoothing (relaxed for better tracking)
    TAU_LIM = np.array([85,85,85,85,40,40,40], dtype=float)  # Increased wrist limits
    USE_TAU_CLIP = False    # Disable clipping for max bandwidth (enable if unstable)
    # Low-pass and slew-rate on tau (greatly reduced filtering)
    TAU_LP_ALPHA = 0.25     # Minimal filtering (was 0.90)
    TAU_SLEW = np.full(7, 5e4, dtype=float)  # Near-disable slew rate limit

    # qdot_ref low-pass (minimal filtering for high bandwidth)
    QDOT_LP_ALPHA = 0.15    # Was 0.85, now much weaker

    # Phase-lock control parameters
    K_PHASE = 8.0           # Phase correction gain (3~20, tune for tracking)
    LP_S = 0.2              # Phase velocity smoothing (0~0.5)

    # ---------- Logs ----------
    N = int(T/dt); t = 0.0
    q_log  = np.zeros((N, model.nq))
    dq_log = np.zeros((N, model.nv))
    tau_log= np.zeros((N, model.nv))
    t_log  = np.zeros(N)
    # EE logs (actual & desired)
    ee_pos_log     = np.zeros((N,3))
    ee_pos_des_log = np.zeros((N,3))
    
    # Phase-lock logs
    phase_log = np.zeros(N)
    phase_error_log = np.zeros(N)
    
    # For replay: store all control inputs (ddq_cmd, tau)
    if args.replay:
        ddq_cmd_log = np.zeros((N, model.nv))
        tau_applied_log = np.zeros((N, model.nv))

    last_viz = time.time()
    tau_prev = np.zeros(model.nv)
    qdot_ref_filt = np.zeros(model.nv)
    
    # Initialize phase state (replaces time-based trajectory)
    s = phase0
    s_dot_prev = omega

    print(f"[run] start ... (cycles={NUM_CYCLES:.1f}, T={T:.2f}s, slowdown={SLOWDOWN}x)")
    print(f"[run] Phase-lock enabled: K_PHASE={K_PHASE}, LP_S={LP_S}")
    for i in range(N):
        wall0 = time.perf_counter()
        
        # Generate desired trajectory from current phase state
        x_des, v_des, R_des, tang_vec = desired_pose_from_phase(s)

        # 1) Resolved-rate IK step
        q_ref, qdot_ref, smin, pos_err = resolved_rate_step6(
            model, data, ee_fid, q, x_des, v_des, R_des,
            kp_pos=kp_pos, kp_ori=kp_ori,
            base_damp=damp, qdot_limit=qdot_lim, rate_alpha=rate_alpha, dt=dt
        )

        # qdot_ref low-pass (minimal filtering)
        qdot_ref_filt = ema(qdot_ref_filt, qdot_ref, QDOT_LP_ALPHA)

        # 2) Joint-space acc command
        e  = pin.difference(model, q, q_ref)
        ed = qdot_ref_filt - dq
        ddq_cmd = Kp_j * e + Kv_j * ed

        # 3) Computed torque
        tau_raw = pin.rnea(model, data, q, dq, ddq_cmd)

        # Torque clip + low-pass + slew-rate (minimal filtering)
        if USE_TAU_CLIP:
            tau_raw = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        # one-pole low-pass
        tau_filt = ema(tau_prev, tau_raw, TAU_LP_ALPHA)
        # slew-rate limit
        max_step = TAU_SLEW * dt
        delta = np.clip(tau_filt - tau_prev, -max_step, max_step)
        tau = tau_prev + delta
        tau_prev = tau.copy()

        # 4) Forward dynamics (semi-implicit Euler)
        M  = pin.crba(model, data, q)
        h  = pin.nle (model, data, q, dq)
        ddq= np.linalg.solve(M, tau - h)

        dq = dq + ddq*dt
        q  = pin.integrate(model, q, dq*dt)

        # 5) Phase-lock control: compute tangential error and adjust phase velocity
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        ee = data.oMf[ee_fid].translation.copy()
        
        e_xyz = x_des - ee  # Position error
        tang_norm = np.linalg.norm(tang_vec)
        if tang_norm > 1e-9:
            t_hat = tang_vec / tang_norm
            e_tan = float(e_xyz.dot(t_hat))  # Tangential error (ahead if positive, behind if negative)
        else:
            e_tan = 0.0
        
        # Phase velocity correction: if behind (e_tan > 0), speed up; if ahead, slow down
        s_dot_cmd = omega + K_PHASE * e_tan
        s_dot = (1.0 - LP_S) * s_dot_cmd + LP_S * s_dot_prev
        s += s_dot * dt
        s_dot_prev = s_dot

        # Logs
        q_log[i], dq_log[i], tau_log[i], t_log[i] = q, dq, tau, t
        phase_log[i] = s
        phase_error_log[i] = e_tan
        if args.replay:
            ddq_cmd_log[i] = ddq_cmd
            tau_applied_log[i] = tau

        # EE logs
        ee_pos_log[i] = ee
        ee_pos_des_log[i] = x_des

        # Viz @ ~100Hz
        if HAS_MESHCAT and (time.time()-last_viz > 0.01):
            if full_viz is not None:
                full_viz.display(q)
            elif simple_viz is not None:
                simple_viz.display(model, data, data.oMf[ee_fid])
            last_viz = time.time()

        # Console diag
        if i % 10 == 0:
            print(f"t={t:.3f} | |ex|={pos_err:.3e} | smin(J)={smin:.2e} | e_tan={e_tan:.3e} | s_dot={s_dot:.3f}")

        # Slow playback
        elapsed = time.perf_counter() - wall0
        to_sleep = dt * SLOWDOWN - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

        t += dt
    print(f"[run] done. Total steps: {N}")

    # ---------- Replay ----------
    if args.replay:
        print("\n[replay] Starting replay from initial state...")
        q_replay = q_init.copy()
        dq_replay = dq_init.copy()
        
        q_replay_log = np.zeros((N, model.nq))
        dq_replay_log = np.zeros((N, model.nv))
        
        # Replay with the same torques
        for i in range(N):
            tau_replay = tau_applied_log[i]
            
            # Forward dynamics
            M  = pin.crba(model, data, q_replay)
            h  = pin.nle (model, data, q_replay, dq_replay)
            ddq_replay = np.linalg.solve(M, tau_replay - h)
            
            dq_replay = dq_replay + ddq_replay*dt
            q_replay  = pin.integrate(model, q_replay, dq_replay*dt)
            
            q_replay_log[i] = q_replay
            dq_replay_log[i] = dq_replay
            
            # Viz replay @ ~100Hz
            if HAS_MESHCAT and (time.time()-last_viz > 0.01):
                pin.forwardKinematics(model, data, q_replay)
                pin.updateFramePlacements(model, data)
                if full_viz is not None:
                    full_viz.display(q_replay)
                elif simple_viz is not None:
                    simple_viz.display(model, data, data.oMf[ee_fid])
                last_viz = time.time()
            
            # Slow playback
            if i % 10 == 0:
                elapsed_replay = time.perf_counter() - wall0
                to_sleep_replay = dt * SLOWDOWN - elapsed_replay
                if to_sleep_replay > 0:
                    time.sleep(to_sleep_replay)
                wall0 = time.perf_counter()
        
        print("[replay] done.")
        
        # Compute differences
        q_diff = np.zeros(N)
        dq_diff = np.zeros(N)
        for i in range(N):
            q_diff[i] = np.linalg.norm(pin.difference(model, q_log[i], q_replay_log[i]))
            dq_diff[i] = np.linalg.norm(dq_log[i] - dq_replay_log[i])
        
        max_q_diff = np.max(q_diff)
        max_dq_diff = np.max(dq_diff)
        mean_q_diff = np.mean(q_diff)
        mean_dq_diff = np.mean(dq_diff)
        
        print(f"\n[replay] State differences:")
        print(f"  q:  max={max_q_diff:.6e}, mean={mean_q_diff:.6e}")
        print(f"  dq: max={max_dq_diff:.6e}, mean={mean_dq_diff:.6e}")
        
        # Plot differences
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12,5))
            plt.subplot(2,1,1)
            plt.plot(t_log, q_diff)
            plt.grid(True); plt.ylabel('||q error|| (rad)')
            plt.title(f'Replay State Difference (max_q={max_q_diff:.3e}, max_dq={max_dq_diff:.3e})')
            plt.subplot(2,1,2)
            plt.plot(t_log, dq_diff)
            plt.grid(True); plt.ylabel('||dq error|| (rad/s)')
            plt.xlabel('t (s)')
            plt.tight_layout()
        except Exception:
            pass

    # ---------- Plots ----------
    try:
        import matplotlib.pyplot as plt

        # Joint positions
        plt.figure(figsize=(12,5))
        for j in range(min(7, model.nv)):
            plt.plot(t_log, q_log[:, j], label=f"J{j+1}")
        plt.legend(ncol=4); plt.grid(True); plt.title("Joint Positions (q)")
        plt.xlabel("t (s)"); plt.ylabel("rad"); plt.tight_layout()

        # Joint torques
        plt.figure(figsize=(12,5))
        for j in range(min(7, model.nv)):
            plt.plot(t_log, tau_log[:, j], label=f"J{j+1}")
        plt.legend(ncol=4); plt.grid(True); plt.title("Joint Torques (tau)")
        plt.xlabel("t (s)"); plt.ylabel("Nm"); plt.tight_layout()

        # EE YZ path: actual vs desired
        plt.figure(figsize=(6,6))
        plt.plot(ee_pos_des_log[:,1], ee_pos_des_log[:,2], '--', label='desired YZ', alpha=0.7)
        plt.plot(ee_pos_log[:,1],     ee_pos_log[:,2],     '-',  label='actual YZ', linewidth=1.5)
        
        # Mark start and end points
        plt.plot(ee_pos_log[0,1], ee_pos_log[0,2], 'go', markersize=10, label='Start', zorder=5)
        plt.plot(ee_pos_log[-1,1], ee_pos_log[-1,2], 'ro', markersize=10, label='End', zorder=5)
        
        # Add text labels with offset
        plt.text(ee_pos_log[0,1], ee_pos_log[0,2], '  Start', fontsize=10, 
                 verticalalignment='bottom', color='green', fontweight='bold')
        plt.text(ee_pos_log[-1,1], ee_pos_log[-1,2], '  End', fontsize=10, 
                 verticalalignment='top', color='red', fontweight='bold')
        
        plt.axis('equal'); plt.grid(True)
        plt.xlabel('Y (m)'); plt.ylabel('Z (m)')
        plt.title('End-Effector path on YZ plane')
        plt.legend(); plt.tight_layout()

        # EE time traces (Y/Z)
        plt.figure(figsize=(12,5))
        plt.plot(t_log, ee_pos_des_log[:,1], '--', label='y*')
        plt.plot(t_log, ee_pos_log[:,1],     '-',  label='y')
        plt.plot(t_log, ee_pos_des_log[:,2], '--', label='z*')
        plt.plot(t_log, ee_pos_log[:,2],     '-',  label='z')
        plt.grid(True); plt.legend()
        plt.title('End-Effector Y/Z vs time'); plt.xlabel('t (s)'); plt.ylabel('m')
        plt.tight_layout()

        # Phase-lock diagnostics
        plt.figure(figsize=(12,8))
        plt.subplot(3,1,1)
        plt.plot(t_log, phase_log)
        plt.grid(True); plt.ylabel('Phase s (rad)')
        plt.title('Phase-Lock Control Diagnostics')
        
        plt.subplot(3,1,2)
        plt.plot(t_log, phase_error_log * 1000)  # Convert to mm
        plt.grid(True); plt.ylabel('Tangential error (mm)')
        plt.axhline(0, color='k', linestyle='--', alpha=0.3)
        
        plt.subplot(3,1,3)
        phase_vel = np.gradient(phase_log, dt)
        plt.plot(t_log, phase_vel, label='actual s_dot')
        plt.axhline(omega, color='r', linestyle='--', label=f'nominal ω={omega:.2f}')
        plt.grid(True); plt.ylabel('Phase velocity (rad/s)')
        plt.xlabel('t (s)'); plt.legend()
        plt.tight_layout()

        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()

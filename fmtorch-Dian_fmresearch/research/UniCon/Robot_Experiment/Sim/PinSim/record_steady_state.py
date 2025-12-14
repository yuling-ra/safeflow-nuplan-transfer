#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记录稳态周期运动的完整状态序列

运行多个周期（默认10个），提取第5个周期后的稳态数据，
识别并保存一个完整周期的机械臂状态（q, dq, phase）。

Usage:
    python record_steady_state.py [--cyc CYCLES] [--output OUTPUT_FILE]
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

# Import helper functions from main script
def so3_log(R):
    tr = np.trace(R); c = np.clip((tr-1.0)*0.5, -1.0, 1.0); th = np.arccos(c)
    if th < 1e-8:
        return 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    w = (1.0/(2.0*np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return th*w

def ema(prev, new, alpha):
    return alpha*prev + (1.0-alpha)*new

def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
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
                  iters=400, tol=1e-4):
    y  = cy + Ay*np.sin(omega*phase0)
    z  = cz + Az*np.sin(2.0*omega*phase0 + phi)
    vy = Ay*omega*np.cos(omega*phase0)
    vz = 2.0*Az*omega*np.cos(2.0*omega*phase0 + phi)
    x_des = np.array([x_plane, y, z], dtype=float)
    v_des = np.array([0.0, vy, vz], dtype=float)

    ez = np.array([1.,0.,0.])
    tang = np.array([0.0, vy, vz])
    if np.linalg.norm(tang) < 1e-8:
        up = np.array([0.,0.,1.]); ex = np.cross(up, ez); ex /= np.linalg.norm(ex); ey = np.cross(ez, ex)
        R_des = np.column_stack([ex, ey, ez])
    else:
        ey = tang/np.linalg.norm(tang); ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
        R_des = np.column_stack([ex, ey, ez])

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

        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        p = data.oMf[ee_fid].translation
        if np.linalg.norm(p - x_des) < tol: break

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
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[frame_id]; p = oMf.translation; R = oMf.rotation

    v_cmd = v_des + kp_pos*(x_des - p)
    w_cmd = kp_ori * so3_log(R_des @ R.T) if kp_ori > 0 else np.zeros(3)

    J6 = pin.computeFrameJacobian(model, data, q, frame_id,
                                  pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    s = np.linalg.svd(J6, compute_uv=False)
    smin = float(s[-1]) if s.size else 0.0

    thr = 0.005
    speed_scale = 1.0 if smin > thr else np.clip(smin/thr, 0.6, 1.0)
    v_cmd *= speed_scale
    w_cmd *= speed_scale

    y = np.hstack([v_cmd, w_cmd])

    lam = max(base_damp, 0.05)
    JJt_reg = J6 @ J6.T + (lam**2)*np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt_reg, y)

    lim = np.full(model.nv, qdot_limit, dtype=float)
    scale = np.max(np.abs(qdot)/lim) if lim.size else 0.0
    if scale > 1.0:
        qdot /= scale

    q_next = pin.integrate(model, q, rate_alpha * qdot * dt)
    return q_next, qdot, smin, np.linalg.norm(x_des - p)

def main():
    parser = argparse.ArgumentParser(description='Record steady-state cycle for Panda FR3 figure-8')
    parser.add_argument('--cyc', type=float, default=10.0,
                        help='Number of cycles to run (default: 10)')
    parser.add_argument('--output', type=str, default='steady_state_cycle.npz',
                        help='Output file name (default: steady_state_cycle.npz)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization for faster recording')
    args = parser.parse_args()
    
    # Load model
    URDF_PATH = os.path.join(os.path.dirname(__file__),
                             "panda_description", "urdf", "panda_stick.urdf")
    assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"
    PACKAGE_DIRS = [os.path.join(os.path.dirname(__file__), "panda_description")]

    model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, PACKAGE_DIRS)
    data  = model.createData()
    print(f"[pin] nq={model.nq}, nv={model.nv}")

    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)

    # Initial state
    initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105], dtype=np.float64)
    q  = initial_qpos.copy()
    dq = np.zeros(model.nv)

    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    p0_ref = data.oMf[ee_fid].translation.copy()

    # Figure-8 parameters
    x_plane = p0_ref[0] + 0.25
    cy, cz = p0_ref[1], p0_ref[2] + 0.12
    Ay, Az = 0.10, 0.05
    omega, phi = 0.6, 0.0
    dt = 0.002

    # Gains (high-bandwidth phase-lock configuration)
    kp_pos, kp_ori = 24.0, 0.5
    damp = 0.08
    qdot_lim = 3.5
    rate_alpha = 0.90
    Kp_j = np.array([120,120,100,100,80,70,60], dtype=float)
    Kv_j = 2.0 * np.sqrt(Kp_j) * 1.0

    # Initial landing
    print("[init] Landing to phase point...")
    phase0_guess = 0.0
    q, dq, phase0 = land_to_phase(
        model, data, ee_fid, q, dq,
        x_plane, cy, cz, Ay, Az, omega, phi,
        phase0_guess, dt,
        kp_pos, kp_ori, damp, qdot_lim, rate_alpha,
        Kp_j, Kv_j,
        iters=500, tol=5e-5
    )
    print(f"[init] Landed on phase0={phase0:.4f} rad")

    # Trajectory function
    def desired_pose_from_phase(s):
        r = 1.0
        Ay_eff, Az_eff = r * Ay, r * Az
        omega_eff = omega
        
        y  = cy + Ay_eff*np.sin(omega_eff*s)
        z  = cz + Az_eff*np.sin(2.0*omega_eff*s + phi)
        vy = Ay_eff*omega_eff*np.cos(omega_eff*s)
        vz = 2.0*Az_eff*omega_eff*np.cos(2.0*omega_eff*s + phi)
        
        x = np.array([x_plane, y, z])
        v = np.array([0.0, vy, vz])
        tang_vec = np.array([0.0, vy, vz])
        
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

    # Simulation parameters
    T_CYCLE = 2.0 * np.pi / omega
    NUM_CYCLES = args.cyc
    T_TOTAL = NUM_CYCLES * T_CYCLE
    
    TAU_LIM = np.array([85,85,85,85,40,40,40], dtype=float)
    USE_TAU_CLIP = False
    TAU_LP_ALPHA = 0.25
    TAU_SLEW = np.full(7, 5e4, dtype=float)
    QDOT_LP_ALPHA = 0.15
    K_PHASE = 8.0
    LP_S = 0.2

    # Logs
    N = int(T_TOTAL/dt)
    t = 0.0
    q_log  = np.zeros((N, model.nq))
    dq_log = np.zeros((N, model.nv))
    phase_log = np.zeros(N)
    t_log  = np.zeros(N)

    tau_prev = np.zeros(model.nv)
    qdot_ref_filt = np.zeros(model.nv)
    s = phase0
    s_dot_prev = omega

    # Visualization (optional)
    full_viz = None
    if not args.no_viz and HAS_MESHCAT and PinMeshcatVis is not None:
        full_viz = PinMeshcatVis(model, collision_model, visual_model)
        try:
            full_viz.initViewer(open=True)
        except Exception:
            full_viz.initViewer(open=False)
        full_viz.loadViewerModel()
        full_viz.display(q)
        print("[viz] Visualization enabled")
    
    last_viz = time.time()

    print(f"[run] Recording {NUM_CYCLES:.1f} cycles ({T_TOTAL:.2f}s)...")
    print(f"[run] Will extract steady state from cycle 5 onwards...")
    
    for i in range(N):
        x_des, v_des, R_des, tang_vec = desired_pose_from_phase(s)

        q_ref, qdot_ref, smin, pos_err = resolved_rate_step6(
            model, data, ee_fid, q, x_des, v_des, R_des,
            kp_pos=kp_pos, kp_ori=kp_ori,
            base_damp=damp, qdot_limit=qdot_lim, rate_alpha=rate_alpha, dt=dt
        )

        qdot_ref_filt = ema(qdot_ref_filt, qdot_ref, QDOT_LP_ALPHA)

        e  = pin.difference(model, q, q_ref)
        ed = qdot_ref_filt - dq
        ddq_cmd = Kp_j * e + Kv_j * ed

        tau_raw = pin.rnea(model, data, q, dq, ddq_cmd)

        if USE_TAU_CLIP:
            tau_raw = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        tau_filt = ema(tau_prev, tau_raw, TAU_LP_ALPHA)
        max_step = TAU_SLEW * dt
        delta = np.clip(tau_filt - tau_prev, -max_step, max_step)
        tau = tau_prev + delta
        tau_prev = tau.copy()

        M  = pin.crba(model, data, q)
        h  = pin.nle (model, data, q, dq)
        ddq= np.linalg.solve(M, tau - h)

        dq = dq + ddq*dt
        q  = pin.integrate(model, q, dq*dt)

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        ee = data.oMf[ee_fid].translation.copy()
        
        e_xyz = x_des - ee
        tang_norm = np.linalg.norm(tang_vec)
        if tang_norm > 1e-9:
            t_hat = tang_vec / tang_norm
            e_tan = float(e_xyz.dot(t_hat))
        else:
            e_tan = 0.0
        
        s_dot_cmd = omega + K_PHASE * e_tan
        s_dot = (1.0 - LP_S) * s_dot_cmd + LP_S * s_dot_prev
        s += s_dot * dt
        s_dot_prev = s_dot

        q_log[i], dq_log[i], phase_log[i], t_log[i] = q, dq, s, t

        if full_viz is not None and (time.time()-last_viz > 0.02):
            full_viz.display(q)
            last_viz = time.time()

        if i % 500 == 0:
            cycle_num = t / T_CYCLE
            print(f"  Cycle {cycle_num:.2f} | t={t:.2f}s | phase={s:.3f}")

        t += dt
    
    print(f"[run] Simulation complete. Total cycles: {NUM_CYCLES:.1f}")

    # Extract steady state (from cycle 5 onwards)
    t_start_steady = 5.0 * T_CYCLE
    idx_start = int(t_start_steady / dt)
    
    print(f"\n[extract] Extracting steady state from t={t_start_steady:.2f}s (index {idx_start})...")
    
    # Find one complete cycle in steady state
    # Use phase to identify cycle boundaries
    phase_steady = phase_log[idx_start:]
    phase_steady_mod = np.mod(phase_steady, 2*np.pi/omega)  # Modulo one cycle
    
    # Find where phase crosses zero (start of cycle)
    crossings = []
    for i in range(1, len(phase_steady_mod)):
        if phase_steady_mod[i-1] > phase_steady_mod[i]:  # Wrapped around
            crossings.append(i)
    
    if len(crossings) < 2:
        print("[error] Could not find complete cycle in steady state!")
        print(f"  Found {len(crossings)} phase crossings, need at least 2")
        return
    
    # Take first complete cycle
    idx_cycle_start = crossings[0]
    idx_cycle_end = crossings[1]
    
    # Extract one complete cycle
    q_cycle = q_log[idx_start + idx_cycle_start : idx_start + idx_cycle_end]
    dq_cycle = dq_log[idx_start + idx_cycle_start : idx_start + idx_cycle_end]
    phase_cycle = phase_log[idx_start + idx_cycle_start : idx_start + idx_cycle_end]
    
    n_frames = len(q_cycle)
    cycle_duration = n_frames * dt
    
    print(f"[extract] Extracted steady-state cycle:")
    print(f"  - Start index: {idx_start + idx_cycle_start}")
    print(f"  - End index: {idx_start + idx_cycle_end}")
    print(f"  - Number of frames: {n_frames}")
    print(f"  - Cycle duration: {cycle_duration:.4f}s (expected: {T_CYCLE:.4f}s)")
    print(f"  - Phase range: [{phase_cycle[0]:.4f}, {phase_cycle[-1]:.4f}]")
    
    # Compute end-effector positions for verification
    ee_positions = np.zeros((n_frames, 3))
    for i in range(n_frames):
        pin.forwardKinematics(model, data, q_cycle[i])
        pin.updateFramePlacements(model, data)
        ee_positions[i] = data.oMf[ee_fid].translation.copy()
    
    # Check cycle closure
    q_error = np.linalg.norm(pin.difference(model, q_cycle[0], q_cycle[-1]))
    dq_error = np.linalg.norm(dq_cycle[0] - dq_cycle[-1])
    ee_error = np.linalg.norm(ee_positions[0] - ee_positions[-1])
    
    print(f"\n[verify] Cycle closure:")
    print(f"  - Joint position error: {q_error:.6f} rad")
    print(f"  - Joint velocity error: {dq_error:.6f} rad/s")
    print(f"  - End-effector position error: {ee_error*1000:.3f} mm")
    
    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    np.savez(output_path,
             q_cycle=q_cycle,
             dq_cycle=dq_cycle,
             phase_cycle=phase_cycle,
             dt=dt,
             n_frames=n_frames,
             omega=omega,
             # Trajectory parameters
             x_plane=x_plane, cy=cy, cz=cz, Ay=Ay, Az=Az, phi=phi,
             # Metadata
             cycle_duration=cycle_duration,
             T_CYCLE=T_CYCLE,
             q_error=q_error,
             dq_error=dq_error,
             ee_error=ee_error,
             ee_positions=ee_positions)
    
    print(f"\n[save] Steady-state cycle saved to: {output_path}")
    print(f"[save] File size: {os.path.getsize(output_path)/1024:.2f} KB")
    print("\n[done] Recording complete!")
    print(f"\nTo use this in simulation, run:")
    print(f"  python pin_fr3_draw_eight.py --init-steady-state {args.output}")

if __name__ == "__main__":
    main()


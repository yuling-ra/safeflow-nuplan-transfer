#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估机械臂进入稳态周期运动的收敛速度

运行长时间仿真，分析每个周期的状态误差，自动检测何时进入稳态。
输出：收敛曲线图 + 稳态阈值分析

Usage:
    python evaluate_convergence.py [--cyc CYCLES] [--threshold THRESH]
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

# Import helper functions
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
    x_des = np.array([x_plane, y, z], dtype=float)
    v_des = np.zeros(3)
    ez = np.array([1.,0.,0.])
    up = np.array([0.,0.,1.])
    ex = np.cross(up, ez); ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
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
    parser = argparse.ArgumentParser(description='Evaluate convergence to steady-state cycle')
    parser.add_argument('--cyc', type=float, default=20.0,
                        help='Number of cycles to evaluate (default: 20)')
    parser.add_argument('--threshold', type=float, default=1e-4,
                        help='Convergence threshold in rad (default: 1e-4)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
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

    # Gains
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
    ee_pos_log = np.zeros((N, 3))
    tau_log = np.zeros((N, model.nv))  # 记录力矩

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

    print(f"\n[run] Evaluating convergence over {NUM_CYCLES:.1f} cycles ({T_TOTAL:.2f}s)...")
    print(f"[run] Convergence threshold: {args.threshold:.6f} rad\n")
    
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
        ee_pos_log[i] = ee
        tau_log[i] = tau

        if full_viz is not None and (time.time()-last_viz > 0.05):
            full_viz.display(q)
            last_viz = time.time()

        if i % 1000 == 0:
            cycle_num = t / T_CYCLE
            print(f"  Cycle {cycle_num:.2f} | t={t:.2f}s")

        t += dt
    
    print(f"[run] Simulation complete.\n")

    # ==================== 周期分析 ====================
    print("[analysis] Analyzing cycle-to-cycle convergence...\n")
    
    # 识别所有周期边界
    phase_mod = np.mod(phase_log, 2*np.pi/omega)
    crossings = [0]  # 起点
    for i in range(1, len(phase_mod)):
        if phase_mod[i-1] > phase_mod[i]:  # 相位回绕
            crossings.append(i)
    
    n_complete_cycles = len(crossings) - 1
    print(f"[analysis] Found {n_complete_cycles} complete cycles")
    
    if n_complete_cycles < 2:
        print("[error] Need at least 2 cycles for analysis")
        return
    
    # 对每个周期计算统计量
    cycle_metrics = []
    for cyc_idx in range(n_complete_cycles):
        start_idx = crossings[cyc_idx]
        end_idx = crossings[cyc_idx + 1] if cyc_idx + 1 < len(crossings) else len(q_log)
        
        # 提取该周期的数据
        q_cycle = q_log[start_idx:end_idx]
        dq_cycle = dq_log[start_idx:end_idx]
        ee_cycle = ee_pos_log[start_idx:end_idx]
        
        # 计算闭合误差
        q_closure = np.linalg.norm(pin.difference(model, q_cycle[0], q_cycle[-1]))
        dq_closure = np.linalg.norm(dq_cycle[0] - dq_cycle[-1])
        ee_closure = np.linalg.norm(ee_cycle[0] - ee_cycle[-1])
        
        # 计算均方根跟踪误差（相对于期望轨迹）
        ee_errors = []
        for j in range(len(q_cycle)):
            idx = start_idx + j
            x_des, _, _, _ = desired_pose_from_phase(phase_log[idx])
            ee_err = np.linalg.norm(ee_cycle[j] - x_des)
            ee_errors.append(ee_err)
        ee_rms = np.sqrt(np.mean(np.array(ee_errors)**2))
        ee_max = np.max(ee_errors)
        
        cycle_metrics.append({
            'cycle': cyc_idx,
            'q_closure': q_closure,
            'dq_closure': dq_closure,
            'ee_closure': ee_closure,
            'ee_rms': ee_rms,
            'ee_max': ee_max,
            'n_frames': len(q_cycle)
        })
    
    # 打印每个周期的指标
    print("\n" + "="*80)
    print(f"{'Cycle':<6} {'q_closure':<12} {'dq_closure':<12} {'ee_closure':<12} {'ee_rms':<12} {'ee_max':<12}")
    print(f"{'':6} {'(rad)':<12} {'(rad/s)':<12} {'(mm)':<12} {'(mm)':<12} {'(mm)':<12}")
    print("="*80)
    
    for m in cycle_metrics:
        print(f"{m['cycle']:<6} {m['q_closure']:<12.6f} {m['dq_closure']:<12.6f} "
              f"{m['ee_closure']*1000:<12.3f} {m['ee_rms']*1000:<12.3f} {m['ee_max']*1000:<12.3f}")
    
    print("="*80)
    
    # 判断收敛
    threshold = args.threshold
    converged_cycle = None
    for i in range(1, len(cycle_metrics)):
        if (cycle_metrics[i]['q_closure'] < threshold and 
            cycle_metrics[i]['dq_closure'] < threshold * 10 and
            cycle_metrics[i]['ee_closure'] < threshold * 100):  # 0.01mm
            converged_cycle = i
            break
    
    if converged_cycle is not None:
        print(f"\n✓ 系统从第 {converged_cycle} 个周期开始进入稳态！")
        print(f"  - 关节位置闭合误差: {cycle_metrics[converged_cycle]['q_closure']:.6f} rad < {threshold:.6f}")
        print(f"  - 关节速度闭合误差: {cycle_metrics[converged_cycle]['dq_closure']:.6f} rad/s < {threshold*10:.6f}")
        print(f"  - 末端位置闭合误差: {cycle_metrics[converged_cycle]['ee_closure']*1000:.3f} mm < {threshold*100:.3f}")
    else:
        print(f"\n✗ 在 {n_complete_cycles} 个周期内未完全收敛到阈值 {threshold:.6f} rad")
        print(f"  最后一个周期的闭合误差: q={cycle_metrics[-1]['q_closure']:.6f} rad")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        
        cycles = [m['cycle'] for m in cycle_metrics]
        q_closures = [m['q_closure'] for m in cycle_metrics]
        dq_closures = [m['dq_closure'] for m in cycle_metrics]
        ee_closures = [m['ee_closure']*1000 for m in cycle_metrics]  # mm
        ee_rms_values = [m['ee_rms']*1000 for m in cycle_metrics]  # mm
        
        # ===== 图1: 收敛分析 =====
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # q closure
        axes[0,0].plot(cycles, q_closures, 'o-', linewidth=2, markersize=6)
        axes[0,0].axhline(threshold, color='r', linestyle='--', label=f'Threshold {threshold:.1e}')
        if converged_cycle is not None:
            axes[0,0].axvline(converged_cycle, color='g', linestyle='--', alpha=0.5, label=f'Converged @ cycle {converged_cycle}')
        axes[0,0].set_xlabel('Cycle'); axes[0,0].set_ylabel('q closure error (rad)')
        axes[0,0].set_title('Joint Position Closure Error')
        axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()
        axes[0,0].set_yscale('log')
        
        # dq closure
        axes[0,1].plot(cycles, dq_closures, 'o-', linewidth=2, markersize=6, color='orange')
        axes[0,1].axhline(threshold*10, color='r', linestyle='--', label=f'Threshold {threshold*10:.1e}')
        if converged_cycle is not None:
            axes[0,1].axvline(converged_cycle, color='g', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Cycle'); axes[0,1].set_ylabel('dq closure error (rad/s)')
        axes[0,1].set_title('Joint Velocity Closure Error')
        axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()
        axes[0,1].set_yscale('log')
        
        # ee closure
        axes[1,0].plot(cycles, ee_closures, 'o-', linewidth=2, markersize=6, color='green')
        axes[1,0].axhline(threshold*100, color='r', linestyle='--', label=f'Threshold {threshold*100:.2f} mm')
        if converged_cycle is not None:
            axes[1,0].axvline(converged_cycle, color='g', linestyle='--', alpha=0.5)
        axes[1,0].set_xlabel('Cycle'); axes[1,0].set_ylabel('EE closure error (mm)')
        axes[1,0].set_title('End-Effector Position Closure Error')
        axes[1,0].grid(True, alpha=0.3); axes[1,0].legend()
        axes[1,0].set_yscale('log')
        
        # ee rms tracking error
        axes[1,1].plot(cycles, ee_rms_values, 'o-', linewidth=2, markersize=6, color='purple')
        if converged_cycle is not None:
            axes[1,1].axvline(converged_cycle, color='g', linestyle='--', alpha=0.5, label=f'Converged @ cycle {converged_cycle}')
        axes[1,1].set_xlabel('Cycle'); axes[1,1].set_ylabel('EE RMS tracking error (mm)')
        axes[1,1].set_title('End-Effector RMS Tracking Error')
        axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(os.path.dirname(__file__), 'convergence_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[plot] Convergence analysis saved to: {output_path}")
        
        # ===== 图2: YZ平面8字轨迹跟随 =====
        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
        
        # 生成期望轨迹（完整的8字）
        s_ref = np.linspace(0, 2*np.pi/omega, 1000)
        y_ref = cy + Ay * np.sin(omega * s_ref)
        z_ref = cz + Az * np.sin(2 * omega * s_ref + phi)
        
        # 左图: 全部轨迹
        axes2[0].plot(y_ref, z_ref, 'b--', linewidth=2, label='Desired trajectory', alpha=0.7)
        axes2[0].plot(ee_pos_log[:, 1], ee_pos_log[:, 2], 'r-', linewidth=1, label='Actual trajectory', alpha=0.8)
        
        # 标记起点和终点
        axes2[0].plot(ee_pos_log[0, 1], ee_pos_log[0, 2], 'go', markersize=12, label='Start', zorder=5)
        axes2[0].plot(ee_pos_log[-1, 1], ee_pos_log[-1, 2], 'rs', markersize=12, label='End', zorder=5)
        
        # 标记收敛点（如果有）
        if converged_cycle is not None and converged_cycle < len(crossings):
            conv_idx = crossings[converged_cycle]
            axes2[0].plot(ee_pos_log[conv_idx, 1], ee_pos_log[conv_idx, 2], 
                         'md', markersize=12, label=f'Converged (cycle {converged_cycle})', zorder=5)
        
        axes2[0].axis('equal')
        axes2[0].grid(True, alpha=0.3)
        axes2[0].set_xlabel('Y (m)', fontsize=12)
        axes2[0].set_ylabel('Z (m)', fontsize=12)
        axes2[0].set_title(f'End-Effector Trajectory on YZ Plane ({NUM_CYCLES:.1f} cycles)', fontsize=14)
        axes2[0].legend(loc='best', fontsize=10)
        
        # 右图: 稳态周期（如果收敛）
        if converged_cycle is not None and converged_cycle < len(crossings) - 1:
            start_idx = crossings[converged_cycle]
            end_idx = crossings[converged_cycle + 1]
            
            axes2[1].plot(y_ref, z_ref, 'b--', linewidth=2, label='Desired trajectory', alpha=0.7)
            axes2[1].plot(ee_pos_log[start_idx:end_idx, 1], 
                         ee_pos_log[start_idx:end_idx, 2], 
                         'r-', linewidth=2, label=f'Steady-state (cycle {converged_cycle})', alpha=0.9)
            
            # 标记起点和终点
            axes2[1].plot(ee_pos_log[start_idx, 1], ee_pos_log[start_idx, 2], 
                         'go', markersize=12, label='Cycle start', zorder=5)
            axes2[1].plot(ee_pos_log[end_idx-1, 1], ee_pos_log[end_idx-1, 2], 
                         'rs', markersize=10, label='Cycle end', zorder=5)
            
            axes2[1].axis('equal')
            axes2[1].grid(True, alpha=0.3)
            axes2[1].set_xlabel('Y (m)', fontsize=12)
            axes2[1].set_ylabel('Z (m)', fontsize=12)
            axes2[1].set_title(f'Steady-State Cycle {converged_cycle} (Zoomed)', fontsize=14)
            axes2[1].legend(loc='best', fontsize=10)
            
            # 显示闭合误差
            closure_text = f"Closure error: {cycle_metrics[converged_cycle]['ee_closure']*1000:.3f} mm"
            axes2[1].text(0.05, 0.95, closure_text, transform=axes2[1].transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            # 如果未收敛，显示最后一个周期
            last_cycle_idx = len(crossings) - 2
            if last_cycle_idx >= 0:
                start_idx = crossings[last_cycle_idx]
                end_idx = crossings[last_cycle_idx + 1] if last_cycle_idx + 1 < len(crossings) else len(ee_pos_log)
                
                axes2[1].plot(y_ref, z_ref, 'b--', linewidth=2, label='Desired trajectory', alpha=0.7)
                axes2[1].plot(ee_pos_log[start_idx:end_idx, 1], 
                             ee_pos_log[start_idx:end_idx, 2], 
                             'orange', linewidth=2, label=f'Last cycle {last_cycle_idx}', alpha=0.9)
                
                axes2[1].plot(ee_pos_log[start_idx, 1], ee_pos_log[start_idx, 2], 
                             'go', markersize=12, label='Cycle start', zorder=5)
                axes2[1].plot(ee_pos_log[end_idx-1, 1], ee_pos_log[end_idx-1, 2], 
                             'rs', markersize=10, label='Cycle end', zorder=5)
                
                axes2[1].axis('equal')
                axes2[1].grid(True, alpha=0.3)
                axes2[1].set_xlabel('Y (m)', fontsize=12)
                axes2[1].set_ylabel('Z (m)', fontsize=12)
                axes2[1].set_title(f'Last Cycle {last_cycle_idx} (Not yet converged)', fontsize=14)
                axes2[1].legend(loc='best', fontsize=10)
                
                closure_text = f"Closure error: {cycle_metrics[last_cycle_idx]['ee_closure']*1000:.3f} mm"
                axes2[1].text(0.05, 0.95, closure_text, transform=axes2[1].transAxes,
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存YZ轨迹图
        output_path_yz = os.path.join(os.path.dirname(__file__), 'trajectory_yz_tracking.png')
        plt.savefig(output_path_yz, dpi=150, bbox_inches='tight')
        print(f"[plot] YZ trajectory tracking saved to: {output_path_yz}")
        
        # ===== 图3: 力矩周期变化 =====
        # 选择要绘制的周期（如果>5个，随机选5个连续的）
        if n_complete_cycles <= 5:
            selected_cycles = list(range(n_complete_cycles))
        else:
            # 随机选择起始点，确保能取到5个连续的周期
            max_start = n_complete_cycles - 5
            start_idx = np.random.randint(0, max_start + 1)
            selected_cycles = list(range(start_idx, start_idx + 5))
        
        n_selected = len(selected_cycles)
        fig3, axes3 = plt.subplots(n_selected, 1, figsize=(16, 3*n_selected))
        if n_selected == 1:
            axes3 = [axes3]  # 确保是列表
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        
        for plot_idx, cyc_idx in enumerate(selected_cycles):
            start_idx_cycle = crossings[cyc_idx]
            end_idx_cycle = crossings[cyc_idx + 1] if cyc_idx + 1 < len(crossings) else len(tau_log)
            
            # 提取该周期的力矩数据
            tau_cycle = tau_log[start_idx_cycle:end_idx_cycle]
            t_cycle = np.arange(len(tau_cycle)) * dt  # 周期内的相对时间
            
            # 绘制7个关节的力矩
            for j in range(model.nv):
                axes3[plot_idx].plot(t_cycle, tau_cycle[:, j], 
                                    color=colors[j], linewidth=1.5, 
                                    label=joint_names[j], alpha=0.85)
            
            # 标记力矩限制（浅色虚线）
            for j in range(model.nv):
                axes3[plot_idx].axhline(TAU_LIM[j], color=colors[j], linestyle=':', 
                                       linewidth=1, alpha=0.3)
                axes3[plot_idx].axhline(-TAU_LIM[j], color=colors[j], linestyle=':', 
                                       linewidth=1, alpha=0.3)
            
            # 标题和标签
            if converged_cycle is not None and cyc_idx >= converged_cycle:
                title_suffix = " ✓ Steady-state"
                title_color = 'green'
                title_weight = 'bold'
            elif converged_cycle is not None and cyc_idx == converged_cycle - 1:
                title_suffix = " ⚠ Pre-convergence"
                title_color = 'orange'
                title_weight = 'normal'
            else:
                title_suffix = ""
                title_color = 'black'
                title_weight = 'normal'
            
            axes3[plot_idx].set_title(f'Cycle {cyc_idx}{title_suffix}', 
                                     fontsize=13, color=title_color, weight=title_weight)
            axes3[plot_idx].set_ylabel('Torque (Nm)', fontsize=11)
            axes3[plot_idx].grid(True, alpha=0.3, linestyle='--')
            axes3[plot_idx].legend(loc='upper right', ncol=7, fontsize=9, framealpha=0.9)
            
            # 显示力矩统计
            tau_max = np.max(np.abs(tau_cycle), axis=0)
            tau_rms = np.sqrt(np.mean(tau_cycle**2, axis=0))
            tau_peak = np.max(tau_max)
            tau_rms_avg = np.mean(tau_rms)
            
            # 检查是否有力矩饱和
            saturated = np.any(np.abs(tau_cycle) >= TAU_LIM * 0.99)
            saturation_warning = " ⚠ SATURATED!" if saturated else ""
            
            stats_text = (f"Peak: {tau_peak:.1f} Nm | RMS (avg): {tau_rms_avg:.1f} Nm | "
                         f"Duration: {t_cycle[-1]:.2f}s{saturation_warning}")
            
            box_color = 'lightyellow' if saturated else 'white'
            axes3[plot_idx].text(0.02, 0.98, stats_text, 
                                transform=axes3[plot_idx].transAxes,
                                fontsize=9, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
            
            # 只在最后一个子图显示x轴标签
            if plot_idx == n_selected - 1:
                axes3[plot_idx].set_xlabel('Time in cycle (s)', fontsize=11)
            else:
                axes3[plot_idx].set_xticklabels([])
        
        plt.tight_layout()
        
        # 保存力矩图
        output_path_tau = os.path.join(os.path.dirname(__file__), 'torque_cycles.png')
        plt.savefig(output_path_tau, dpi=150, bbox_inches='tight')
        print(f"[plot] Torque cycles saved to: {output_path_tau}")
        print(f"       Selected cycles: {selected_cycles}")
        
        plt.show()
    except Exception as e:
        print(f"[warn] Could not generate plot: {e}")
    
    # 保存数据
    output_data = os.path.join(os.path.dirname(__file__), 'convergence_data.npz')
    np.savez(output_data,
             cycle_metrics=cycle_metrics,
             converged_cycle=converged_cycle if converged_cycle is not None else -1,
             threshold=threshold,
             n_complete_cycles=n_complete_cycles)
    print(f"[save] Convergence data saved to: {output_data}")
    
    print("\n[done] Convergence evaluation complete!")

if __name__ == "__main__":
    main()


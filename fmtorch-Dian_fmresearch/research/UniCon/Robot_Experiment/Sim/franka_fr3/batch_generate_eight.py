#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Generation Script for FR3 Figure-8 Drawing
批量生成机器人绘制图8轨迹的数据集

特性：
1. 无渲染模式快速生成大量轨迹
2. 每N条轨迹迭代保存到npz文件，防止中断丢失
3. 每条轨迹使用不同的随机起点
4. 包含数据质量检查和评估
"""

import argparse
import time
import numpy as np
import mujoco
from pathlib import Path
from datetime import datetime

# ---------------------- SO(3) Utils ----------------------
def _so3_log(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]], dtype=float)
    w = (1.0 / (2.0 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2],
                                                  R[0, 2] - R[2, 0],
                                                  R[1, 0] - R[0, 1]], dtype=float)
    return theta * w

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

# ---------------------- IK-style 6D step ----------------------
def resolved_rate_step_6d(model, data, site_id,
                          x_des: np.ndarray, v_des: np.ndarray,
                          R_des: np.ndarray, q_ref: np.ndarray,
                          kp_cart: float = 8.0, kp_ori: float = 6.0,
                          damping: float = 0.02, dt: float = 0.002) -> np.ndarray:
    x = data.site_xpos[site_id].copy()
    R = data.site_xmat[site_id].reshape(3, 3).copy()
    x_err = x_des - x
    v_cmd = v_des + kp_cart * x_err
    R_err = R_des @ R.T
    w_cmd = kp_ori * _so3_log(R_err)
    J_pos = np.zeros((3, model.nv))
    J_ori = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, J_pos, J_ori, site_id)
    J6 = np.vstack([J_pos, J_ori])
    y = np.hstack([v_cmd, w_cmd])
    JJt = J6 @ J6.T
    lam2I = (damping ** 2) * np.eye(6)
    qdot = J6.T @ np.linalg.solve(JJt + lam2I, y)
    q_ref_new = q_ref + qdot * dt
    # clip joint limits
    idx_q = 0
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            r = model.jnt_range[j]
            if r[0] < r[1]:
                q_ref_new[idx_q] = np.clip(q_ref_new[idx_q], r[0], r[1])
            idx_q += 1
    return q_ref_new

def desired_orientation(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """水平方向的末端执行器姿态"""
    z_axis_world = np.array([1.0, 0.0, 0.0])
    up_hint = np.array([0.0, 0.0, 1.0])
    x_axis_world = np.cross(up_hint, z_axis_world)
    if np.linalg.norm(x_axis_world) < 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0])
        x_axis_world = np.cross(up_hint, z_axis_world)
    x_axis_world = _normalize(x_axis_world)
    z_axis_world = _normalize(z_axis_world)
    y_axis_world = np.cross(z_axis_world, x_axis_world)
    return np.column_stack([x_axis_world, y_axis_world, z_axis_world])

# ---------------------- Single Trajectory Generation ----------------------
def generate_single_trajectory(model, data, traj_params, ctrl_params, seed):
    """生成单条轨迹"""
    np.random.seed(seed)
    
    # Reset to initial state
    initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105])
    data.qpos[:7] = initial_qpos
    data.qvel[:] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)
    
    # Get site IDs
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_tip")
    
    # Trajectory parameters
    x0_home = data.site_xpos[site_id].copy()
    cy, cz = x0_home[1], x0_home[2] + 0.15
    x_plane = traj_params['x_plane']
    Ay = traj_params['Ay']
    Az = traj_params['Az']
    omega = traj_params['omega']
    phi = traj_params['phi']
    
    # Generate random initial pose outside figure-8
    rand_margin = 0.05
    while True:
        y_offset = np.random.uniform(-Ay - rand_margin, Ay + rand_margin)
        z_offset = np.random.uniform(-Az - rand_margin, Az + rand_margin)
        if abs(y_offset) > Ay or abs(z_offset) > Az:
            break
    x_rand = np.array([x_plane, cy + y_offset, cz + z_offset], dtype=float)
    
    # Solve IK to reach random start
    q_ref = data.qpos[:7].copy()
    R_h = desired_orientation(None, None)
    ctrl_dt = ctrl_params['ctrl_dt']
    
    for _ in range(200):
        q_ref = resolved_rate_step_6d(model, data, site_id,
                                      x_des=x_rand, v_des=np.zeros(3),
                                      R_des=R_h, q_ref=q_ref,
                                      kp_cart=10.0, kp_ori=8.0, damping=0.01, dt=ctrl_dt)
        data.qpos[:7] = q_ref
        mujoco.mj_kinematics(model, data)
        if np.linalg.norm(data.site_xpos[site_id] - x_rand) < 1e-3:
            break
    mujoco.mj_forward(model, data)
    
    # Find closest phase on figure-8
    num_samples = 200
    thetas = np.linspace(0, 2*np.pi/omega, num_samples, endpoint=False)
    traj_y = cy + Ay * np.sin(omega * thetas)
    traj_z = cz + Az * np.sin(2.0 * omega * thetas + phi)
    rand2d = x_rand[1:]
    d2 = (traj_y - rand2d[0])**2 + (traj_z - rand2d[1])**2
    t_traj_start = thetas[np.argmin(d2)]
    
    # PD gains
    Kp = np.array(ctrl_params['Kp'], dtype=float)
    Kv = np.array(ctrl_params['Kv'], dtype=float)
    ctrl_range = model.actuator_ctrlrange.copy()
    
    # Trajectory recording
    INITIAL_HOLD = ctrl_params['initial_hold']
    cycle_T = 2*np.pi/omega
    max_time = INITIAL_HOLD + cycle_T
    
    qpos_traj = []
    qvel_traj = []
    ctrl_traj = []
    ee_pos_traj = []
    ee_vel_traj = []
    tip_pos_traj = []
    times = []
    
    # Simulation loop
    while data.time < max_time:
        t = data.time
        
        # Desired EE target
        if t < INITIAL_HOLD:
            x_des = x_rand
            v_des = np.zeros(3)
            R_des = R_h
        else:
            tau = (t - INITIAL_HOLD) + t_traj_start
            y_des = cy + Ay * np.sin(omega * tau)
            z_des = cz + Az * np.sin(2.0 * omega * tau + phi)
            x_des = np.array([x_plane, y_des, z_des])
            vy = Ay * omega * np.cos(omega * tau)
            vz = 2.0 * Az * omega * np.cos(2.0 * omega * tau + phi)
            v_des = np.array([0.0, vy, vz])
            R_des = desired_orientation(x_des, v_des)
        
        # Update q_ref
        q_ref = resolved_rate_step_6d(model, data, site_id,
                                      x_des, v_des, R_des, q_ref,
                                      kp_cart=ctrl_params['kp_cart'], 
                                      kp_ori=ctrl_params['kp_ori'],
                                      damping=ctrl_params['damping'], 
                                      dt=ctrl_dt)
        
        # PD control
        torque_cmd = Kp * (q_ref - data.qpos[:7]) - Kv * data.qvel[:7]
        torque_cmd = np.clip(torque_cmd, ctrl_range[:7, 0], ctrl_range[:7, 1])
        data.ctrl[:] = 0.0
        data.ctrl[:7] = torque_cmd
        
        # Record before step
        qpos_traj.append(data.qpos.copy())
        qvel_traj.append(data.qvel.copy())
        ctrl_traj.append(data.ctrl.copy())
        ee_pos_traj.append(data.site_xpos[site_id].copy())
        
        # Compute EE velocity (numerical)
        J_pos = np.zeros((3, model.nv))
        J_ori = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, J_pos, J_ori, site_id)
        ee_vel = J_pos @ data.qvel
        ee_vel_traj.append(ee_vel.copy())
        
        tip_pos_traj.append(data.site_xpos[tip_id].copy())
        times.append(data.time)
        
        # Step simulation
        mujoco.mj_step(model, data)
    
    # Convert to arrays
    trajectory_data = {
        'qpos': np.array(qpos_traj),
        'qvel': np.array(qvel_traj),
        'ctrl': np.array(ctrl_traj),
        'ee_pos': np.array(ee_pos_traj),
        'ee_vel': np.array(ee_vel_traj),
        'tip_pos': np.array(tip_pos_traj),
        'times': np.array(times),
        'start_pos': x_rand,
        'start_phase': t_traj_start,
        'seed': seed,
    }
    
    return trajectory_data

# ---------------------- Batch Generation ----------------------
def batch_generate(args):
    """批量生成轨迹数据"""
    print("=" * 80)
    print(f"Batch Generation for FR3 Figure-8 Drawing")
    print(f"Target: {args.num_trajectories} trajectories")
    print(f"Save interval: every {args.save_interval} trajectories")
    print("=" * 80)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    
    # Trajectory parameters
    radius = args.radius
    traj_params = {
        'x_plane': args.x_plane,
        'Ay': args.ay if args.ay is not None else radius,
        'Az': args.az if args.az is not None else radius * 0.5,
        'omega': args.omega,
        'phi': args.phi,
    }
    
    # Control parameters
    ctrl_dt = args.dt if args.dt > 0 else model.opt.timestep
    ctrl_params = {
        'ctrl_dt': ctrl_dt,
        'kp_cart': args.kp_cart,
        'kp_ori': args.kp_ori,
        'damping': args.damping,
        'Kp': args.kp,
        'Kv': args.kv,
        'initial_hold': args.initial_hold,
    }
    
    # Storage
    all_trajectories = []
    
    # Output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"fr3_figure8_batch_{timestamp}"
    
    # Generation loop
    start_time = time.time()
    for i in range(args.num_trajectories):
        traj_start = time.time()
        
        # Use different seed for each trajectory
        seed = args.base_seed + i
        
        try:
            traj_data = generate_single_trajectory(model, data, traj_params, ctrl_params, seed)
            all_trajectories.append(traj_data)
            
            traj_time = time.time() - traj_start
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (args.num_trajectories - i - 1)
            
            print(f"[{i+1}/{args.num_trajectories}] "
                  f"Traj time: {traj_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min | "
                  f"Steps: {len(traj_data['times'])}")
            
            # Incremental save
            if (i + 1) % args.save_interval == 0 or (i + 1) == args.num_trajectories:
                save_path = output_dir / f"{base_filename}_checkpoint_{i+1}.npz"
                save_batch_data(all_trajectories, save_path)
                print(f"  -> Saved checkpoint to {save_path}")
                
        except Exception as e:
            print(f"[ERROR] Failed to generate trajectory {i}: {e}")
            continue
    
    # Final save
    final_path = output_dir / f"{base_filename}_final.npz"
    save_batch_data(all_trajectories, final_path)
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Generation Complete!")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Average time per trajectory: {total_time/len(all_trajectories):.2f}s")
    print(f"Final save: {final_path}")
    print("=" * 80)
    
    # Evaluation
    if not args.no_eval:
        print("\nRunning evaluation...")
        evaluate_batch(all_trajectories, traj_params)
    
    return all_trajectories, final_path

def save_batch_data(trajectories, save_path):
    """保存批量轨迹数据到npz文件"""
    # Stack all trajectory data
    batch_data = {
        'qpos': np.array([t['qpos'] for t in trajectories], dtype=object),
        'qvel': np.array([t['qvel'] for t in trajectories], dtype=object),
        'ctrl': np.array([t['ctrl'] for t in trajectories], dtype=object),
        'ee_pos': np.array([t['ee_pos'] for t in trajectories], dtype=object),
        'ee_vel': np.array([t['ee_vel'] for t in trajectories], dtype=object),
        'tip_pos': np.array([t['tip_pos'] for t in trajectories], dtype=object),
        'times': np.array([t['times'] for t in trajectories], dtype=object),
        'start_positions': np.array([t['start_pos'] for t in trajectories]),
        'start_phases': np.array([t['start_phase'] for t in trajectories]),
        'seeds': np.array([t['seed'] for t in trajectories]),
        'num_trajectories': len(trajectories),
    }
    
    np.savez(save_path, **batch_data)

# ---------------------- Evaluation ----------------------
def evaluate_batch(trajectories, traj_params):
    """评估批量生成的轨迹质量（不绘图）"""
    print("\n" + "=" * 80)
    print("BATCH EVALUATION REPORT")
    print("=" * 80)
    
    num_trajs = len(trajectories)
    
    # 1. Basic statistics
    traj_lengths = [len(t['times']) for t in trajectories]
    traj_durations = [t['times'][-1] for t in trajectories]
    
    print(f"\n1. Basic Statistics:")
    print(f"   Total trajectories: {num_trajs}")
    print(f"   Avg trajectory length: {np.mean(traj_lengths):.1f} ± {np.std(traj_lengths):.1f} steps")
    print(f"   Avg trajectory duration: {np.mean(traj_durations):.2f} ± {np.std(traj_durations):.2f} s")
    print(f"   Min/Max duration: {np.min(traj_durations):.2f}s / {np.max(traj_durations):.2f}s")
    
    # 2. Starting position diversity
    start_positions = np.array([t['start_pos'] for t in trajectories])
    start_phases = np.array([t['start_phase'] for t in trajectories])
    
    print(f"\n2. Starting Position Diversity:")
    print(f"   Y range: [{start_positions[:, 1].min():.3f}, {start_positions[:, 1].max():.3f}]")
    print(f"   Z range: [{start_positions[:, 2].min():.3f}, {start_positions[:, 2].max():.3f}]")
    print(f"   Start phase range: [{start_phases.min():.3f}, {start_phases.max():.3f}] rad")
    
    # 3. Tracking performance
    cy = 0.0  # approximate, would need to compute from trajectories
    cz = 0.0  # approximate
    tracking_errors = []
    
    for t in trajectories:
        tip_pos = t['tip_pos']
        times = t['times']
        omega = traj_params['omega']
        Ay = traj_params['Ay']
        Az = traj_params['Az']
        phi = traj_params['phi']
        x_plane = traj_params['x_plane']
        
        # Compute desired trajectory (approximate center from first EE position)
        if len(tip_pos) > 10:
            # Use average position to estimate center
            cy_est = np.mean(tip_pos[10:, 1])
            cz_est = np.mean(tip_pos[10:, 2])
            
            for i, (tip, time) in enumerate(zip(tip_pos, times)):
                tau = time  # simplified, doesn't account for initial hold
                y_des = cy_est + Ay * np.sin(omega * tau)
                z_des = cz_est + Az * np.sin(2.0 * omega * tau + phi)
                x_des = np.array([x_plane, y_des, z_des])
                error = np.linalg.norm(tip - x_des)
                tracking_errors.append(error)
    
    tracking_errors = np.array(tracking_errors)
    
    print(f"\n3. Tracking Performance:")
    print(f"   Mean tracking error: {np.mean(tracking_errors):.4f} m")
    print(f"   Median tracking error: {np.median(tracking_errors):.4f} m")
    print(f"   95th percentile error: {np.percentile(tracking_errors, 95):.4f} m")
    print(f"   Max tracking error: {np.max(tracking_errors):.4f} m")
    
    # 4. Control effort statistics
    all_torques = []
    for t in trajectories:
        all_torques.append(t['ctrl'][:, :7])  # first 7 joints
    
    all_torques = np.concatenate(all_torques, axis=0)
    
    print(f"\n4. Control Effort:")
    print(f"   Mean abs torque: {np.mean(np.abs(all_torques)):.2f} Nm")
    print(f"   Max abs torque: {np.max(np.abs(all_torques)):.2f} Nm")
    print(f"   Per-joint max torque:")
    for j in range(7):
        print(f"      Joint {j+1}: {np.max(np.abs(all_torques[:, j])):.2f} Nm")
    
    # 5. Velocity statistics
    all_qvel = []
    for t in trajectories:
        all_qvel.append(t['qvel'][:, :7])
    
    all_qvel = np.concatenate(all_qvel, axis=0)
    
    print(f"\n5. Joint Velocity:")
    print(f"   Mean abs velocity: {np.mean(np.abs(all_qvel)):.3f} rad/s")
    print(f"   Max abs velocity: {np.max(np.abs(all_qvel)):.3f} rad/s")
    
    # 6. Data quality checks
    print(f"\n6. Data Quality Checks:")
    
    # Check for NaN or Inf
    has_nan = 0
    has_inf = 0
    for t in trajectories:
        if np.any(np.isnan(t['qpos'])) or np.any(np.isnan(t['ctrl'])):
            has_nan += 1
        if np.any(np.isinf(t['qpos'])) or np.any(np.isinf(t['ctrl'])):
            has_inf += 1
    
    print(f"   Trajectories with NaN: {has_nan} / {num_trajs}")
    print(f"   Trajectories with Inf: {has_inf} / {num_trajs}")
    
    # Check for consistent timesteps
    dt_variations = []
    for t in trajectories:
        if len(t['times']) > 1:
            dts = np.diff(t['times'])
            dt_variations.append(np.std(dts))
    
    print(f"   Timestep consistency (std): {np.mean(dt_variations):.6f}s")
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80 + "\n")

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("Batch generation for FR3 figure-8 drawing")
    
    # Basic settings
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--num-trajectories", type=int, default=5000,
                        help="Number of trajectories to generate")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N trajectories")
    parser.add_argument("--output-dir", type=str, default="./batch_data",
                        help="Directory to save generated data")
    parser.add_argument("--base-seed", type=int, default=1000,
                        help="Base random seed (each trajectory uses base_seed + index)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after generation")
    
    # Trajectory parameters
    parser.add_argument("--dt", type=float, default=0.0, 
                        help="controller dt; 0=>use model.opt.timestep")
    parser.add_argument("--x-plane", type=float, default=0.40)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--omega", type=float, default=0.6)
    parser.add_argument("--ay", type=float, default=None)
    parser.add_argument("--az", type=float, default=None)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--initial-hold", type=float, default=1.0,
                        help="Time to hold at initial position before drawing")
    
    # IK/PD parameters
    parser.add_argument("--kp-cart", type=float, default=8.0)
    parser.add_argument("--kp-ori", type=float, default=6.0)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--kp", type=float, nargs=7, 
                        default=[4500,4500,3500,3500,2000,2000,2000])
    parser.add_argument("--kv", type=float, nargs=7, 
                        default=[450,450,350,350,200,200,200])
    
    args = parser.parse_args()
    
    # Run batch generation
    trajectories, save_path = batch_generate(args)
    
    print(f"\nAll done! Data saved to: {save_path}")

if __name__ == "__main__":
    main() 
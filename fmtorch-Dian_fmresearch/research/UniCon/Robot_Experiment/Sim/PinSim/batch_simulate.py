#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch simulation script for pin_fr3_draw_eight.py
Run multiple seeds in parallel, save trajectories incrementally every 30 runs.
"""

import os
import sys
import numpy as np
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
import time

def run_single_simulation(args_tuple):
    """Run a single simulation with given seed. Returns trajectory data."""
    seed, script_path, cyc, dt = args_tuple
    
    # Import here to avoid issues with multiprocessing
    import pinocchio as pin
    
    # Build model
    current_dir = os.path.dirname(script_path)
    URDF_PATH = os.path.join(current_dir, "panda_description", "urdf", "panda_stick.urdf")
    PACKAGE_DIRS = [os.path.join(current_dir, "panda_description")]
    
    model, _, _ = pin.buildModelsFromUrdf(URDF_PATH, PACKAGE_DIRS)
    data = model.createData()
    
    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    
    # Set random seed
    np.random.seed(seed)
    
    # Initial state
    initial_qpos = np.array([0.6923, -0.6893, -0.5691, -2.4745, -2.5981, 2.7076, 0.7105], dtype=np.float64)
    q = initial_qpos.copy()
    dq = np.zeros(model.nv)
    
    # Workspace & trajectory parameters
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p0_ref = data.oMf[ee_fid].translation.copy()
    
    x_plane = p0_ref[0] + 0.25
    cy, cz = p0_ref[1], p0_ref[2] + 0.12
    Ay, Az = 0.10, 0.05
    omega, phi = 0.6, 0.0
    
    # Randomize initial position
    random_scale = 1.5
    y_rand = cy + random_scale * Ay * (2.0 * np.random.rand() - 1.0)
    z_rand = cz + random_scale * Az * (2.0 * np.random.rand() - 1.0)
    target_ee_init = np.array([x_plane, y_rand, z_rand])
    
    # Gains
    kp_pos, kp_ori = 16.0, 4.0
    damp = 0.14
    qdot_lim = 2.2
    rate_alpha = 0.60
    
    Kp_j = np.array([70,70,60,60,50,45,35], dtype=float)
    Kv_j = 2.0 * np.sqrt(Kp_j) * 1.4
    
    # Helper functions (inline to avoid pickling issues)
    def ema(prev, new, alpha):
        return alpha*prev + (1.0-alpha)*new
    
    def so3_log(R):
        tr = np.trace(R); c = np.clip((tr-1.0)*0.5, -1.0, 1.0); th = np.arccos(c)
        if th < 1e-8:
            return 0.5*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        w = (1.0/(2.0*np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        return th*w
    
    def resolved_rate_step6(model, data, frame_id, q, x_des, v_des, R_des,
                            kp_pos, kp_ori, base_damp, qdot_limit, rate_alpha, dt):
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
        
        speed_scale = np.clip(smin / 0.02, 0.25, 1.0)
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
    
    def nearest_phase(y0, z0, cy, cz, Ay, Az, omega, phi, K=1200):
        thetas = np.linspace(0.0, 2.0*np.pi/omega, K, endpoint=False)
        y = cy + Ay*np.sin(omega*thetas)
        z = cz + Az*np.sin(2.0*omega*thetas + phi)
        i = np.argmin((y - y0)**2 + (z - z0)**2)
        return float(thetas[i])
    
    def land_to_phase(model, data, ee_fid, q, dq, x_plane, cy, cz, Ay, Az, omega, phi,
                      phase0, dt, kp_pos, kp_ori, damp, qdot_lim, rate_alpha, Kp_j, Kv_j,
                      iters=500, tol=5e-5):
        y = cy + Ay*np.sin(omega*phase0)
        z = cz + Az*np.sin(2.0*omega*phase0 + phi)
        vy = Ay*omega*np.cos(omega*phase0)
        vz = 2.0*Az*omega*np.cos(2.0*omega*phase0 + phi)
        x_des = np.array([x_plane, y, z], dtype=float)
        v_des = np.array([0.0, vy, vz], dtype=float)
        
        ez = np.array([1.,0.,0.])
        tang = np.array([0.0, vy, vz])
        if np.linalg.norm(tang) < 1e-8:
            up = np.array([0.,0.,1.])
            ex = np.cross(up, ez); ex /= np.linalg.norm(ex)
            ey = np.cross(ez, ex)
            R_des = np.column_stack([ex, ey, ez])
        else:
            ey = tang/np.linalg.norm(tang)
            ex = np.cross(ey, ez); ex /= np.linalg.norm(ex)
            R_des = np.column_stack([ex, ey, ez])
        
        for it in range(iters):
            q_ref, qdot_ref, _, _ = resolved_rate_step6(
                model, data, ee_fid, q, x_des, v_des, R_des,
                kp_pos, kp_ori, damp, qdot_lim, rate_alpha, dt
            )
            e = pin.difference(model, q, q_ref)
            ed = (qdot_ref - dq)
            ddq_cmd = Kp_j * e + Kv_j * ed
            tau = pin.rnea(model, data, q, dq, ddq_cmd)
            M = pin.crba(model, data, q)
            h = pin.nle(model, data, q, dq)
            ddq = np.linalg.solve(M, tau - h)
            dq = dq + ddq*dt
            q = pin.integrate(model, q, dq*dt)
            
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            p = data.oMf[ee_fid].translation
            if np.linalg.norm(p - x_des) < tol:
                break
        
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        p = data.oMf[ee_fid].translation
        phase0 = nearest_phase(p[1], p[2], cy, cz, Ay, Az, omega, phi)
        return q, dq, phase0
    
    # Target orientation for IK
    ez = np.array([1.,0.,0.])
    up = np.array([0.,0.,1.])
    ey = np.cross(up, ez)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    R_init = np.column_stack([ex, ey, ez])
    
    # Solve IK for randomized initial position
    for _ in range(300):
        q_ref, qdot_ref, _, pos_err = resolved_rate_step6(
            model, data, ee_fid, q, target_ee_init, np.zeros(3), R_init,
            kp_pos, kp_ori, damp, qdot_lim, rate_alpha, dt
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
    
    # Landing to nearest phase
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p0 = data.oMf[ee_fid].translation.copy()
    
    phase0_guess = nearest_phase(p0[1], p0[2], cy, cz, Ay, Az, omega, phi)
    q, dq, phase0 = land_to_phase(
        model, data, ee_fid, q, dq,
        x_plane, cy, cz, Ay, Az, omega, phi,
        phase0_guess, dt,
        kp_pos, kp_ori, damp, qdot_lim, rate_alpha,
        Kp_j, Kv_j,
        iters=500, tol=5e-5
    )
    
    # Store initial state
    q_init = q.copy()
    dq_init = dq.copy()
    
    # Calculate total time
    T_CYCLE = 2.0 * np.pi / omega
    NUM_CYCLES = cyc
    T_TOTAL = NUM_CYCLES * T_CYCLE
    RAMP_T = 1.2
    
    # Trajectory function
    def desired_pose(t):
        tau = t + phase0
        r = 1.0  # No ramp for batch mode
        Ay_eff, Az_eff = r * Ay, r * Az
        omega_eff = omega
        
        y = cy + Ay_eff*np.sin(omega_eff*tau)
        z = cz + Az_eff*np.sin(2.0*omega_eff*tau + phi)
        vy = Ay_eff*omega_eff*np.cos(omega_eff*tau)
        vz = 2.0*Az_eff*omega_eff*np.cos(2.0*omega_eff*tau + phi)
        
        x = np.array([x_plane, y, z])
        v = np.array([0.0, vy, vz])
        
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
        return x, v, R
    
    # Torque settings
    TAU_LIM = np.array([85,85,85,85,20,20,20], dtype=float)
    TAU_LP_ALPHA = 0.90
    TAU_SLEW = np.array([500,500,500,500,300,300,300], dtype=float)
    QDOT_LP_ALPHA = 0.85
    
    # Main simulation loop
    N = int(T_TOTAL / dt)
    t = 0.0
    
    q_log = np.zeros((N, model.nq))
    dq_log = np.zeros((N, model.nv))
    tau_log = np.zeros((N, model.nv))
    t_log = np.zeros(N)
    ee_pos_log = np.zeros((N, 3))
    ee_pos_des_log = np.zeros((N, 3))
    
    tau_prev = np.zeros(model.nv)
    qdot_ref_filt = np.zeros(model.nv)
    
    for i in range(N):
        x_des, v_des, R_des = desired_pose(t)
        
        q_ref, qdot_ref, smin, pos_err = resolved_rate_step6(
            model, data, ee_fid, q, x_des, v_des, R_des,
            kp_pos, kp_ori, damp, qdot_lim, rate_alpha, dt
        )
        
        qdot_ref_filt = ema(qdot_ref_filt, qdot_ref, QDOT_LP_ALPHA)
        
        e = pin.difference(model, q, q_ref)
        ed = qdot_ref_filt - dq
        ddq_cmd = Kp_j * e + Kv_j * ed
        
        tau_raw = pin.rnea(model, data, q, dq, ddq_cmd)
        tau_raw = np.clip(tau_raw, -TAU_LIM, TAU_LIM)
        
        tau_filt = ema(tau_prev, tau_raw, TAU_LP_ALPHA)
        max_step = TAU_SLEW * dt
        delta = np.clip(tau_filt - tau_prev, -max_step, max_step)
        tau = tau_prev + delta
        tau_prev = tau.copy()
        
        M = pin.crba(model, data, q)
        h = pin.nle(model, data, q, dq)
        ddq = np.linalg.solve(M, tau - h)
        
        dq = dq + ddq*dt
        q = pin.integrate(model, q, dq*dt)
        
        q_log[i] = q
        dq_log[i] = dq
        tau_log[i] = tau
        t_log[i] = t
        
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        ee = data.oMf[ee_fid].translation.copy()
        ee_pos_log[i] = ee
        ee_pos_des_log[i] = x_des
        
        t += dt
    
    # Return all necessary data for lossless replay
    result = {
        'seed': seed,
        'q_init': q_init,
        'dq_init': dq_init,
        'q_log': q_log,
        'dq_log': dq_log,
        'tau_log': tau_log,
        't_log': t_log,
        'ee_pos_log': ee_pos_log,
        'ee_pos_des_log': ee_pos_des_log,
        'phase0': phase0,
        'random_init_yz': np.array([y_rand, z_rand]),
        'N': N,
        'dt': dt,
        'cyc': cyc,
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Batch simulation for FR3 figure-8 drawing')
    parser.add_argument('--start_seed', type=int, default=1,
                        help='Starting seed (default: 1)')
    parser.add_argument('--end_seed', type=int, default=7001,
                        help='Ending seed (exclusive, default: 7001)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--cyc', type=float, default=1.5,
                        help='Number of cycles to draw (default: 1.5)')
    parser.add_argument('--dt', type=float, default=0.002,
                        help='Time step (default: 0.002)')
    parser.add_argument('--save_every', type=int, default=30,
                        help='Save every N simulations (default: 30)')
    parser.add_argument('--output', type=str, default='batch_trajectories.npz',
                        help='Output NPZ file path')
    args = parser.parse_args()
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print(f"Using {num_workers} workers")
    
    # Get script path
    script_path = os.path.abspath(__file__)
    
    # Prepare seeds
    seeds = list(range(args.start_seed, args.end_seed))
    total_sims = len(seeds)
    print(f"Running {total_sims} simulations (seeds {args.start_seed} to {args.end_seed-1})")
    
    # Storage
    all_results = []
    completed = 0
    start_time = time.time()
    
    # Create argument tuples
    arg_tuples = [(seed, script_path, args.cyc, args.dt) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(run_single_simulation, arg_tuples):
            all_results.append(result)
            completed += 1
            
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (total_sims - completed) / rate if rate > 0 else 0
            
            print(f"Progress: {completed}/{total_sims} ({100*completed/total_sims:.1f}%) | "
                  f"Rate: {rate:.2f} sim/s | ETA: {eta/60:.1f} min | "
                  f"Last seed: {result['seed']}")
            
            # Save every N simulations
            if completed % args.save_every == 0 or completed == total_sims:
                print(f"Saving checkpoint at {completed} simulations...")
                save_results(all_results, args.output)
    
    # Final save
    print(f"\nAll simulations complete! Total time: {(time.time()-start_time)/60:.1f} min")
    print(f"Final save to {args.output}")
    save_results(all_results, args.output)
    
    # Print summary statistics
    print_summary(all_results)


def save_results(results, output_path):
    """Save results to NPZ file with proper structure."""
    if not results:
        return
    
    # Sort by seed
    results = sorted(results, key=lambda x: x['seed'])
    
    # Prepare arrays
    seeds = np.array([r['seed'] for r in results])
    N_max = max(r['N'] for r in results)
    n_results = len(results)
    nq = results[0]['q_log'].shape[1]
    nv = results[0]['dq_log'].shape[1]
    
    # Pre-allocate arrays
    q_init_all = np.zeros((n_results, nq))
    dq_init_all = np.zeros((n_results, nv))
    q_log_all = np.zeros((n_results, N_max, nq))
    dq_log_all = np.zeros((n_results, N_max, nv))
    tau_log_all = np.zeros((n_results, N_max, nv))
    t_log_all = np.zeros((n_results, N_max))
    ee_pos_log_all = np.zeros((n_results, N_max, 3))
    ee_pos_des_log_all = np.zeros((n_results, N_max, 3))
    phase0_all = np.zeros(n_results)
    random_init_yz_all = np.zeros((n_results, 2))
    N_all = np.zeros(n_results, dtype=int)
    
    for i, r in enumerate(results):
        N = r['N']
        q_init_all[i] = r['q_init']
        dq_init_all[i] = r['dq_init']
        q_log_all[i, :N] = r['q_log']
        dq_log_all[i, :N] = r['dq_log']
        tau_log_all[i, :N] = r['tau_log']
        t_log_all[i, :N] = r['t_log']
        ee_pos_log_all[i, :N] = r['ee_pos_log']
        ee_pos_des_log_all[i, :N] = r['ee_pos_des_log']
        phase0_all[i] = r['phase0']
        random_init_yz_all[i] = r['random_init_yz']
        N_all[i] = N
    
    # Save to NPZ
    np.savez_compressed(
        output_path,
        seeds=seeds,
        q_init=q_init_all,
        dq_init=dq_init_all,
        q_log=q_log_all,
        dq_log=dq_log_all,
        tau_log=tau_log_all,
        t_log=t_log_all,
        ee_pos_log=ee_pos_log_all,
        ee_pos_des_log=ee_pos_des_log_all,
        phase0=phase0_all,
        random_init_yz=random_init_yz_all,
        N_per_traj=N_all,
        dt=results[0]['dt'],
        cyc=results[0]['cyc'],
    )
    print(f"Saved {n_results} trajectories to {output_path}")


def print_summary(results):
    """Print summary statistics."""
    if not results:
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Tracking errors
    errors = []
    for r in results:
        ee_err = np.linalg.norm(r['ee_pos_log'] - r['ee_pos_des_log'], axis=1)
        errors.append({
            'mean': np.mean(ee_err),
            'max': np.max(ee_err),
            'std': np.std(ee_err),
        })
    
    mean_errors = [e['mean'] for e in errors]
    max_errors = [e['max'] for e in errors]
    
    print(f"Tracking Error (position):")
    print(f"  Mean error:  {np.mean(mean_errors)*1000:.3f} ± {np.std(mean_errors)*1000:.3f} mm")
    print(f"  Max error:   {np.mean(max_errors)*1000:.3f} ± {np.std(max_errors)*1000:.3f} mm")
    print(f"  Best case:   {np.min(mean_errors)*1000:.3f} mm (mean), {np.min(max_errors)*1000:.3f} mm (max)")
    print(f"  Worst case:  {np.max(mean_errors)*1000:.3f} mm (mean), {np.max(max_errors)*1000:.3f} mm (max)")
    
    # Torque statistics
    tau_all = np.concatenate([r['tau_log'] for r in results], axis=0)
    print(f"\nTorque Statistics:")
    print(f"  Mean |τ|:  {np.mean(np.abs(tau_all), axis=0)}")
    print(f"  Max |τ|:   {np.max(np.abs(tau_all), axis=0)}")
    
    print("="*60)


if __name__ == "__main__":
    main() 
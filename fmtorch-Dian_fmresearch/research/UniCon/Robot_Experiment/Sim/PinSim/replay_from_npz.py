#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Replay trajectories from batch NPZ file.
Verify lossless replay by applying stored torques and comparing states.
"""

import os
import numpy as np
import argparse
import pinocchio as pin
import matplotlib.pyplot as plt

def load_trajectories(npz_path):
    """Load all trajectories from NPZ file."""
    data = np.load(npz_path)
    print(f"Loaded NPZ file: {npz_path}")
    print(f"  Number of trajectories: {len(data['seeds'])}")
    print(f"  Seeds: {data['seeds'][0]} to {data['seeds'][-1]}")
    print(f"  dt: {data['dt']}")
    print(f"  cyc: {data['cyc']}")
    print(f"  Data keys: {list(data.keys())}")
    return data

def replay_single_trajectory(model, data, ee_fid, traj_idx, npz_data, verbose=True):
    """Replay a single trajectory and compute differences."""
    
    # Extract trajectory data
    q_init = npz_data['q_init'][traj_idx]
    dq_init = npz_data['dq_init'][traj_idx]
    tau_log = npz_data['tau_log'][traj_idx]
    q_orig = npz_data['q_log'][traj_idx]
    dq_orig = npz_data['dq_log'][traj_idx]
    N = npz_data['N_per_traj'][traj_idx]
    dt = float(npz_data['dt'])
    seed = npz_data['seeds'][traj_idx]
    
    if verbose:
        print(f"\nReplaying trajectory {traj_idx} (seed={seed}, N={N})")
    
    # Initialize state
    q_replay = q_init.copy()
    dq_replay = dq_init.copy()
    
    # Replay logs
    q_replay_log = np.zeros((N, model.nq))
    dq_replay_log = np.zeros((N, model.nv))
    
    # Replay simulation
    for i in range(N):
        tau = tau_log[i]
        
        # Forward dynamics
        M = pin.crba(model, data, q_replay)
        h = pin.nle(model, data, q_replay, dq_replay)
        ddq = np.linalg.solve(M, tau - h)
        
        dq_replay = dq_replay + ddq * dt
        q_replay = pin.integrate(model, q_replay, dq_replay * dt)
        
        q_replay_log[i] = q_replay
        dq_replay_log[i] = dq_replay
    
    # Compute differences
    q_diff = np.zeros(N)
    dq_diff = np.zeros(N)
    for i in range(N):
        q_diff[i] = np.linalg.norm(pin.difference(model, q_orig[i], q_replay_log[i]))
        dq_diff[i] = np.linalg.norm(dq_orig[i] - dq_replay_log[i])
    
    if verbose:
        print(f"  q error:  max={np.max(q_diff):.6e}, mean={np.mean(q_diff):.6e}")
        print(f"  dq error: max={np.max(dq_diff):.6e}, mean={np.mean(dq_diff):.6e}")
    
    return {
        'q_diff': q_diff,
        'dq_diff': dq_diff,
        'q_replay': q_replay_log,
        'dq_replay': dq_replay_log,
        'seed': seed,
    }

def main():
    parser = argparse.ArgumentParser(description='Replay trajectories from NPZ file')
    parser.add_argument('npz_file', type=str,
                        help='Path to NPZ file containing trajectories')
    parser.add_argument('--traj_idx', type=int, default=None,
                        help='Replay specific trajectory index (default: all)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot results')
    parser.add_argument('--urdf', type=str, default=None,
                        help='Path to URDF file (default: auto-detect)')
    args = parser.parse_args()
    
    # Load data
    npz_data = load_trajectories(args.npz_file)
    
    # Load robot model
    if args.urdf is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "panda_description", "urdf", "panda_stick.urdf")
        package_dirs = [os.path.join(script_dir, "panda_description")]
    else:
        urdf_path = args.urdf
        package_dirs = [os.path.dirname(urdf_path)]
    
    model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs)
    data = model.createData()
    
    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    
    print(f"\nLoaded robot model: nq={model.nq}, nv={model.nv}")
    
    # Replay trajectories
    n_trajs = len(npz_data['seeds'])
    
    if args.traj_idx is not None:
        # Replay single trajectory
        result = replay_single_trajectory(model, data, ee_fid, args.traj_idx, npz_data, verbose=True)
        
        if args.plot:
            t_log = npz_data['t_log'][args.traj_idx]
            N = npz_data['N_per_traj'][args.traj_idx]
            
            plt.figure(figsize=(12, 5))
            plt.subplot(2, 1, 1)
            plt.plot(t_log[:N], result['q_diff'])
            plt.grid(True)
            plt.ylabel('||q error|| (rad)')
            plt.title(f"Replay Error - Trajectory {args.traj_idx} (seed={result['seed']})")
            
            plt.subplot(2, 1, 2)
            plt.plot(t_log[:N], result['dq_diff'])
            plt.grid(True)
            plt.ylabel('||dq error|| (rad/s)')
            plt.xlabel('t (s)')
            plt.tight_layout()
            plt.show()
    else:
        # Replay all trajectories
        print(f"\nReplaying all {n_trajs} trajectories...")
        
        all_q_errors = []
        all_dq_errors = []
        
        for idx in range(n_trajs):
            result = replay_single_trajectory(model, data, ee_fid, idx, npz_data, verbose=False)
            all_q_errors.append({
                'max': np.max(result['q_diff']),
                'mean': np.mean(result['q_diff']),
            })
            all_dq_errors.append({
                'max': np.max(result['dq_diff']),
                'mean': np.mean(result['dq_diff']),
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Replayed {idx+1}/{n_trajs} trajectories...")
        
        # Print statistics
        q_max_errors = [e['max'] for e in all_q_errors]
        q_mean_errors = [e['mean'] for e in all_q_errors]
        dq_max_errors = [e['max'] for e in all_dq_errors]
        dq_mean_errors = [e['mean'] for e in all_dq_errors]
        
        print("\n" + "="*60)
        print("REPLAY VERIFICATION STATISTICS")
        print("="*60)
        print(f"Position error (q):")
        print(f"  Mean of max errors:  {np.mean(q_max_errors):.6e} ± {np.std(q_max_errors):.6e} rad")
        print(f"  Mean of mean errors: {np.mean(q_mean_errors):.6e} ± {np.std(q_mean_errors):.6e} rad")
        print(f"  Overall max error:   {np.max(q_max_errors):.6e} rad")
        
        print(f"\nVelocity error (dq):")
        print(f"  Mean of max errors:  {np.mean(dq_max_errors):.6e} ± {np.std(dq_max_errors):.6e} rad/s")
        print(f"  Mean of mean errors: {np.mean(dq_mean_errors):.6e} ± {np.std(dq_mean_errors):.6e} rad/s")
        print(f"  Overall max error:   {np.max(dq_max_errors):.6e} rad/s")
        print("="*60)
        
        if args.plot:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            axes[0, 0].hist(q_max_errors, bins=50)
            axes[0, 0].set_xlabel('Max q error (rad)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Distribution of Max Position Errors')
            axes[0, 0].grid(True)
            
            axes[0, 1].hist(q_mean_errors, bins=50)
            axes[0, 1].set_xlabel('Mean q error (rad)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Distribution of Mean Position Errors')
            axes[0, 1].grid(True)
            
            axes[1, 0].hist(dq_max_errors, bins=50)
            axes[1, 0].set_xlabel('Max dq error (rad/s)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Distribution of Max Velocity Errors')
            axes[1, 0].grid(True)
            
            axes[1, 1].hist(dq_mean_errors, bins=50)
            axes[1, 1].set_xlabel('Mean dq error (rad/s)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Distribution of Mean Velocity Errors')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Sanity Check Tool for RobotPin_Dataset_10k.npz

This script performs comprehensive validation of the robot trajectory dataset:

1. **Trajectory Replay Validation** (100 random samples):
   - Replays trajectories using stored torques (tau_log) from initial states (q_init, dq_init)
   - Uses Pinocchio forward dynamics (same as pin_fr3_draw_eight.py)
   - Verifies that replayed states match original states with high precision
   - Reports maximum and mean errors in joint positions (q) and velocities (dq)

2. **Forward Kinematics Consistency Check** (100 random samples):
   - Computes end-effector positions from q_log using forward kinematics
   - Compares against stored ee_pos_log
   - Validates that stored EE positions are consistent with joint positions

3. **Figure-8 Trajectory Visualization** (20 random samples):
   - Plots actual EE trajectories on YZ plane
   - Overlays desired figure-8 reference trajectory
   - Visual verification that robot draws correct figure-8 patterns
   - Saves plot for inspection

Usage:
    python sanity_check_dataset.py --data RobotPin_Dataset_10k.npz --n_replay 100 --n_plot 20

Requirements:
    - Pinocchio (for dynamics and kinematics)
    - NumPy
    - Matplotlib
"""

import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pinocchio as pin


def load_dataset(npz_path):
    """Load dataset from NPZ file"""
    data = np.load(npz_path)
    return {
        'seeds': data['seeds'],
        'q_init': data['q_init'],
        'dq_init': data['dq_init'],
        'q_log': data['q_log'],
        'dq_log': data['dq_log'],
        'tau_log': data['tau_log'],
        't_log': data['t_log'],
        'ee_pos_log': data['ee_pos_log'],
        'ee_pos_des_log': data['ee_pos_des_log'],
        'phase0': data['phase0'],
        'random_init_yz': data['random_init_yz'],
        'N_per_traj': data['N_per_traj'],
        'dt': float(data['dt']),
        'cyc': float(data['cyc']),
    }


def setup_robot_model(script_dir):
    """Setup Pinocchio robot model"""
    URDF_PATH = os.path.join(script_dir, "panda_description", "urdf", "panda_stick.urdf")
    PACKAGE_DIRS = [os.path.join(script_dir, "panda_description")]
    
    if not os.path.isfile(URDF_PATH):
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")
    
    model, _, _ = pin.buildModelsFromUrdf(URDF_PATH, PACKAGE_DIRS)
    data = model.createData()
    
    EE_NAME = "panda_tool_tip"
    ee_fid = model.getFrameId(EE_NAME)
    
    return model, data, ee_fid


def replay_trajectory(model, data, q_init, dq_init, tau_log, dt, N):
    """
    Replay trajectory using stored torques from initial state.
    Uses same forward dynamics as pin_fr3_draw_eight.py
    
    Returns:
        q_replay_log: (N, nq) replayed joint positions
        dq_replay_log: (N, nv) replayed joint velocities
    """
    q = q_init.copy()
    dq = dq_init.copy()
    
    q_replay_log = np.zeros((N, model.nq))
    dq_replay_log = np.zeros((N, model.nv))
    
    for i in range(N):
        tau = tau_log[i]
        
        # Forward dynamics (same as original simulation)
        M = pin.crba(model, data, q)
        h = pin.nle(model, data, q, dq)
        ddq = np.linalg.solve(M, tau - h)
        
        # Semi-implicit Euler integration (same as original)
        dq = dq + ddq * dt
        q = pin.integrate(model, q, dq * dt)
        
        q_replay_log[i] = q
        dq_replay_log[i] = dq
    
    return q_replay_log, dq_replay_log


def compute_ee_positions(model, data, ee_fid, q_log):
    """
    Compute end-effector positions from joint positions using forward kinematics
    
    Returns:
        ee_positions: (N, 3) end-effector positions
    """
    N = q_log.shape[0]
    ee_positions = np.zeros((N, 3))
    
    for i in range(N):
        pin.forwardKinematics(model, data, q_log[i])
        pin.updateFramePlacements(model, data)
        ee_positions[i] = data.oMf[ee_fid].translation.copy()
    
    return ee_positions


def check_replay_accuracy(model, q_log_orig, dq_log_orig, q_replay_log, dq_replay_log):
    """
    Compute state differences between original and replayed trajectories
    
    Returns:
        dict with max/mean errors for q and dq
    """
    N = q_log_orig.shape[0]
    
    q_diff = np.zeros(N)
    dq_diff = np.zeros(N)
    
    for i in range(N):
        # Use Pinocchio's difference for manifold-aware comparison
        q_diff[i] = np.linalg.norm(pin.difference(model, q_log_orig[i], q_replay_log[i]))
        dq_diff[i] = np.linalg.norm(dq_log_orig[i] - dq_replay_log[i])
    
    return {
        'q_max': np.max(q_diff),
        'q_mean': np.mean(q_diff),
        'dq_max': np.max(dq_diff),
        'dq_mean': np.mean(dq_diff),
        'q_diff_series': q_diff,
        'dq_diff_series': dq_diff,
    }


def check_fk_consistency(ee_pos_log_stored, ee_pos_computed):
    """
    Check consistency between stored EE positions and FK-computed positions
    
    Returns:
        dict with max/mean position errors
    """
    pos_diff = np.linalg.norm(ee_pos_log_stored - ee_pos_computed, axis=1)
    
    return {
        'pos_max': np.max(pos_diff),
        'pos_mean': np.mean(pos_diff),
        'pos_diff_series': pos_diff,
    }


def reconstruct_reference_trajectory(phase0, cyc, dt, N_traj):
    """
    Reconstruct the reference figure-8 trajectory
    Uses same parameters as pin_fr3_draw_eight.py
    """
    # Figure-8 parameters (from pin_fr3_draw_eight.py)
    # Note: We use approximate values since exact workspace depends on initial q
    cy, cz = 0.0, 0.62  # Approximate center (will be relative to actual trajectory)
    Ay, Az = 0.10, 0.05
    omega, phi = 0.6, 0.0
    
    t_array = np.arange(N_traj) * dt
    y_ref = np.zeros(N_traj)
    z_ref = np.zeros(N_traj)
    
    for i, t in enumerate(t_array):
        tau = t + phase0
        y_ref[i] = cy + Ay * np.sin(omega * tau)
        z_ref[i] = cz + Az * np.sin(2.0 * omega * tau + phi)
    
    return y_ref, z_ref


def main():
    parser = argparse.ArgumentParser(description='Sanity check for robot trajectory dataset')
    parser.add_argument('--data', type=str, default='RobotPin_Dataset_10k.npz',
                        help='Path to NPZ dataset file')
    parser.add_argument('--n_replay', type=int, default=100,
                        help='Number of trajectories to validate (replay + FK check)')
    parser.add_argument('--n_plot', type=int, default=20,
                        help='Number of trajectories to plot')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*80)
    print("Dataset Sanity Check Tool")
    print("="*80)
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load dataset
    print(f"\n[1/5] Loading dataset: {args.data}")
    dataset = load_dataset(args.data)
    n_total = len(dataset['seeds'])
    print(f"      Total trajectories: {n_total}")
    print(f"      dt: {dataset['dt']:.6f} s")
    print(f"      cycles: {dataset['cyc']:.2f}")
    
    # Setup robot model
    print(f"\n[2/5] Setting up robot model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model, data, ee_fid = setup_robot_model(script_dir)
    print(f"      Model loaded: nq={model.nq}, nv={model.nv}")
    
    # Sample trajectories for validation
    n_validate = min(args.n_replay, n_total)
    validate_indices = np.random.choice(n_total, size=n_validate, replace=False)
    print(f"\n[3/5] Validating {n_validate} random trajectories...")
    
    replay_errors = []
    fk_errors = []
    
    for idx_count, traj_idx in enumerate(validate_indices):
        # Get trajectory data
        q_init = dataset['q_init'][traj_idx]
        dq_init = dataset['dq_init'][traj_idx]
        N_traj = dataset['N_per_traj'][traj_idx]
        q_log_orig = dataset['q_log'][traj_idx, :N_traj]
        dq_log_orig = dataset['dq_log'][traj_idx, :N_traj]
        tau_log = dataset['tau_log'][traj_idx, :N_traj]
        ee_pos_log_stored = dataset['ee_pos_log'][traj_idx, :N_traj]
        dt = dataset['dt']
        
        # Test 1: Replay trajectory
        q_replay, dq_replay = replay_trajectory(model, data, q_init, dq_init, tau_log, dt, N_traj)
        replay_error = check_replay_accuracy(model, q_log_orig, dq_log_orig, q_replay, dq_replay)
        replay_errors.append(replay_error)
        
        # Test 2: FK consistency check
        ee_pos_computed = compute_ee_positions(model, data, ee_fid, q_log_orig)
        fk_error = check_fk_consistency(ee_pos_log_stored, ee_pos_computed)
        fk_errors.append(fk_error)
        
        # Progress
        if (idx_count + 1) % 10 == 0 or (idx_count + 1) == n_validate:
            print(f"      Progress: {idx_count+1}/{n_validate} trajectories validated")
    
    # Aggregate results
    print(f"\n[4/5] Results Summary:")
    print("-"*80)
    
    print("\n  Test 1: Trajectory Replay Accuracy")
    print("  " + "-"*76)
    q_max_errors = [e['q_max'] for e in replay_errors]
    q_mean_errors = [e['q_mean'] for e in replay_errors]
    dq_max_errors = [e['dq_max'] for e in replay_errors]
    dq_mean_errors = [e['dq_mean'] for e in replay_errors]
    
    print(f"    Joint Position (q):")
    print(f"      Max error:   {np.max(q_max_errors):.6e} rad (worst case)")
    print(f"      Mean error:  {np.mean(q_mean_errors):.6e} rad (average across all)")
    print(f"      Best case:   {np.min(q_max_errors):.6e} rad (max error)")
    
    print(f"\n    Joint Velocity (dq):")
    print(f"      Max error:   {np.max(dq_max_errors):.6e} rad/s (worst case)")
    print(f"      Mean error:  {np.mean(dq_mean_errors):.6e} rad/s (average across all)")
    print(f"      Best case:   {np.min(dq_max_errors):.6e} rad/s (max error)")
    
    # Determine if replay is accurate
    replay_pass = np.max(q_max_errors) < 1e-10 and np.max(dq_max_errors) < 1e-10
    if replay_pass:
        print(f"\n    ✅ PASS: Trajectory replay is accurate (errors < 1e-10)")
    else:
        print(f"\n    ⚠️  WARNING: Replay errors detected (errors >= 1e-10)")
    
    print("\n  Test 2: Forward Kinematics Consistency")
    print("  " + "-"*76)
    pos_max_errors = [e['pos_max'] for e in fk_errors]
    pos_mean_errors = [e['pos_mean'] for e in fk_errors]
    
    print(f"    End-Effector Position:")
    print(f"      Max error:   {np.max(pos_max_errors):.6e} m (worst case)")
    print(f"      Mean error:  {np.mean(pos_mean_errors):.6e} m (average across all)")
    print(f"      Best case:   {np.min(pos_max_errors):.6e} m (max error)")
    
    fk_pass = np.max(pos_max_errors) < 1e-10
    if fk_pass:
        print(f"\n    ✅ PASS: FK consistency verified (errors < 1e-10)")
    else:
        print(f"\n    ⚠️  WARNING: FK inconsistencies detected (errors >= 1e-10)")
    
    # Plot figure-8 trajectories
    print(f"\n[5/5] Plotting {args.n_plot} random figure-8 trajectories...")
    n_plot = min(args.n_plot, n_total)
    plot_indices = np.random.choice(n_total, size=n_plot, replace=False)
    
    # Create grid of subplots (4x5 for 20 trajectories)
    n_rows = 4
    n_cols = 5
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    for plot_idx, traj_idx in enumerate(plot_indices):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get trajectory data
        N_traj = dataset['N_per_traj'][traj_idx]
        ee_pos_actual = dataset['ee_pos_log'][traj_idx, :N_traj]
        ee_pos_desired = dataset['ee_pos_des_log'][traj_idx, :N_traj]
        seed = dataset['seeds'][traj_idx]
        phase0 = dataset['phase0'][traj_idx]
        
        # Plot desired (reference) trajectory
        ax.plot(ee_pos_desired[:, 1], ee_pos_desired[:, 2], 
                '--', color='gray', alpha=0.5, linewidth=1.5, label='Desired')
        
        # Plot actual trajectory
        ax.plot(ee_pos_actual[:, 1], ee_pos_actual[:, 2], 
                '-', color='blue', linewidth=2, label='Actual', alpha=0.8)
        
        # Mark start and end
        ax.plot(ee_pos_actual[0, 1], ee_pos_actual[0, 2], 
                'go', markersize=6, zorder=5)
        ax.plot(ee_pos_actual[-1, 1], ee_pos_actual[-1, 2], 
                'ro', markersize=6, zorder=5)
        
        # Compute tracking error
        tracking_error = np.linalg.norm(ee_pos_actual - ee_pos_desired, axis=1)
        mean_error = np.mean(tracking_error) * 1000  # mm
        max_error = np.max(tracking_error) * 1000  # mm
        
        ax.set_title(f'Seed {seed}\nErr: {mean_error:.1f}±{max_error:.1f}mm', 
                     fontsize=9)
        ax.set_xlabel('Y (m)', fontsize=8)
        ax.set_ylabel('Z (m)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.tick_params(labelsize=7)
        
        if plot_idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    fig.suptitle(f'Figure-8 Trajectory Validation ({n_plot} Random Samples)\n'
                 f'Dataset: {os.path.basename(args.data)}', 
                 fontsize=16, fontweight='bold')
    
    output_path = os.path.join(args.output_dir, 'figure8_trajectories_validation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"      Saved: {output_path}")
    
    print("\n" + "="*80)
    print("Sanity Check Complete!")
    print("="*80)
    
    # Final verdict
    print("\n📊 FINAL VERDICT:")
    if replay_pass and fk_pass:
        print("  ✅ ALL TESTS PASSED - Dataset is valid and consistent!")
    else:
        print("  ⚠️  SOME TESTS FAILED - Please review warnings above")
    
    print("\n💡 Files generated:")
    print(f"  - {output_path}")
    print("\n")


if __name__ == '__main__':
    main()


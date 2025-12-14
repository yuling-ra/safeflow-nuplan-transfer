#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize B-Spline compression-reconstruction comparison
Show original vs reconstructed trajectories for a single sample
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from spline_utils import (
    normalized_times, bspline_design_matrix, bspline_deriv_design_matrix,
    ridge_pinv, encode_theta, decode_signal
)
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv, rmse
)


def compress_reconstruct(q, dq, tau, 
                         Mq, Mdq, Mtau, 
                         deg_q, deg_dq, deg_tau,
                         lam_q, lam_dq, lam_tau,
                         sample_idx=0):
    """
    Compress and reconstruct a single trajectory
    
    Returns:
        q_rec, dq_rec, tau_rec: reconstructed trajectories
        metrics: RMSE for each component
    """
    N, T, J = q.shape
    
    # Z-score normalization (per group)
    mq, sq = zscore_fit_stats(q)
    mdq, sdq = zscore_fit_stats(dq)
    mtau, stau = zscore_fit_stats(tau)
    
    qz = zscore_apply(q, mq, sq)
    dqz = zscore_apply(dq, mdq, sdq)
    tauz = zscore_apply(tau, mtau, stau)
    
    # Normalized time
    t = normalized_times(T)
    
    # Construct spline basis matrices
    Bq = bspline_design_matrix(t, Mq, deg_q)
    Pq = ridge_pinv(Bq, lam_q)
    
    Bdq = bspline_design_matrix(t, Mdq, deg_dq)
    Pdq = ridge_pinv(Bdq, lam_dq)
    
    Btau = bspline_design_matrix(t, Mtau, deg_tau)
    Ptau = ridge_pinv(Btau, lam_tau)
    
    # Encode-decode for the selected sample
    i = sample_idx
    
    # q
    theta_q = encode_theta(Pq, qz[i])
    qz_hat = decode_signal(Bq, theta_q)
    q_rec = zscore_inv(qz_hat[None, :, :], mq, sq)[0]
    
    # dq
    theta_dq = encode_theta(Pdq, dqz[i])
    dqz_hat = decode_signal(Bdq, theta_dq)
    dq_rec = zscore_inv(dqz_hat[None, :, :], mdq, sdq)[0]
    
    # tau
    theta_tau = encode_theta(Ptau, tauz[i])
    tauz_hat = decode_signal(Btau, theta_tau)
    tau_rec = zscore_inv(tauz_hat[None, :, :], mtau, stau)[0]
    
    # Calculate RMSE
    metrics = {
        'rmse_q': rmse(q[i], q_rec),
        'rmse_dq': rmse(dq[i], dq_rec),
        'rmse_tau': rmse(tau[i], tau_rec),
    }
    
    return q_rec, dq_rec, tau_rec, metrics


def plot_reconstruction_comparison(q_orig, q_rec, 
                                   quantity_name, 
                                   unit,
                                   title_suffix,
                                   out_path,
                                   rmse_val):
    """
    Plot original vs reconstructed trajectory for one physical quantity
    Each joint in a separate subplot
    
    Args:
        q_orig: (T, J) original trajectory
        q_rec: (T, J) reconstructed trajectory
        quantity_name: e.g., "Position", "Velocity", "Torque"
        unit: e.g., "rad", "rad/s", "Nm"
        title_suffix: e.g., "Strategy1"
        out_path: output file path
        rmse_val: RMSE value
    """
    T, J = q_orig.shape
    t = np.linspace(0, T-1, T)
    
    # Create figure with subplots for each joint
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7']
    
    for j in range(J):
        row = j // 3
        col = j % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot original and reconstructed
        ax.plot(t, q_orig[:, j], 'b-', linewidth=2, label='Original', alpha=0.8)
        ax.plot(t, q_rec[:, j], 'r--', linewidth=2, label='Reconstructed', alpha=0.8)
        
        # Calculate per-joint RMSE
        joint_rmse = np.sqrt(np.mean((q_orig[:, j] - q_rec[:, j])**2))
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel(f'{quantity_name} ({unit})', fontsize=10)
        ax.set_title(f'{joint_names[j]} (RMSE: {joint_rmse:.6f})', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    # Overall title
    fig.suptitle(f'{quantity_name} Trajectory: Original vs Reconstructed\n'
                 f'{title_suffix} | Overall RMSE: {rmse_val:.6f}', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize trajectory reconstruction comparison')
    
    # Data parameters
    parser.add_argument('--data', type=str, required=True, help='NPZ data file path')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--out', type=str, default='./reconstruction_comparison', help='Output directory')
    
    # Strategy 1: Fixed q high-fidelity
    parser.add_argument('--strategy1-Mq', type=int, default=128)
    parser.add_argument('--strategy1-Mdq', type=int, default=64)
    parser.add_argument('--strategy1-Mtau', type=int, default=64)
    parser.add_argument('--strategy1-deg-q', type=int, default=5)
    parser.add_argument('--strategy1-deg-dq', type=int, default=3)
    parser.add_argument('--strategy1-deg-tau', type=int, default=5)
    parser.add_argument('--strategy1-lam-q', type=float, default=1e-6)
    parser.add_argument('--strategy1-lam-dq', type=float, default=1e-6)
    parser.add_argument('--strategy1-lam-tau', type=float, default=1e-5)
    
    # Strategy 2: 1k budget
    parser.add_argument('--strategy2-Mq', type=int, default=110)
    parser.add_argument('--strategy2-Mdq', type=int, default=16)
    parser.add_argument('--strategy2-Mtau', type=int, default=16)
    parser.add_argument('--strategy2-deg-q', type=int, default=5)
    parser.add_argument('--strategy2-deg-dq', type=int, default=3)
    parser.add_argument('--strategy2-deg-tau', type=int, default=5)
    parser.add_argument('--strategy2-lam-q', type=float, default=1e-6)
    parser.add_argument('--strategy2-lam-dq', type=float, default=1e-6)
    parser.add_argument('--strategy2-lam-tau', type=float, default=1e-5)
    
    args = parser.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    print("="*80)
    print("B-Spline Trajectory Reconstruction Visualization")
    print("="*80)
    
    # Load data
    print(f"\n[Loading] Data from {args.data}...")
    dataset = load_dataset(args.data)
    q = dataset['q']
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    
    print(f"[Data] Shape: N={N}, T={T}, J={J}")
    print(f"[Sample] Visualizing sample index: {args.sample_idx}")
    
    if args.sample_idx >= N:
        print(f"[Error] Sample index {args.sample_idx} exceeds dataset size {N}")
        return
    
    # ========================================================================
    # Strategy 1: Fixed q high-fidelity
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 1: Fixed q High-Fidelity")
    print("="*80)
    print(f"Configuration:")
    print(f"  M: q={args.strategy1_Mq}, dq={args.strategy1_Mdq}, tau={args.strategy1_Mtau}")
    print(f"  Degree: q={args.strategy1_deg_q}, dq={args.strategy1_deg_dq}, tau={args.strategy1_deg_tau}")
    print(f"  Lambda: q={args.strategy1_lam_q}, dq={args.strategy1_lam_dq}, tau={args.strategy1_lam_tau}")
    print(f"  Latent dims: {args.strategy1_Mq*7 + args.strategy1_Mdq*7 + args.strategy1_Mtau*7}")
    
    q_rec1, dq_rec1, tau_rec1, metrics1 = compress_reconstruct(
        q, dq, tau,
        args.strategy1_Mq, args.strategy1_Mdq, args.strategy1_Mtau,
        args.strategy1_deg_q, args.strategy1_deg_dq, args.strategy1_deg_tau,
        args.strategy1_lam_q, args.strategy1_lam_dq, args.strategy1_lam_tau,
        args.sample_idx
    )
    
    print(f"\nReconstruction RMSE:")
    print(f"  Position (q):   {metrics1['rmse_q']:.6f}")
    print(f"  Velocity (dq):  {metrics1['rmse_dq']:.6f}")
    print(f"  Torque (tau):   {metrics1['rmse_tau']:.6f}")
    
    # Plot Strategy 1 results
    print(f"\n[Plotting] Generating Strategy 1 visualizations...")
    
    plot_reconstruction_comparison(
        q[args.sample_idx], q_rec1,
        'Position', 'rad',
        f'Strategy 1 (Latent: {args.strategy1_Mq*7 + args.strategy1_Mdq*7 + args.strategy1_Mtau*7})',
        os.path.join(args.out, 'strategy1_position_comparison.png'),
        metrics1['rmse_q']
    )
    
    plot_reconstruction_comparison(
        dq[args.sample_idx], dq_rec1,
        'Velocity', 'rad/s',
        f'Strategy 1 (Latent: {args.strategy1_Mq*7 + args.strategy1_Mdq*7 + args.strategy1_Mtau*7})',
        os.path.join(args.out, 'strategy1_velocity_comparison.png'),
        metrics1['rmse_dq']
    )
    
    plot_reconstruction_comparison(
        tau[args.sample_idx], tau_rec1,
        'Torque', 'Nm',
        f'Strategy 1 (Latent: {args.strategy1_Mq*7 + args.strategy1_Mdq*7 + args.strategy1_Mtau*7})',
        os.path.join(args.out, 'strategy1_torque_comparison.png'),
        metrics1['rmse_tau']
    )
    
    # ========================================================================
    # Strategy 2: 1k budget
    # ========================================================================
    print("\n" + "="*80)
    print("Strategy 2: 1k Budget")
    print("="*80)
    print(f"Configuration:")
    print(f"  M: q={args.strategy2_Mq}, dq={args.strategy2_Mdq}, tau={args.strategy2_Mtau}")
    print(f"  Degree: q={args.strategy2_deg_q}, dq={args.strategy2_deg_dq}, tau={args.strategy2_deg_tau}")
    print(f"  Lambda: q={args.strategy2_lam_q}, dq={args.strategy2_lam_dq}, tau={args.strategy2_lam_tau}")
    print(f"  Latent dims: {args.strategy2_Mq*7 + args.strategy2_Mdq*7 + args.strategy2_Mtau*7}")
    
    q_rec2, dq_rec2, tau_rec2, metrics2 = compress_reconstruct(
        q, dq, tau,
        args.strategy2_Mq, args.strategy2_Mdq, args.strategy2_Mtau,
        args.strategy2_deg_q, args.strategy2_deg_dq, args.strategy2_deg_tau,
        args.strategy2_lam_q, args.strategy2_lam_dq, args.strategy2_lam_tau,
        args.sample_idx
    )
    
    print(f"\nReconstruction RMSE:")
    print(f"  Position (q):   {metrics2['rmse_q']:.6f}")
    print(f"  Velocity (dq):  {metrics2['rmse_dq']:.6f}")
    print(f"  Torque (tau):   {metrics2['rmse_tau']:.6f}")
    
    # Plot Strategy 2 results
    print(f"\n[Plotting] Generating Strategy 2 visualizations...")
    
    plot_reconstruction_comparison(
        q[args.sample_idx], q_rec2,
        'Position', 'rad',
        f'Strategy 2 (Latent: {args.strategy2_Mq*7 + args.strategy2_Mdq*7 + args.strategy2_Mtau*7})',
        os.path.join(args.out, 'strategy2_position_comparison.png'),
        metrics2['rmse_q']
    )
    
    plot_reconstruction_comparison(
        dq[args.sample_idx], dq_rec2,
        'Velocity', 'rad/s',
        f'Strategy 2 (Latent: {args.strategy2_Mq*7 + args.strategy2_Mdq*7 + args.strategy2_Mtau*7})',
        os.path.join(args.out, 'strategy2_velocity_comparison.png'),
        metrics2['rmse_dq']
    )
    
    plot_reconstruction_comparison(
        tau[args.sample_idx], tau_rec2,
        'Torque', 'Nm',
        f'Strategy 2 (Latent: {args.strategy2_Mq*7 + args.strategy2_Mdq*7 + args.strategy2_Mtau*7})',
        os.path.join(args.out, 'strategy2_torque_comparison.png'),
        metrics2['rmse_tau']
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("Visualization Complete!")
    print("="*80)
    print(f"\nGenerated 6 comparison plots:")
    print(f"  - {args.out}/strategy1_position_comparison.png")
    print(f"  - {args.out}/strategy1_velocity_comparison.png")
    print(f"  - {args.out}/strategy1_torque_comparison.png")
    print(f"  - {args.out}/strategy2_position_comparison.png")
    print(f"  - {args.out}/strategy2_velocity_comparison.png")
    print(f"  - {args.out}/strategy2_torque_comparison.png")
    
    print(f"\nComparison Summary:")
    print(f"{'Strategy':<12} {'Position RMSE':<15} {'Velocity RMSE':<15} {'Torque RMSE':<15} {'Latent Dims':<12}")
    print("-"*80)
    print(f"{'Strategy 1':<12} {metrics1['rmse_q']:<15.6f} {metrics1['rmse_dq']:<15.6f} "
          f"{metrics1['rmse_tau']:<15.6f} {args.strategy1_Mq*7 + args.strategy1_Mdq*7 + args.strategy1_Mtau*7:<12}")
    print(f"{'Strategy 2':<12} {metrics2['rmse_q']:<15.6f} {metrics2['rmse_dq']:<15.6f} "
          f"{metrics2['rmse_tau']:<15.6f} {args.strategy2_Mq*7 + args.strategy2_Mdq*7 + args.strategy2_Mtau*7:<12}")
    print("="*80)


if __name__ == '__main__':
    main()

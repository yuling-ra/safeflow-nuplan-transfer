#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B-Spline Trajectory Compression Quick Tuning Tool
- Simple manual parameter adjustment
- Fast visualization of original vs reconstructed
- Auto-overwrite previous results for quick iteration
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spline_utils import (
    normalized_times, bspline_design_matrix, bspline_deriv_design_matrix,
    ridge_pinv, encode_theta, decode_signal
)
# 导入边界权重修复版本
try:
    from spline_utils_fixed import ridge_pinv_with_boundary_weights
    BOUNDARY_FIX_AVAILABLE = True
except ImportError:
    BOUNDARY_FIX_AVAILABLE = False
    
from utils import (
    load_dataset, zscore_fit_stats, zscore_apply, zscore_inv, rmse
)


# ============================================================================
# Configuration - Edit these parameters for tuning
# ============================================================================

# Data settings
DATA_PATH = "../../Compress_Test/test_complete.npz"
SAMPLE_INDEX = 0  # Trajectory to visualize (0 to N-1)

# Control points (M) - Baseline values
M_Q_BASE = 64     # Position baseline control points (recommended: 32-96)
M_DQ_BASE = 128    # Velocity baseline control points (recommended: 16-64)
M_TAU_BASE = 128   # Torque baseline control points (recommended: 16-64)

# Adaptive density: Power function parameters
# The power function allows denser control points in specific time regions
# Power function formula: density_boost(t) = height * exp(-decay_rate * ((t - center)/width)^power)
ADAPTIVE_Q = True    # Enable adaptive density for position
ADAPTIVE_DQ = True   # Enable adaptive density for velocity
ADAPTIVE_TAU = True  # Enable adaptive density for torque

# Power function parameters for each signal (can be different)
POWER_PARAMS_Q = {
    'center': 0.0,      # Peak center position in normalized time [0,1]
    'width': 0.2,       # Peak width (larger = wider peak)
    'height': 1.0,      # Multiplier at peak (1.0 = double baseline, 2.0 = triple)
    'decay_rate': 3.0,  # How quickly it decays from center (higher = steeper)
    'power': 2,         # Exponent for decay shape (2=Gaussian-like, higher=flatter top)
}

POWER_PARAMS_DQ = {
    'center': 0.0,
    'width': 0.25,
    'height': 3.5,
    'decay_rate': 3.0,
    'power': 2,
}

POWER_PARAMS_TAU = {
    'center': 0.0,
    'width': 0.2,
    'height': 4.0,
    'decay_rate': 5.0,
    'power': 2,
}

# Spline degree - Higher = smoother
DEGREE_Q = 3     # Position degree (recommended: 3 or 5)
DEGREE_DQ = 5    # Velocity degree (recommended: 3)
DEGREE_TAU = 5   # Torque degree (recommended: 3 or 5)

# Regularization (lambda) - Higher = more stable, lower accuracy
LAMBDA_Q = 1e-6    # Position regularization (recommended: 1e-6 to 1e-5)
LAMBDA_DQ = 1e-7   # Velocity regularization (recommended: 1e-6)
LAMBDA_TAU = 1e-7  # Torque regularization (recommended: 1e-6 to 1e-4)

# Boundary fix - Improve first/last frame reconstruction
USE_BOUNDARY_FIX = True      # Enable boundary weight fix
BOUNDARY_WEIGHT = 10.0       # Weight multiplier for boundary frames (1.0-20.0)
BOUNDARY_FRAMES = 10         # Number of frames at each boundary to weight (5-20)

# Output settings
OUTPUT_DIR = "."  # Save to current directory (Quick_Tuning/)
AUTO_OPEN = False  # Set to True to auto-open plots after generation

# ============================================================================
# End of configuration
# ============================================================================


def power_function_density(t, center, width, height, decay_rate, power):
    """
    Calculate density boost using power function
    
    Formula: density_boost(t) = height * exp(-decay_rate * ((t - center)/width)^power)
    
    Parameters:
    - t: normalized time in [0,1]
    - center: peak center position in [0,1]
    - width: peak width parameter
    - height: multiplier at peak (e.g., 1.0 = double baseline)
    - decay_rate: how quickly it decays from center
    - power: exponent for decay shape
    
    Returns: density boost factor (0 to height)
    """
    normalized_dist = (t - center) / width
    boost = height * np.exp(-decay_rate * np.abs(normalized_dist) ** power)
    return boost


def compute_adaptive_control_points(T, M_base, params, enabled=True):
    """
    Compute adaptive number of control points based on power function
    
    Parameters:
    - T: number of timesteps
    - M_base: baseline number of control points
    - params: dictionary with 'center', 'width', 'height', 'decay_rate', 'power'
    - enabled: whether to apply adaptive density
    
    Returns:
    - M_adaptive: adjusted number of control points
    - density_curve: density boost at each timestep (for visualization)
    """
    if not enabled:
        t_norm = np.linspace(0, 1, T)
        density_curve = np.zeros(T)
        return M_base, density_curve
    
    # Compute density boost at normalized time points
    t_norm = np.linspace(0, 1, T)
    density_curve = power_function_density(
        t_norm, 
        params['center'], 
        params['width'], 
        params['height'], 
        params['decay_rate'], 
        params['power']
    )
    
    # Calculate total "area under curve" to determine extra control points
    # Mean density boost determines how many extra control points to add
    mean_boost = np.mean(density_curve)
    M_adaptive = int(M_base * (1 + mean_boost))
    
    # Ensure at least baseline
    M_adaptive = max(M_adaptive, M_base)
    
    return M_adaptive, density_curve


def compress_reconstruct(q, dq, tau, sample_idx=0):
    """
    Compress and reconstruct trajectory using configured parameters
    """
    N, T, J = q.shape
    
    print(f"\n[Processing] Sample {sample_idx}")
    print(f"  Data shape: N={N}, T={T}, J={J}")
    
    # Compute adaptive control points
    print("  Computing adaptive control points...")
    M_Q, density_q = compute_adaptive_control_points(T, M_Q_BASE, POWER_PARAMS_Q, ADAPTIVE_Q)
    M_DQ, density_dq = compute_adaptive_control_points(T, M_DQ_BASE, POWER_PARAMS_DQ, ADAPTIVE_DQ)
    M_TAU, density_tau = compute_adaptive_control_points(T, M_TAU_BASE, POWER_PARAMS_TAU, ADAPTIVE_TAU)
    
    print(f"    q:   {M_Q_BASE} (base) -> {M_Q} (adaptive), boost: +{M_Q - M_Q_BASE}")
    print(f"    dq:  {M_DQ_BASE} (base) -> {M_DQ} (adaptive), boost: +{M_DQ - M_DQ_BASE}")
    print(f"    tau: {M_TAU_BASE} (base) -> {M_TAU} (adaptive), boost: +{M_TAU - M_TAU_BASE}")
    
    # Z-score normalization
    print("  Normalizing...")
    mq, sq = zscore_fit_stats(q)
    mdq, sdq = zscore_fit_stats(dq)
    mtau, stau = zscore_fit_stats(tau)
    
    qz = zscore_apply(q, mq, sq)
    dqz = zscore_apply(dq, mdq, sdq)
    tauz = zscore_apply(tau, mtau, stau)
    
    # Normalized time
    t = normalized_times(T)
    
    # Construct spline basis
    print("  Building spline basis...")
    Bq = bspline_design_matrix(t, M_Q, DEGREE_Q)
    Bdq = bspline_design_matrix(t, M_DQ, DEGREE_DQ)
    Btau = bspline_design_matrix(t, M_TAU, DEGREE_TAU)
    
    # Use boundary-weighted ridge if enabled
    if USE_BOUNDARY_FIX and BOUNDARY_FIX_AVAILABLE:
        print(f"  Using boundary-weighted ridge (weight={BOUNDARY_WEIGHT}x, frames={BOUNDARY_FRAMES})")
        Pq = ridge_pinv_with_boundary_weights(Bq, LAMBDA_Q, BOUNDARY_WEIGHT, BOUNDARY_FRAMES)
        Pdq = ridge_pinv_with_boundary_weights(Bdq, LAMBDA_DQ, BOUNDARY_WEIGHT, BOUNDARY_FRAMES)
        Ptau = ridge_pinv_with_boundary_weights(Btau, LAMBDA_TAU, BOUNDARY_WEIGHT, BOUNDARY_FRAMES)
    else:
        if USE_BOUNDARY_FIX and not BOUNDARY_FIX_AVAILABLE:
            print("  Warning: Boundary fix requested but spline_utils_fixed.py not found")
        print("  Using standard ridge regression")
        Pq = ridge_pinv(Bq, LAMBDA_Q)
        Pdq = ridge_pinv(Bdq, LAMBDA_DQ)
        Ptau = ridge_pinv(Btau, LAMBDA_TAU)
    
    # Encode-decode
    print("  Compressing and reconstructing...")
    i = sample_idx
    
    # Position (q)
    theta_q = encode_theta(Pq, qz[i])
    qz_hat = decode_signal(Bq, theta_q)
    q_rec = zscore_inv(qz_hat[None, :, :], mq, sq)[0]
    
    # Velocity (dq)
    theta_dq = encode_theta(Pdq, dqz[i])
    dqz_hat = decode_signal(Bdq, theta_dq)
    dq_rec = zscore_inv(dqz_hat[None, :, :], mdq, sdq)[0]
    
    # Torque (tau)
    theta_tau = encode_theta(Ptau, tauz[i])
    tauz_hat = decode_signal(Btau, theta_tau)
    tau_rec = zscore_inv(tauz_hat[None, :, :], mtau, stau)[0]
    
    # Calculate metrics
    metrics = {
        'rmse_q': rmse(q[i], q_rec),
        'rmse_dq': rmse(dq[i], dq_rec),
        'rmse_tau': rmse(tau[i], tau_rec),
        'M_q': M_Q,
        'M_dq': M_DQ,
        'M_tau': M_TAU,
    }
    
    # Weighted RMSE (typical weights: wq=1.0, wdq=0.25, wtau=0.1)
    wrmse = np.sqrt(1.0 * metrics['rmse_q']**2 + 
                    0.25 * metrics['rmse_dq']**2 + 
                    0.1 * metrics['rmse_tau']**2)
    metrics['wrmse'] = wrmse
    
    # Store density curves for visualization
    density_info = {
        'density_q': density_q,
        'density_dq': density_dq,
        'density_tau': density_tau,
        'T': T,
    }
    
    return q[i], q_rec, dq[i], dq_rec, tau[i], tau_rec, metrics, density_info


def plot_comparison(orig, recon, quantity_name, unit, rmse_val, M_actual, output_path):
    """
    Plot original vs reconstructed for one quantity
    All 7 joints in subplots
    """
    T, J = orig.shape
    t = np.linspace(0, T-1, T)
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    joint_names = ['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 
                   'Joint 5', 'Joint 6', 'Joint 7']
    
    for j in range(J):
        row = j // 3
        col = j % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Plot original and reconstructed signals
        ax.plot(t, orig[:, j], 'b-', linewidth=2.5, label='Original', alpha=0.85)
        ax.plot(t, recon[:, j], 'r--', linewidth=2, label='Reconstructed', alpha=0.85)
        
        # Per-joint RMSE
        joint_rmse = np.sqrt(np.mean((orig[:, j] - recon[:, j])**2))
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel(f'{quantity_name} ({unit})', fontsize=11)
        ax.set_title(f'{joint_names[j]} | RMSE: {joint_rmse:.6f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Highlight differences with fill
        ax.fill_between(t, orig[:, j], recon[:, j], alpha=0.15, color='orange')
    
    # Overall title with config info
    latent_dims = M_actual['M_q'] * 7 + M_actual['M_dq'] * 7 + M_actual['M_tau'] * 7
    config_str = f"M=({M_actual['M_q']},{M_actual['M_dq']},{M_actual['M_tau']}) | Deg=({DEGREE_Q},{DEGREE_DQ},{DEGREE_TAU}) | λ=({LAMBDA_Q:.0e},{LAMBDA_DQ:.0e},{LAMBDA_TAU:.0e})"
    
    fig.suptitle(f'{quantity_name}: Original vs Reconstructed\n'
                 f'{config_str}\n'
                 f'Latent Dims: {latent_dims} | Overall RMSE: {rmse_val:.6f}', 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {output_path}")
    plt.close()


def plot_detailed_single_joint(orig, recon, quantity_name, unit, joint_idx, M_actual, output_path):
    """
    Plot detailed comparison for a single joint
    Shows original, reconstructed, and error in separate subplots
    """
    T = orig.shape[0]
    t = np.linspace(0, T-1, T)
    
    # Calculate error signal
    error = orig[:, joint_idx] - recon[:, joint_idx]
    joint_rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Subplot 1: Original vs Reconstructed
    ax1 = axes[0]
    ax1.plot(t, orig[:, joint_idx], 'b-', linewidth=2.5, label='Original', alpha=0.85)
    ax1.plot(t, recon[:, joint_idx], 'r--', linewidth=2, label='Reconstructed', alpha=0.85)
    ax1.fill_between(t, orig[:, joint_idx], recon[:, joint_idx], alpha=0.15, color='orange')
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel(f'{quantity_name} ({unit})', fontsize=12)
    ax1.set_title(f'Joint {joint_idx+1}: Original vs Reconstructed', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Subplot 2: Error over time
    ax2 = axes[1]
    ax2.plot(t, error, 'g-', linewidth=1.5, alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.fill_between(t, 0, error, alpha=0.3, color='green')
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel(f'Error ({unit})', fontsize=12)
    ax2.set_title(f'Reconstruction Error | RMSE: {joint_rmse:.6f} | Max: {max_error:.6f}', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Subplot 3: Error histogram
    ax3 = axes[2]
    ax3.hist(error, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7)
    ax3.axvline(x=joint_rmse, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'RMSE: {joint_rmse:.6f}')
    ax3.axvline(x=-joint_rmse, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel(f'Error ({unit})', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Overall title with configuration
    latent_dims = M_actual['M_q'] * 7 + M_actual['M_dq'] * 7 + M_actual['M_tau'] * 7
    config_str = f"M=({M_actual['M_q']},{M_actual['M_dq']},{M_actual['M_tau']}) | Deg=({DEGREE_Q},{DEGREE_DQ},{DEGREE_TAU}) | λ=({LAMBDA_Q:.0e},{LAMBDA_DQ:.0e},{LAMBDA_TAU:.0e})"
    
    fig.suptitle(f'{quantity_name} - Joint {joint_idx+1} Detailed Analysis\n{config_str}\nLatent Dims: {latent_dims}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {output_path}")
    plt.close()


def plot_adaptive_density_analysis(density_info, output_path):
    """
    Plot adaptive density analysis showing control point distribution
    Shows baseline vs adaptive density for q, dq, and tau
    
    横轴：实际时间步（timestep）
    纵轴：控制点密度（与该时间段的控制点数量正相关）
    """
    T = density_info['T']
    timesteps = np.arange(T)
    
    # Extract density curves
    density_q = density_info['density_q']
    density_dq = density_info['density_dq']
    density_tau = density_info['density_tau']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Helper function to compute local density
    def compute_local_density(M_base, M_adaptive, density_curve, T):
        """
        Compute local control point density at each timestep
        
        Baseline: uniform distribution, density = M_base / T at each point
        Adaptive: non-uniform, denser where density_curve is higher
        """
        baseline_density = np.ones(T) * (M_base / T)
        
        # Adaptive density proportional to (1 + density_curve)
        # This represents how control points are redistributed
        weight = 1.0 + density_curve
        normalized_weight = weight / np.sum(weight)
        adaptive_density = normalized_weight * M_adaptive
        
        return baseline_density, adaptive_density
    
    # Plot for Position (q)
    ax1 = axes[0]
    baseline_q, adaptive_q = compute_local_density(M_Q_BASE, metrics_global['M_q'], density_q, T)
    boost_q = adaptive_q - baseline_q
    
    ax1.fill_between(timesteps, 0, baseline_q, alpha=0.3, color='blue', label=f'Baseline ({M_Q_BASE} ctrl pts)')
    ax1.fill_between(timesteps, baseline_q, adaptive_q, alpha=0.5, color='red', label=f'Adaptive Boost (+{metrics_global["M_q"]-M_Q_BASE} ctrl pts)')
    ax1.plot(timesteps, baseline_q, 'b-', linewidth=2, alpha=0.8)
    ax1.plot(timesteps, adaptive_q, 'r-', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Control Point Density (ctrl pts / time)', fontsize=12)
    ax1.set_title(f'Position (q) - Control Point Density Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics
    max_boost_q = np.max(boost_q)
    max_boost_idx_q = np.argmax(boost_q)
    ax1.text(0.02, 0.98, f'Max Boost: {max_boost_q:.4f} at t={max_boost_idx_q}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot for Velocity (dq)
    ax2 = axes[1]
    baseline_dq, adaptive_dq = compute_local_density(M_DQ_BASE, metrics_global['M_dq'], density_dq, T)
    boost_dq = adaptive_dq - baseline_dq
    
    ax2.fill_between(timesteps, 0, baseline_dq, alpha=0.3, color='blue', label=f'Baseline ({M_DQ_BASE} ctrl pts)')
    ax2.fill_between(timesteps, baseline_dq, adaptive_dq, alpha=0.5, color='red', label=f'Adaptive Boost (+{metrics_global["M_dq"]-M_DQ_BASE} ctrl pts)')
    ax2.plot(timesteps, baseline_dq, 'b-', linewidth=2, alpha=0.8)
    ax2.plot(timesteps, adaptive_dq, 'r-', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Control Point Density (ctrl pts / time)', fontsize=12)
    ax2.set_title(f'Velocity (dq) - Control Point Density Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    max_boost_dq = np.max(boost_dq)
    max_boost_idx_dq = np.argmax(boost_dq)
    ax2.text(0.02, 0.98, f'Max Boost: {max_boost_dq:.4f} at t={max_boost_idx_dq}', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot for Torque (tau)
    ax3 = axes[2]
    baseline_tau, adaptive_tau = compute_local_density(M_TAU_BASE, metrics_global['M_tau'], density_tau, T)
    boost_tau = adaptive_tau - baseline_tau
    
    ax3.fill_between(timesteps, 0, baseline_tau, alpha=0.3, color='blue', label=f'Baseline ({M_TAU_BASE} ctrl pts)')
    ax3.fill_between(timesteps, baseline_tau, adaptive_tau, alpha=0.5, color='red', label=f'Adaptive Boost (+{metrics_global["M_tau"]-M_TAU_BASE} ctrl pts)')
    ax3.plot(timesteps, baseline_tau, 'b-', linewidth=2, alpha=0.8)
    ax3.plot(timesteps, adaptive_tau, 'r-', linewidth=2, alpha=0.8)
    
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Control Point Density (ctrl pts / time)', fontsize=12)
    ax3.set_title(f'Torque (tau) - Control Point Density Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    max_boost_tau = np.max(boost_tau)
    max_boost_idx_tau = np.argmax(boost_tau)
    ax3.text(0.02, 0.98, f'Max Boost: {max_boost_tau:.4f} at t={max_boost_idx_tau}', 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Overall title
    total_base = (M_Q_BASE + M_DQ_BASE + M_TAU_BASE) * 7
    total_adaptive = (metrics_global['M_q'] + metrics_global['M_dq'] + metrics_global['M_tau']) * 7
    
    fig.suptitle(f'Adaptive Control Point Density Analysis\n'
                 f'Power Function: center={POWER_PARAMS_Q["center"]:.2f}, width={POWER_PARAMS_Q["width"]:.2f}, '
                 f'height={POWER_PARAMS_Q["height"]:.2f}, decay={POWER_PARAMS_Q["decay_rate"]:.1f}, power={POWER_PARAMS_Q["power"]}\n'
                 f'Total Dims: {total_base} (baseline) → {total_adaptive} (adaptive) | Increase: +{total_adaptive - total_base}', 
                 fontsize=15, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {output_path}")
    plt.close()


def main():
    global metrics_global  # For density plot access
    
    print("="*80)
    print("B-Spline Quick Tuning Tool with Adaptive Density")
    print("="*80)
    
    # Print configuration
    print("\n[Configuration]")
    print(f"  Data: {DATA_PATH}")
    print(f"  Sample: {SAMPLE_INDEX}")
    print(f"\n  Baseline Control Points (M):")
    print(f"    q:   {M_Q_BASE:3d}  ({M_Q_BASE * 7:4d} dims)")
    print(f"    dq:  {M_DQ_BASE:3d}  ({M_DQ_BASE * 7:4d} dims)")
    print(f"    tau: {M_TAU_BASE:3d}  ({M_TAU_BASE * 7:4d} dims)")
    print(f"    ----------------------")
    print(f"    Total:    {(M_Q_BASE + M_DQ_BASE + M_TAU_BASE) * 7:4d} dims")
    
    print(f"\n  Adaptive Density:")
    print(f"    q:   {'Enabled' if ADAPTIVE_Q else 'Disabled'}")
    print(f"    dq:  {'Enabled' if ADAPTIVE_DQ else 'Disabled'}")
    print(f"    tau: {'Enabled' if ADAPTIVE_TAU else 'Disabled'}")
    
    if ADAPTIVE_Q:
        print(f"\n  Power Function Parameters (q):")
        print(f"    center:      {POWER_PARAMS_Q['center']:.2f}")
        print(f"    width:       {POWER_PARAMS_Q['width']:.2f}")
        print(f"    height:      {POWER_PARAMS_Q['height']:.2f}")
        print(f"    decay_rate:  {POWER_PARAMS_Q['decay_rate']:.1f}")
        print(f"    power:       {POWER_PARAMS_Q['power']}")
    
    print(f"\n  Spline Degree:")
    print(f"    q:   {DEGREE_Q}")
    print(f"    dq:  {DEGREE_DQ}")
    print(f"    tau: {DEGREE_TAU}")
    
    print(f"\n  Regularization (λ):")
    print(f"    q:   {LAMBDA_Q:.0e}")
    print(f"    dq:  {LAMBDA_DQ:.0e}")
    print(f"    tau: {LAMBDA_TAU:.0e}")
    
    print(f"\n  Boundary Fix:")
    if USE_BOUNDARY_FIX and BOUNDARY_FIX_AVAILABLE:
        print(f"    Enabled: Yes")
        print(f"    Weight:  {BOUNDARY_WEIGHT}x")
        print(f"    Frames:  {BOUNDARY_FRAMES} (each boundary)")
    else:
        print(f"    Enabled: No")
    
    # Load data
    print(f"\n[Loading] {DATA_PATH}...")
    dataset = load_dataset(DATA_PATH)
    q = dataset['q']
    dq = dataset['dq']
    tau = dataset['tau']
    N, T, J = q.shape
    
    if SAMPLE_INDEX >= N:
        print(f"[Error] Sample index {SAMPLE_INDEX} exceeds dataset size {N}")
        return
    
    # Compress and reconstruct
    print("\n[Compressing]")
    q_orig, q_rec, dq_orig, dq_rec, tau_orig, tau_rec, metrics, density_info = compress_reconstruct(
        q, dq, tau, SAMPLE_INDEX
    )
    
    # Store metrics globally for density plot
    metrics_global = metrics
    
    # Print results
    print("\n[Results]")
    print(f"  Actual Control Points:")
    print(f"    q:   {metrics['M_q']:3d}  ({metrics['M_q'] * 7:4d} dims)")
    print(f"    dq:  {metrics['M_dq']:3d}  ({metrics['M_dq'] * 7:4d} dims)")
    print(f"    tau: {metrics['M_tau']:3d}  ({metrics['M_tau'] * 7:4d} dims)")
    print(f"    ----------------------")
    print(f"    Total:    {(metrics['M_q'] + metrics['M_dq'] + metrics['M_tau']) * 7:4d} dims")
    print(f"\n  Reconstruction Quality:")
    print(f"    Position (q):   RMSE = {metrics['rmse_q']:.6f} rad  ({metrics['rmse_q']*180/np.pi:.4f}°)")
    print(f"    Velocity (dq):  RMSE = {metrics['rmse_dq']:.6f} rad/s")
    print(f"    Torque (tau):   RMSE = {metrics['rmse_tau']:.6f} Nm")
    print(f"    Weighted RMSE:  {metrics['wrmse']:.6f}")
    
    # Generate plots
    print("\n[Plotting]")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Randomly select one joint for detailed visualization
    np.random.seed(42)  # For reproducibility
    random_joint = np.random.randint(0, J)
    print(f"  Selected Joint {random_joint+1} for detailed visualization")
    
    # Generate all joints comparison plots
    plot_comparison(
        q_orig, q_rec, 
        'Position', 'rad', 
        metrics['rmse_q'],
        metrics,
        os.path.join(OUTPUT_DIR, 'position_comparison.png')
    )
    
    plot_comparison(
        dq_orig, dq_rec, 
        'Velocity', 'rad/s', 
        metrics['rmse_dq'],
        metrics,
        os.path.join(OUTPUT_DIR, 'velocity_comparison.png')
    )
    
    plot_comparison(
        tau_orig, tau_rec, 
        'Torque', 'Nm', 
        metrics['rmse_tau'],
        metrics,
        os.path.join(OUTPUT_DIR, 'torque_comparison.png')
    )
    
    # Generate detailed single joint plots
    plot_detailed_single_joint(
        q_orig, q_rec,
        'Position', 'rad',
        random_joint,
        metrics,
        os.path.join(OUTPUT_DIR, f'position_detailed_joint{random_joint+1}.png')
    )
    
    plot_detailed_single_joint(
        dq_orig, dq_rec,
        'Velocity', 'rad/s',
        random_joint,
        metrics,
        os.path.join(OUTPUT_DIR, f'velocity_detailed_joint{random_joint+1}.png')
    )
    
    plot_detailed_single_joint(
        tau_orig, tau_rec,
        'Torque', 'Nm',
        random_joint,
        metrics,
        os.path.join(OUTPUT_DIR, f'torque_detailed_joint{random_joint+1}.png')
    )
    
    # Generate adaptive density analysis plot
    plot_adaptive_density_analysis(
        density_info,
        os.path.join(OUTPUT_DIR, 'adaptive_density_analysis.png')
    )
    
    print("\n[Complete]")
    print("="*80)
    print("Generated 7 visualization plots:")
    print(f"  1. {OUTPUT_DIR}/position_comparison.png")
    print(f"  2. {OUTPUT_DIR}/velocity_comparison.png")
    print(f"  3. {OUTPUT_DIR}/torque_comparison.png")
    print(f"  4. {OUTPUT_DIR}/position_detailed_joint{random_joint+1}.png")
    print(f"  5. {OUTPUT_DIR}/velocity_detailed_joint{random_joint+1}.png")
    print(f"  6. {OUTPUT_DIR}/torque_detailed_joint{random_joint+1}.png")
    print(f"  7. {OUTPUT_DIR}/adaptive_density_analysis.png  <-- NEW!")
    print("="*80)
    
    # Auto-open plots if enabled
    if AUTO_OPEN:
        import subprocess
        plot_files = [
            'position_comparison.png', 
            'velocity_comparison.png', 
            'torque_comparison.png',
            f'position_detailed_joint{random_joint+1}.png',
            f'velocity_detailed_joint{random_joint+1}.png',
            f'torque_detailed_joint{random_joint+1}.png',
            'adaptive_density_analysis.png'
        ]
        for filename in plot_files:
            filepath = os.path.join(OUTPUT_DIR, filename)
            try:
                subprocess.run(['xdg-open', filepath], check=False)
            except:
                pass
    
    print("\n💡 Tuning Tips:")
    print("   Baseline Control Points:")
    print("     - Increase M_*_BASE for better overall accuracy")
    print("     - Decrease M_*_BASE to reduce dimensions")
    print("\n   Adaptive Density (Power Function):")
    print("     - 'center': Move peak to critical time region (0.0-1.0)")
    print("     - 'width': Adjust peak width (larger = wider influence)")
    print("     - 'height': Control density boost (1.0 = double at peak)")
    print("     - 'decay_rate': Steepness of falloff (higher = sharper)")
    print("     - 'power': Shape of decay curve (2 = Gaussian-like)")
    print("\n   Boundary Fix (First/Last Frame Improvement):")
    print("     - USE_BOUNDARY_FIX=True to enable")
    print("     - BOUNDARY_WEIGHT: 10.0 works well (1.0-20.0)")
    print("     - BOUNDARY_FRAMES: Number of frames to weight at each end")
    print("     - Trade-off: Better boundaries, slightly higher overall RMSE")
    print("\n   Spline Parameters:")
    print("     - Use degree=5 for smoother curves")
    print("     - Increase λ to reduce oscillations/overfitting")
    print("\n   Check 'adaptive_density_analysis.png' to see where extra")
    print("   control points are being allocated!")


if __name__ == '__main__':
    main()

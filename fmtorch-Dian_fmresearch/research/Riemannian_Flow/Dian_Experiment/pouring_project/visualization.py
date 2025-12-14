import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
import sys
from copy import deepcopy
import traceback

# Ensure the script can find sibling modules (preprocess.py)
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from scipy.spatial.transform import Rotation as R
from preprocess import se3_log

# ==============================================================================
# Numerical Error Evaluation
# ==============================================================================

def rotation_geodesic_distance(R1, R2):
    """计算两个旋转矩阵之间的测地距离（度数）"""
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def evaluate_reconstruction(T_gt, T_recon, method_name):
    """评估重构误差"""
    num_frames = T_gt.shape[0]
    rot_errors, trans_errors = [], []

    for k in range(num_frames):
        rot_err = rotation_geodesic_distance(T_gt[k, :3, :3], T_recon[k, :3, :3])
        trans_err = np.linalg.norm(T_gt[k, :3, 3] - T_recon[k, :3, 3])
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)
    
    rot_errors, trans_errors = np.array(rot_errors), np.array(trans_errors)
    
    T_end_error = np.linalg.inv(T_recon[-1]) @ T_gt[-1]
    xi_end = se3_log(T_end_error)
    
    metrics = {
        'method': method_name,
        'rotation': {'mean_deg': rot_errors.mean(), 'max_deg': rot_errors.max(), 'p95_deg': np.percentile(rot_errors, 95)},
        'translation': {'mean_m': trans_errors.mean(), 'max_m': trans_errors.max(), 'p95_m': np.percentile(trans_errors, 95)},
        'endpoint': {'rot_rad': np.linalg.norm(xi_end[:3]), 'trans_m': np.linalg.norm(xi_end[3:])}
    }
    return metrics, rot_errors, trans_errors

# ==============================================================================
# Matplotlib Visualizations
# ==============================================================================

def plot_reconstruction_error(rot_err_abs, trans_err_abs, rot_err_delta, trans_err_delta, save_path: Path):
    """绘制预处理重构误差曲线和分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rotation error
    axes[0, 0].plot(rot_err_abs, label='R9+t3', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(rot_err_delta, label='ΔSE(3)', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('Rotation Error (degrees)')
    axes[0, 0].set_title('Rotation Error Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Translation error
    axes[0, 1].plot(trans_err_abs * 100, label='R9+t3', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(trans_err_delta * 100, label='ΔSE(3)', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Translation Error (cm)')
    axes[0, 1].set_title('Translation Error Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Rotation error distribution
    axes[1, 0].hist(rot_err_abs, bins=50, alpha=0.5, label='R9+t3', edgecolor='black')
    axes[1, 0].hist(rot_err_delta, bins=50, alpha=0.5, label='ΔSE(3)', edgecolor='black')
    axes[1, 0].set_xlabel('Rotation Error (degrees)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Rotation Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Translation error distribution
    axes[1, 1].hist(trans_err_abs * 100, bins=50, alpha=0.5, label='R9+t3', edgecolor='black')
    axes[1, 1].hist(trans_err_delta * 100, bins=50, alpha=0.5, label='ΔSE(3)', edgecolor='black')
    axes[1, 1].set_xlabel('Translation Error (cm)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Translation Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstruction error plot saved to: {save_path}")

def plot_comprehensive_trajectories(T_seq_raw, T_recon_abs, T_recon_delta_abs, rot_err_abs, trans_err_abs, rot_err_delta, trans_err_delta, save_path: Path):
    """绘制轨迹、分量和误差的综合对比图"""
    fig = plt.figure(figsize=(20, 12))
    
    # === Translation Trajectories ===
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 1, 3], 'k-', label='Ground Truth', linewidth=2)
    ax1.plot(T_recon_abs[:, 0, 3], T_recon_abs[:, 1, 3], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax1.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 1, 3], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax1.set_title('XY Plane Trajectory'); ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axis('equal')

    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 2, 3], 'k-', label='Ground Truth', linewidth=2)
    ax2.plot(T_recon_abs[:, 0, 3], T_recon_abs[:, 2, 3], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax2.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 2, 3], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax2.set_title('XZ Plane Trajectory'); ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axis('equal')

    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(T_seq_raw[:, 1, 3], T_seq_raw[:, 2, 3], 'k-', label='Ground Truth', linewidth=2)
    ax3.plot(T_recon_abs[:, 1, 3], T_recon_abs[:, 2, 3], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax3.plot(T_recon_delta_abs[:, 1, 3], T_recon_delta_abs[:, 2, 3], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax3.set_title('YZ Plane Trajectory'); ax3.set_xlabel('Y (m)'); ax3.set_ylabel('Z (m)'); ax3.legend(); ax3.grid(True, alpha=0.3); ax3.axis('equal')

    ax4 = fig.add_subplot(3, 4, 4, projection='3d')
    ax4.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 1, 3], T_seq_raw[:, 2, 3], 'k-', label='Ground Truth', linewidth=2)
    ax4.plot(T_recon_abs[:, 0, 3], T_recon_abs[:, 1, 3], T_recon_abs[:, 2, 3], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax4.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 1, 3], T_recon_delta_abs[:, 2, 3], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax4.set_title('3D Trajectory'); ax4.set_xlabel('X'); ax4.set_ylabel('Y'); ax4.set_zlabel('Z'); ax4.legend()

    # === Translation Components Over Time ===
    frames = np.arange(len(T_seq_raw))
    for i, (label, comp) in enumerate(zip(['X', 'Y', 'Z'], [0, 1, 2])):
        ax = fig.add_subplot(3, 4, 5 + i)
        ax.plot(frames, T_seq_raw[:, comp, 3], 'k-', label='Ground Truth', linewidth=2)
        ax.plot(frames, T_recon_abs[:, comp, 3], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
        ax.plot(frames, T_recon_delta_abs[:, comp, 3], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
        ax.set_title(f'{label} Translation Over Time'); ax.set_xlabel('Frame'); ax.set_ylabel(f'{label} (m)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(3, 4, 8)
    ax8.plot(frames, trans_err_abs * 100, 'r-', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax8.plot(frames, trans_err_delta * 100, 'b-', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax8.set_title('Translation Error Magnitude'); ax8.set_xlabel('Frame'); ax8.set_ylabel('Error (cm)'); ax8.legend(); ax8.grid(True, alpha=0.3)
    
    def extract_euler_angles(T_seq):
        """Extract ZYX Euler angles from SE(3) sequence"""
        angles = []
        for T in T_seq:
            r = R.from_matrix(T[:3, :3])
            # Use 'xyz' for roll, pitch, yaw
            angles.append(r.as_euler('xyz', degrees=True))
        return np.array(angles)

    euler_gt = extract_euler_angles(T_seq_raw)
    euler_abs = extract_euler_angles(T_recon_abs)
    euler_delta = extract_euler_angles(T_recon_delta_abs)

    for i, (label, comp) in enumerate(zip(['Roll', 'Pitch', 'Yaw'], [0, 1, 2])): # XYZ order
        ax = fig.add_subplot(3, 4, 9 + i)
        ax.plot(frames, euler_gt[:, comp], 'k-', label='Ground Truth', linewidth=2)
        ax.plot(frames, euler_abs[:, comp], 'r--', label='R9+t3', linewidth=1.5, alpha=0.7)
        ax.plot(frames, euler_delta[:, comp], 'b:', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
        ax.set_title(f'{label} Angle Over Time'); ax.set_xlabel('Frame'); ax.set_ylabel(f'{label} (deg)'); ax.legend(); ax.grid(True, alpha=0.3)

    ax12 = fig.add_subplot(3, 4, 12)
    ax12.plot(frames, rot_err_abs, 'r-', label='R9+t3', linewidth=1.5, alpha=0.7)
    ax12.plot(frames, rot_err_delta, 'b-', label='ΔSE(3)', linewidth=1.5, alpha=0.7)
    ax12.set_title('Rotation Geodesic Error'); ax12.set_xlabel('Frame'); ax12.set_ylabel('Error (deg)'); ax12.legend(); ax12.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Comprehensive trajectory plot saved to: {save_path}")

def plot_sweep_results(results_summary, sweep_results, save_path: Path):
    """可视化sweep实验结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Best loss distribution
    best_losses = [r['best_loss'] for r in results_summary]
    axes[0, 0].bar(range(len(best_losses)), best_losses, alpha=0.7)
    axes[0, 0].set_title('Sweep Best Loss Distribution'); axes[0, 0].set_xlabel('Config Index (sorted)'); axes[0, 0].set_ylabel('Best Loss'); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_yscale('log')

    # Hidden dimension vs loss
    hidden_dims = [r['hidden_dim'] for r in results_summary]
    scatter = axes[0, 1].scatter(hidden_dims, best_losses, c=range(len(hidden_dims)), cmap='viridis', alpha=0.6, s=100)
    axes[0, 1].set_title('Hidden Dimension vs Loss'); axes[0, 1].set_xlabel('Hidden Dimension'); axes[0, 1].set_ylabel('Best Loss'); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_yscale('log')
    plt.colorbar(scatter, ax=axes[0, 1], label='Rank')

    # Learning rate vs loss
    lrs = [r['lr'] for r in results_summary]
    schedulers = [r['scheduler'] for r in results_summary]
    for sched in set(schedulers):
        mask = [s == sched for s in schedulers]
        axes[1, 0].scatter([lrs[i] for i in range(len(lrs)) if mask[i]], [best_losses[i] for i in range(len(best_losses)) if mask[i]], label=sched, alpha=0.7, s=100)
    axes[1, 0].set_title('LR vs Loss'); axes[1, 0].set_xlabel('Learning Rate'); axes[1, 0].set_ylabel('Best Loss'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_xscale('log'); axes[1, 0].set_yscale('log')

    # Training curves for top 5
    for i in range(min(5, len(results_summary))):
        name = results_summary[i]['name']
        for r in sweep_results:
            if r['name'] == name:
                loss_history = r['loss_history']
                window = 50
                smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid') if len(loss_history) > window else loss_history
                axes[1, 1].plot(smoothed, label=f"{i+1}. {name}", alpha=0.7, linewidth=1.5)
                break
    axes[1, 1].set_title('Top 5 Training Curves'); axes[1, 1].set_xlabel('Training Steps'); axes[1, 1].set_ylabel('Loss (smoothed)'); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Sweep results plot saved to: {save_path}")

def plot_fm_vs_recon_error(rot_err_delta, trans_err_delta, rot_err_fm, trans_err_fm, save_path: Path):
    """对比预处理重构与FM生成的误差"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(rot_err_delta, label='Preprocessing Recon', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(rot_err_fm, label='FM Generation', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('Rotation Error Comparison'); axes[0, 0].set_xlabel('Frame Index'); axes[0, 0].set_ylabel('Error (degrees)'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_yscale('log')

    axes[0, 1].plot(trans_err_delta * 100, label='Preprocessing Recon', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(trans_err_fm * 100, label='FM Generation', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_title('Translation Error Comparison'); axes[0, 1].set_xlabel('Frame Index'); axes[0, 1].set_ylabel('Error (cm)'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_yscale('log')

    axes[1, 0].hist(rot_err_delta, bins=50, alpha=0.5, label='Preprocessing Recon', edgecolor='black')
    axes[1, 0].hist(rot_err_fm, bins=50, alpha=0.5, label='FM Generation', edgecolor='black')
    axes[1, 0].set_title('Rotation Error Distribution'); axes[1, 0].set_xlabel('Error (degrees)'); axes[1, 0].set_ylabel('Frequency'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_xscale('log')

    axes[1, 1].hist(trans_err_delta * 100, bins=50, alpha=0.5, label='Preprocessing Recon', edgecolor='black')
    axes[1, 1].hist(trans_err_fm * 100, bins=50, alpha=0.5, label='FM Generation', edgecolor='black')
    axes[1, 1].set_title('Translation Error Distribution'); axes[1, 1].set_xlabel('Error (cm)'); axes[1, 1].set_ylabel('Frequency'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3); axes[1, 1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"FM vs Recon error plot saved to: {save_path}")

def plot_final_trajectory_comparison(T_seq_raw, T_recon_delta_abs, T_fm_abs, save_path: Path):
    """绘制最终的GT, 重构, FM生成轨迹对比图"""
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131)
    ax1.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 1, 3], 'k-', label='Ground Truth', linewidth=2)
    ax1.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 1, 3], 'b--', label='Preprocessing Recon', linewidth=1.5, alpha=0.7)
    ax1.plot(T_fm_abs[:, 0, 3], T_fm_abs[:, 1, 3], 'r:', label='FM Generation', linewidth=1.5, alpha=0.7)
    ax1.set_title('XY Plane Trajectory'); ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axis('equal')

    ax2 = fig.add_subplot(132)
    ax2.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 2, 3], 'k-', label='Ground Truth', linewidth=2)
    ax2.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 2, 3], 'b--', label='Preprocessing Recon', linewidth=1.5, alpha=0.7)
    ax2.plot(T_fm_abs[:, 0, 3], T_fm_abs[:, 2, 3], 'r:', label='FM Generation', linewidth=1.5, alpha=0.7)
    ax2.set_title('XZ Plane Trajectory'); ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)'); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.axis('equal')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(T_seq_raw[:, 0, 3], T_seq_raw[:, 1, 3], T_seq_raw[:, 2, 3], 'k-', label='Ground Truth', linewidth=2)
    ax3.plot(T_recon_delta_abs[:, 0, 3], T_recon_delta_abs[:, 1, 3], T_recon_delta_abs[:, 2, 3], 'b--', label='Preprocessing Recon', linewidth=1.5, alpha=0.7)
    ax3.plot(T_fm_abs[:, 0, 3], T_fm_abs[:, 1, 3], T_fm_abs[:, 2, 3], 'r:', label='FM Generation', linewidth=1.5, alpha=0.7)
    ax3.set_title('3D Trajectory Comparison'); ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z'); ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Final trajectory comparison plot saved to: {save_path}")

# ==============================================================================
# Lightweight Matplotlib Visualizations
# ==============================================================================

def plot_trajectory_stick(T_seq, save_path: Path, skip_size=5, stick_length=0.1, n_segments=10, color_start=(1,0,0,0.8), color_end=(0,0,1,0.8)):
    """
    使用matplotlib将轨迹可视化为一系列彩色的3D“火柴棒”。

    Args:
        T_seq (np.ndarray): (N, 4, 4) 的SE(3)轨迹序列。
        save_path (Path): 图像保存路径。
        skip_size (int): 每隔多少帧绘制一个火柴棒。
        stick_length (float): 火柴棒的长度。
        n_segments (int): 用于渲染颜色渐变的线段数。
        color_start (tuple): 起始颜色 (R, G, B, A)。
        color_end (tuple): 结束颜色 (R, G, B, A)。
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    positions = T_seq[:, :3, 3]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'k-', alpha=0.2, label='Trajectory Path')

    for i in range(0, len(T_seq), skip_size):
        T = T_seq[i]
        pos = T[:3, 3]
        z_axis = T[:3, 2]  # 使用局部坐标系的Z轴作为火柴棒方向

        p1 = pos - stick_length / 2 * z_axis
        p2 = pos + stick_length / 2 * z_axis

        points_x = np.linspace(p1[0], p2[0], n_segments)
        points_y = np.linspace(p1[1], p2[1], n_segments)
        points_z = np.linspace(p1[2], p2[2], n_segments)
        
        colors = [np.array(color_start) * (1 - t) + np.array(color_end) * t for t in np.linspace(0, 1, n_segments - 1)]

        for j in range(n_segments - 1):
            ax.plot(points_x[j:j+2], points_y[j:j+2], points_z[j:j+2], color=colors[j], linewidth=3)
    
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('Stick Visualization of SE(3) Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Stick visualization saved to: {save_path}")

def plot_trajectory_simple(T_seq, save_path: Path):
    """
    轨迹的简化可视化，分别绘制3D位置轨迹和3D旋转变化。

    Args:
        T_seq (np.ndarray): (N, 4, 4) 的SE(3)轨迹序列。
        save_path (Path): 图像保存路径。
    """
    fig = plt.figure(figsize=(16, 7))

    # --- 1. 3D位置轨迹 ---
    ax1 = fig.add_subplot(121, projection='3d')
    pos = T_seq[:, :3, 3]
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='Translation')
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='red', s=100, marker='x', label='End')
    ax1.set_title('Translation Trajectory (XYZ)')
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.axis('equal')

    # --- 2. 3D旋转变化 (帧 vs. 欧拉角) ---
    ax2 = fig.add_subplot(122, projection='3d')
    try:
        angles = []
        for T in T_seq:
            angles.append(R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True))
        euler_angles = np.array(angles)
        
        frames = np.arange(len(T_seq))
        roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        
        points = np.array([frames, roll, pitch]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import Line3DCollection
        from matplotlib.colors import Normalize
        norm = Normalize(yaw.min(), yaw.max())
        lc = Line3DCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(yaw)
        lc.set_linewidth(2)
        line = ax2.add_collection3d(lc)
        
        cbar = fig.colorbar(line, ax=ax2, pad=0.1)
        cbar.set_label('Yaw (degrees)')
        
        ax2.set_xlim(frames.min(), frames.max())
        ax2.set_ylim(roll.min(), roll.max())
        ax2.set_zlim(pitch.min(), pitch.max())

    except Exception as e:
        ax2.text(0.5, 0.5, f"无法绘制旋转: {e}", ha='center', va='center')

    ax2.set_title('Rotation Over Time')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Roll (degrees)')
    ax2.set_zlabel('Pitch (degrees)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Simple trajectory visualization saved to: {save_path}")

# ==============================================================================
# Open3D Visualization
# ==============================================================================

VIS_UTILS_AVAILABLE = False
try:
    module_path = Path(__file__).resolve().parent.parent.parent / 'MMLfD-Tutorial_Modified'
    if str(module_path) not in sys.path:
        sys.path.append(str(module_path))
    from vis_utils.open3d_utils import get_mesh_bottle, get_mesh_mug
    import open3d as o3d
    VIS_UTILS_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 'vis_utils' 或 'open3d'。3D可视化功能将被禁用。")
    traceback.print_exc(limit=1)

def visualize_3d_trajectories(trajectories, labels, colors, raw_data_dict, skip_size=1, save_path=None):
    """使用Open3D进行3D轨迹可视化"""
    if not VIS_UTILS_AVAILABLE:
        print("\nOpen3D或vis_utils不可用，正在保存轨迹数据以供离线可视化...")
        data_to_save = {label.replace(' ', '_'): traj for label, traj in zip(labels, trajectories)}
        data_to_save.update({k: v for k, v in raw_data_dict.items() if k in ['bottle_idx', 'mug_idx', 'offset']})
        np.savez(save_path, **data_to_save)
        print(f"✓ 数据已保存到: {save_path}")
        return

    print("初始化3D可视化...")
    
    try:
        model_root_dir = Path(__file__).resolve().parent.parent.parent / 'MMLfD-Tutorial_Modified'
        mesh_bottle = get_mesh_bottle(root=str(model_root_dir / '3dmodels/bottles'), bottle_idx=raw_data_dict['bottle_idx'])
        mesh_mug = get_mesh_mug(root=str(model_root_dir / '3dmodels/mugs'), mug_idx=raw_data_dict['mug_idx'])
        mesh_bottle.compute_vertex_normals()
        mesh_mug.compute_vertex_normals()
    except Exception as e:
        print(f"错误: 加载3D模型失败: {e}")
        return
    
    geometries = []
    mesh_table = o3d.geometry.TriangleMesh.create_box(width=1.5, height=1.5, depth=0.02)
    mesh_table.translate([-0.75, -0.75, -0.02]).paint_uniform_color([0.8, 0.6, 0.4])
    mesh_table.compute_vertex_normals()
    geometries.append(mesh_table)
    
    mesh_mug.paint_uniform_color([0.5, 0.5, 0.5])
    geometries.append(mesh_mug)

    for traj, label, color in zip(trajectories, labels, colors):
        for t_idx in range(0, len(traj), skip_size):
            bottle_instance = deepcopy(mesh_bottle)
            bottle_instance.transform(traj[t_idx]).paint_uniform_color(color)
            geometries.append(bottle_instance)
            
    o3d.visualization.draw_geometries(geometries, window_name=f"轨迹对比: {', '.join(labels)}", width=1280, height=720)
    print("可视化窗口已关闭。")

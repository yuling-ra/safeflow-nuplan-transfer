import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import argparse
import zipfile
from scipy.spatial.transform import Rotation as R

# --- Local Imports ---
# Assuming this script is in the same directory as the others
from model import EnhancedVectorFieldNet
from preprocess import reconstruct_incremental_pose, align_to_first_frame, se3_log

# ==============================================================================
# Helper functions (copied from main_67_firsttry.py for simplicity)
# ==============================================================================

def quaternion_to_matrix(quat):
    """Converts a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    # Scipy expects [x, y, z, w]
    return R.from_quat(quat).as_matrix()

def load_trajectory_from_json_in_zip(zip_path, json_file):
    """Loads a single trajectory from a JSON file inside a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file) as f:
            data = json.load(f)
    trajectory_raw = data['trajectory']
    T_seq = np.zeros((len(trajectory_raw), 4, 4))
    for i, frame in enumerate(trajectory_raw):
        T_seq[i, :3, 3] = frame[:3]
        # The data is in [w, x, y, z], convert to [x, y, z, w] for scipy
        quat_wxyz = frame[3:]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        T_seq[i, :3, :3] = quaternion_to_matrix(quat_xyzw)
        T_seq[i, 3, 3] = 1.0
    return T_seq, {'traj': T_seq}

def get_json_files_from_zip(zip_path):
    """Lists all .json files in a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        return [name for name in z.namelist() if name.endswith('.json')]

# ==============================================================================
# Inference and Visualization
# ==============================================================================

@torch.no_grad()
def generate_trajectories(model, config, device, num_trajectories=3, num_steps=100):
    """Generates trajectories by solving the ODE from noise."""
    print(f"--- Generating {num_trajectories} trajectories ---")
    model.eval()
    
    seq_len = config['seq_len']
    data_dim = config['data_dim']
    
    # Start with random noise at t=0
    x_t = torch.randn((num_trajectories, seq_len, data_dim), device=device)
    
    # Time steps for integration
    ts = torch.linspace(0, 1, num_steps, device=device)
    
    for i in range(num_steps - 1):
        t_current = ts[i]
        t_next = ts[i+1]
        dt = t_next - t_current
        
        t_tensor = torch.full((num_trajectories,), t_current, device=device)
        v = model(x_t, t_tensor)
        x_t = x_t + v * dt
        
        if (i + 1) % 20 == 0:
            print(f"  Integration step {i+1}/{num_steps-1}")

    print("--- Generation complete ---")
    return x_t.cpu().numpy()

def load_real_trajectory_T_seq(config, return_start_poses=False):
    """Loads a single real trajectory from the dataset for comparison."""
    print("--- Loading one real trajectory from dataset ---")
    script_dir = Path(__file__).parent
    zip_path = script_dir / 'dataset/gestor_pouring_dataset_6.zip'
    json_files = get_json_files_from_zip(zip_path)
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {zip_path}")
        
    T_seq_raw, _ = load_trajectory_from_json_in_zip(zip_path, json_files[0])
    T_seq_aligned, T_0 = align_to_first_frame(T_seq_raw)
    
    seq_len = config['seq_len']
    # Adjust trajectory length to match model's expected sequence length
    if T_seq_aligned.shape[0] > seq_len:
        T_seq_aligned = T_seq_aligned[:seq_len]
    elif T_seq_aligned.shape[0] < seq_len:
        # Note: Z_delta will be shorter by 1, so we pad to seq_len for T_seq
        # The generated Z_delta will have length seq_len - 1
        last_frame = T_seq_aligned[-1]
        padding = np.tile(last_frame, (seq_len - T_seq_aligned.shape[0], 1, 1))
        T_seq_aligned = np.vstack([T_seq_aligned, padding])
    
    if return_start_poses:
        # T_seq_aligned_start should be the first frame of the aligned trajectory
        T_seq_aligned_start = T_seq_aligned[0]
        return T_seq_aligned, T_0, T_seq_aligned_start
        
    return T_seq_aligned

def visualize_trajectories_as_sticks(trajectories_T, colors, skip=3, stick_length=0.05):
    """
    Visualizes trajectories as a sequence of sticks in a 3D matplotlib plot.
    Each stick has a base color and a lighter tip color to show orientation.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for traj_idx, T_seq in enumerate(trajectories_T):
        main_color, tip_color = colors[traj_idx]
        
        for step in range(0, len(T_seq), skip):
            T = T_seq[step]
            p_base_local = np.array([0, 0, 0, 1])
            p_top_local = np.array([0, 0, stick_length, 1])
            
            p_base_world = T @ p_base_local
            p_top_world = T @ p_top_local
            
            ax.plot(
                [p_base_world[0], p_top_world[0]],
                [p_base_world[1], p_top_world[1]],
                [p_base_world[2], p_top_world[2]],
                color=main_color,
                linewidth=1.5
            )
            ax.scatter(
                p_top_world[0], p_top_world[1], p_top_world[2],
                color=tip_color,
                s=10,
                depthshade=True
            )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Generated and Real Trajectories')
    ax.view_init(elev=20., azim=-45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.3, 0.3]); ax.set_ylim([-0.3, 0.3]); ax.set_zlim([0, 0.6])
    plt.show()

def visualize_analysis_plots(all_deltas, labels):
    """
    Visualizes the SO(3) and XYZ deltas for multiple trajectories.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True)
    fig.suptitle('Trajectory Delta Analysis (SO3-ω and XYZ-v)', fontsize=16)
    
    colors = ['#FF0000'] + ['#0000FF', '#4169E1', '#ADD8E6']
    linestyles = ['-', '--', ':', '-.']

    # Titles for each subplot
    titles_omega = ['SO(3) delta (ω_x)', 'SO(3) delta (ω_y)', 'SO(3) delta (ω_z)']
    titles_v = ['XYZ delta (v_x)', 'XYZ delta (v_y)', 'XYZ delta (v_z)']

    for i, (delta_seq, label) in enumerate(zip(all_deltas, labels)):
        timesteps = range(delta_seq.shape[0])
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # Plot omega (SO3 delta)
        for j in range(3):
            axes[0, j].plot(timesteps, delta_seq[:, j], label=label, color=color, linestyle=linestyle)
            axes[0, j].set_title(titles_omega[j])
            axes[0, j].grid(True, linestyle='--', alpha=0.6)

        # Plot v (XYZ delta)
        for j in range(3):
            axes[1, j].plot(timesteps, delta_seq[:, j+3], label=label, color=color, linestyle=linestyle)
            axes[1, j].set_title(titles_v[j])
            axes[1, j].grid(True, linestyle='--', alpha=0.6)
            axes[1, j].set_xlabel('Timestep')

    # Add legends
    for ax_row in axes:
        for ax in ax_row:
            ax.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main(args):
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config_path = run_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = EnhancedVectorFieldNet(**{
        k: v for k, v in config.items() if k in [
            'seq_len', 'data_dim', 'hidden_dim', 'num_layers', 
            'time_embed_dim', 'dropout', 'num_heads', 'reduction', 
            'window', 'fourier_k', 'bridge_k', 'bridge_q'
        ]
    })
    
    model_path = run_dir / 'best_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model and configuration loaded.")

    # --- Load pre-computed stats for data reconstruction ---
    stats_path = run_dir / '05_whiten_delta.json'
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    print("Loaded data statistics for reconstruction.")

    # --- Load a real trajectory to get a valid starting pose ---
    real_trajectory_T, T_0, T_seq_aligned_start = load_real_trajectory_T_seq(config, return_start_poses=True)

    # --- Generate Trajectories ---
    generated_trajectories_delta = generate_trajectories(model, config, device, num_trajectories=3)

    if args.mode == 'analysis':
        print("--- Preparing data for analysis plots ---")
        # 1. Get real deltas from the real trajectory
        num_frames = real_trajectory_T.shape[0]
        real_delta_xi = np.array([se3_log(np.linalg.inv(real_trajectory_T[k-1]) @ real_trajectory_T[k]) for k in range(1, num_frames)])

        # 2. Get generated deltas by "un-whitening" the model output
        mu_omega = np.array(stats['mu_omega'])
        sigma_omega = np.array(stats['sigma_omega'])
        mu_v = np.array(stats['mu_v'])
        sigma_v = np.array(stats['sigma_v'])
        
        generated_deltas_xi = []
        for z_delta in generated_trajectories_delta:
            # Truncate generated sequence to match the length of the real delta sequence
            z_delta_truncated = z_delta[:real_delta_xi.shape[0], :]
            omega_recon = z_delta_truncated[:, :3] * sigma_omega + mu_omega
            v_recon = z_delta_truncated[:, 3:] * sigma_v + mu_v
            xi_recon = np.concatenate([omega_recon, v_recon], axis=1)
            generated_deltas_xi.append(xi_recon)
        
        labels = ['Real Trajectory (Dataset)'] + [f'Generated Trajectory {i+1}' for i in range(len(generated_deltas_xi))]
        all_deltas = [real_delta_xi] + generated_deltas_xi
        
        visualize_analysis_plots(all_deltas, labels=labels)

    elif args.mode == 'sticks':
        print("--- Reconstructing generated trajectories to pose matrices ---")
        reconstructed_generated_T = []
        num_deltas = real_trajectory_T.shape[0] - 1
        for z_delta in generated_trajectories_delta:
            z_delta_truncated = z_delta[:num_deltas, :]
            T_seq = reconstruct_incremental_pose(
                z_delta_truncated, 
                stats, 
                T_0, 
                T_seq_aligned_start
            )
            reconstructed_generated_T.append(T_seq)

        all_trajectories = reconstructed_generated_T + [real_trajectory_T]
        
        colors_generated = [
            ('#0000FF', '#ADD8E6'),
            ('#00008B', '#87CEFA'),
            ('#4169E1', '#B0E0E6')
        ]
        color_real = ('#FF0000', '#FFC0CB')
        all_colors = colors_generated + [color_real]
        
        print("--- Launching visualization ---")
        visualize_trajectories_as_sticks(all_trajectories, colors=all_colors, skip=args.skip)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and visualize pouring trajectories.")
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the training run directory")
    parser.add_argument('--skip', type=int, default=3, help="Frames to skip for 'sticks' visualization.")
    parser.add_argument('--mode', type=str, default='sticks', choices=['sticks', 'analysis'], help="Visualization mode.")
    args = parser.parse_args()
    main(args)

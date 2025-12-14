import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from pathlib import Path
import argparse
import zipfile
from scipy.spatial.transform import Rotation as R
from torchdiffeq import odeint

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
def generate_trajectories(model, config, device, conditions):
    """Generates trajectories by solving the ODE from noise using dopri5 (torchdiffeq)."""
    num_trajectories = conditions.shape[0]
    print(f"--- Generating {num_trajectories} trajectories from {num_trajectories} conditions (dopri5) ---")
    model.eval()

    seq_len = config['seq_len']
    data_dim = config['data_dim']

    x0 = torch.randn((num_trajectories, seq_len, data_dim), device=device)

    def ode_func(t, x):
        # x shape: (B, L, C)
        t_tensor = torch.full((num_trajectories,), t.item(), device=device)
        return model(x, t_tensor, condition=conditions)

    t_span = torch.tensor([0.0, 1.0], device=device)
    sol = odeint(ode_func, x0, t_span, method='dopri5', rtol=1e-5, atol=1e-5, options={'max_num_steps': 1000})
    x_T = sol[-1]

    print("--- Generation complete ---")
    return x_T.cpu().numpy()

def load_real_trajectories_T_seq(config, num_to_load=2):
    """Loads multiple real trajectories from the dataset for comparison."""
    print(f"--- Loading {num_to_load} real trajectories from dataset ---")
    script_dir = Path(__file__).parent
    zip_path = script_dir / 'dataset/gestor_pouring_dataset_6.zip'
    json_files = get_json_files_from_zip(zip_path)

    if len(json_files) < num_to_load:
        raise ValueError(f"Not enough JSON files in {zip_path} to load {num_to_load} trajectories.")

    trajectories = []
    T_0_list = []
    T_seq_aligned_start_list = []

    for i in range(num_to_load):
        T_seq_raw, _ = load_trajectory_from_json_in_zip(zip_path, json_files[i])
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

        T_seq_aligned_start = T_seq_aligned[0]
        
        trajectories.append(T_seq_aligned)
        T_0_list.append(T_0)
        T_seq_aligned_start_list.append(T_seq_aligned_start)

    return trajectories, T_0_list, T_seq_aligned_start_list

def visualize_trajectories_as_paths(trajectories_T, colors, skip=1):
    """
    Visualizes trajectories as 3D paths.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    for traj_idx, T_seq in enumerate(trajectories_T):
        main_color, _ = colors[traj_idx]  # Use only the main color
        positions = T_seq[::skip, :3, 3]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=main_color, alpha=0.8)

        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color=main_color, marker='o', s=50)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color=main_color, marker='x', s=50)
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Generated and Real Trajectories (Paths)')
    ax.view_init(elev=20., azim=-45)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-0.3, 0.3]); ax.set_ylim([-0.3, 0.3]); ax.set_zlim([0, 0.6])
    plt.show()

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
            'window', 'fourier_k', 'bridge_k', 'bridge_q', 'cond_dim'
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

    # --- Load real trajectories to get valid starting poses ---
    num_real = args.num_real
    num_generated_per_real = args.num_generated

    real_trajectories_T, T_0_list, T_seq_aligned_start_list = load_real_trajectories_T_seq(
        config, num_to_load=num_real
    )
    
    # --- Create condition tensor from the initial poses ---
    cond_6d_list = [se3_log(T_0) for T_0 in T_0_list]
    conditions_np = np.array(cond_6d_list)
    expanded_conditions_np = np.repeat(conditions_np, num_generated_per_real, axis=0)
    conditions = torch.from_numpy(expanded_conditions_np).float().to(device)
    
    total_generated = conditions.shape[0]
    print(f"Using {num_real} initial poses to generate {num_generated_per_real} trajs each (total {total_generated}).")


    # --- Generate Trajectories ---
    generated_trajectories_delta = generate_trajectories(model, config, device, conditions)

    # --- Reconstructing generated trajectories to pose matrices ---
    print("--- Reconstructing generated trajectories to pose matrices ---")
    expanded_T_0_list = [item for item in T_0_list for _ in range(num_generated_per_real)]
    expanded_T_seq_aligned_start_list = [item for item in T_seq_aligned_start_list for _ in range(num_generated_per_real)]

    reconstructed_generated_T = []
    for i in range(generated_trajectories_delta.shape[0]):
        T_seq = reconstruct_incremental_pose(
            generated_trajectories_delta[i], 
            stats, 
            expanded_T_0_list[i], 
            expanded_T_seq_aligned_start_list[i]
        )
        reconstructed_generated_T.append(T_seq)

    # --- Prepare trajectories and colors for visualization ---
    all_trajectories = []
    all_colors = []

    # Transform real trajectories back to the world frame for correct comparison
    world_frame_real_T = [T_0 @ T_seq_aligned for T_0, T_seq_aligned in zip(T_0_list, real_trajectories_T)]

    # Define a color palette for generated trajectories
    colors_generated_options = [
        ('#0000FF', '#ADD8E6'),  # Blue
        ('#008000', '#90EE90'),  # Green
        ('#FFA500', '#FFE4B5'),  # Orange
        ('#9400D3', '#E6E6FA'),  # Violet
        ('#00CED1', '#E0FFFF'),  # Dark Turquoise
    ]
    color_real = ('#FF0000', '#FFC0CB') # Red

    gen_traj_idx = 0
    for i in range(num_real):
        # Add generated trajectories for this condition
        gen_color = colors_generated_options[i % len(colors_generated_options)]
        for j in range(num_generated_per_real):
            all_trajectories.append(reconstructed_generated_T[gen_traj_idx])
            all_colors.append(gen_color)
            gen_traj_idx += 1
        
        # Add the corresponding real trajectory (now in world frame)
        all_trajectories.append(world_frame_real_T[i])
        all_colors.append(color_real)
    
    print("--- Launching visualization ---")
    if args.vis_mode == 'stick':
        visualize_trajectories_as_sticks(all_trajectories, colors=all_colors, skip=args.skip)
    elif args.vis_mode == 'path':
        visualize_trajectories_as_paths(all_trajectories, colors=all_colors, skip=args.skip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and visualize pouring trajectories.")
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the training run directory")
    parser.add_argument('--num_real', type=int, default=1, help="Number of real trajectories to load for conditions.")
    parser.add_argument('--num_generated', type=int, default=1, help="Number of trajectories to generate per real trajectory.")
    parser.add_argument('--vis_mode', type=str, default='stick', choices=['stick', 'path'], help="Visualization mode ('stick' or 'path').")
    parser.add_argument('--skip', type=int, default=6, help="Frames to skip for visualization.")
    args = parser.parse_args()
    main(args)

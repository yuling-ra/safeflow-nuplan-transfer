import json
import random
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import matplotlib.pyplot as plt

def load_trajectory_as_SE3(file_path: Path) -> np.ndarray:
    """Loads a trajectory from a JSON file and converts it to a sequence of SE(3) matrices."""
    with file_path.open('r') as f:
        data = json.load(f)
    T_seq = []
    trajectory_data = data.get('trajectory', [])
    for point in trajectory_data:
        if len(point) == 7:
            pos, quat = point[:3], point[3:]
            # Ensure quaternion is not zero-norm
            norm = np.linalg.norm(quat)
            if norm < 1e-6: continue
            quat_norm = np.array(quat) / norm
            
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat_norm).as_matrix()
            T[:3, 3] = pos
            T_seq.append(T)
    if not T_seq:
        raise ValueError(f"No valid trajectory data found in {file_path}")
    return np.array(T_seq)

def plot_interactive_sticks(trajectories: list, labels: list, skip_size=5, stick_length=0.1):
    """
    Interactively visualizes multiple trajectories as 3D sticks in a single plot.
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a set of distinct color pairs
    colors = [('red', 'maroon'), ('blue', 'navy'), ('green', 'darkgreen'), ('purple', 'indigo'), ('orange', 'saddlebrown')]

    for i, (T_seq, label) in enumerate(zip(trajectories, labels)):
        path_color, stick_color = colors[i % len(colors)]
        
        # Plot the path of the trajectory's center point
        positions = T_seq[:, :3, 3]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=path_color, alpha=0.4, label=f'Path: {label}')

        # Plot sticks representing orientation at intervals
        for frame_idx in range(0, len(T_seq), skip_size):
            T = T_seq[frame_idx]
            pos = T[:3, 3]
            # Use local Z-axis (typically 'forward' or 'up' for end-effectors) for stick direction
            z_axis = T[:3, 2] 

            p1 = pos - stick_length / 2 * z_axis
            p2 = pos + stick_length / 2 * z_axis
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=stick_color, linewidth=2.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Interactive 3D Trajectory Visualization')
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    # Set equal aspect ratio for a more intuitive view
    all_positions = np.vstack([t[:, :3, 3] for t in trajectories])
    min_coords = all_positions.min(axis=0)
    max_coords = all_positions.max(axis=0)
    center = (max_coords + min_coords) / 2
    max_range = (max_coords - min_coords).max()
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

    print("Displaying interactive plot. Close the window to exit.")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Interactively visualize trajectories from the pouring dataset.")
    parser.add_argument('--dataset_version', type=str, choices=['6', '7'], default='7', help="Dataset version.")
    parser.add_argument('--num_trajectories', type=int, default=3, help="Number of random trajectories to visualize.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--skip_size', type=int, default=5, help="For visualization clarity, plot one stick every N frames. Use 1 for full density.")
    
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    dataset_root_dir = script_dir / 'dataset'
    dataset_dir = dataset_root_dir / f"gestor_pouring_dataset_{args.dataset_version}"

    if not dataset_dir.is_dir():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    json_files = list(dataset_dir.glob('sample_*.json'))
    if not json_files:
        print(f"Error: No .json files found in {dataset_dir}")
        return

    if len(json_files) < args.num_trajectories:
        print(f"Warning: Requested {args.num_trajectories}, found {len(json_files)}. Visualizing all.")
        num_to_visualize = len(json_files)
    else:
        num_to_visualize = args.num_trajectories
        
    selected_files = random.sample(json_files, num_to_visualize)
    print(f"Selected files: {[f.name for f in selected_files]}")

    trajectories = []
    labels = []
    for file_path in selected_files:
        try:
            print(f"  - Loading {file_path.name}...")
            T_sequence = load_trajectory_as_SE3(file_path)
            trajectories.append(T_sequence)
            labels.append(file_path.stem)
        except Exception as e:
            print(f"    Error processing {file_path.name}: {e}")
    
    if trajectories:
        plot_interactive_sticks(trajectories, labels, skip_size=args.skip_size)

if __name__ == '__main__':
    main()

import os
import json
import random
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

# Ensure the script can find sibling modules (visualization.py)
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

from visualization import plot_trajectory_stick, plot_trajectory_simple, plot_final_trajectory_comparison

def load_trajectory_as_SE3(file_path: Path) -> np.ndarray:
    """
    Loads a trajectory from a JSON file and converts it to a sequence of SE(3) matrices.

    Args:
        file_path (Path): Path to the .json file.

    Returns:
        np.ndarray: A numpy array of shape (N, 4, 4) representing the SE(3) trajectory.
    """
    with file_path.open('r') as f:
        data = json.load(f)
    
    T_seq = []
    trajectory_data = data.get('trajectory', [])
    
    for point in trajectory_data:
        if len(point) == 7:
            pos = point[:3]
            quat = point[3:]  # Assuming [qx, qy, qz, qw] format
            
            # Normalize the quaternion to ensure it's a valid rotation
            quat_norm = np.array(quat) / np.linalg.norm(quat)
            
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat_norm).as_matrix()
            T[:3, 3] = pos
            T_seq.append(T)
            
    if not T_seq:
        raise ValueError(f"No valid trajectory data found in {file_path}")
        
    return np.array(T_seq)

def main():
    """Main function to parse arguments and run the visualization."""
    parser = argparse.ArgumentParser(description="Visualize random trajectories from the pouring dataset.")
    
    parser.add_argument(
        '--dataset_version', type=str, choices=['6', '7'], default='6',
        help="Dataset version to use ('6' or '7'). Default is 6."
    )
    parser.add_argument(
        '--num_trajectories', type=int, default=3,
        help="Number of random trajectories to visualize. Default is 3."
    )
    parser.add_argument(
        '--vis_method', type=str, choices=['stick', 'simple'], default='stick',
        help="Visualization method to use ('stick' or 'simple'). Default is 'stick'."
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # --- Path Setup ---
    script_dir = Path(__file__).resolve().parent
    dataset_root_dir = script_dir / 'dataset'
    output_dir = script_dir / 'visualization_outputs'
    output_dir.mkdir(exist_ok=True)
    
    dataset_dir = dataset_root_dir / f"gestor_pouring_dataset_{args.dataset_version}"
    if not dataset_dir.is_dir():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return

    # --- File Selection ---
    json_files = list(dataset_dir.glob('sample_*.json'))
    if len(json_files) < args.num_trajectories:
        print(f"Warning: Requested {args.num_trajectories} trajectories, but only found {len(json_files)}. Visualizing all available.")
        num_to_visualize = len(json_files)
    else:
        num_to_visualize = args.num_trajectories
        
    selected_files = random.sample(json_files, num_to_visualize)
    print(f"Selected {len(selected_files)} files from dataset {args.dataset_version} for visualization...")

    # --- Visualization Loop ---
    vis_functions = {
        'stick': plot_trajectory_stick,
        'simple': plot_trajectory_simple,
    }
    vis_func = vis_functions[args.vis_method]

    for file_path in selected_files:
        try:
            print(f"  - Processing {file_path.name}...")
            T_sequence = load_trajectory_as_SE3(file_path)
            
            save_name = f"{file_path.stem}_{args.vis_method}.png"
            save_path = output_dir / save_name
            
            # Call the selected visualization function
            vis_func(T_sequence, save_path=save_path)
            
        except Exception as e:
            print(f"    Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc(limit=2)

    print(f"\nVisualizations saved in: {output_dir.relative_to(Path.cwd())}")

if __name__ == '__main__':
    main()

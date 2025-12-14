# verify_dataset.py
# -*- coding: utf-8 -*-
"""
Standalone script to verify and visualize OCP datasets.

This script performs several checks on a given .npz dataset file:
1.  Existence and readability of the file.
2.  Presence of all required data keys.
3.  Consistency of data shapes (e.g., states_steps = control_steps + 1).
4.  Checks for NaN or Inf values in states and actions.
5.  Visualizes a random sample of trajectories, including:
    - Trajectories plotted on the track map.
    - Distribution of velocities.
    - Distribution of control inputs (acceleration/steering).
    - Detailed view of a single trajectory's dynamics.

Example usage from project root:
python research/UniCon/raceline/verify_dataset.py \
    --dataset ./outputs/ocp_dataset_inverse_2a1131b3.npz \
    --map research/UniCon/raceline/nuerburgring_segment_map.npz \
    --samples 20
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from raceline_core import load_map, MapData, OCPConfig, simulate_vehicle_bc
except ImportError:
    print("Error: Could not import 'raceline_core'. Make sure you are running this script from the project root directory.")
    exit(1)


def check_dynamics(states: np.ndarray, actions: np.ndarray, map_data: MapData, config: OCPConfig, tolerance=0.1) -> bool:
    """Resimulate trajectory and check for consistency."""
    x0 = states[0]
    state_steps = states.shape[0]
    
    L_pix = config.wheelbase_m * map_data.resolution
    resim_states = simulate_vehicle_bc(
        x0, actions, config.dt, state_steps, L_pix, delta_rate_max=config.delta_rate_max
    )
    
    error = np.mean(np.abs(states[:, :2] - resim_states[:, :2]))

    if error > tolerance:
        print(f"    ❌ Dynamics check failed. Mean position error: {error:.4f} > tolerance {tolerance}")
        return False
    
    return True

def get_dataset_stats(data: np.lib.npyio.NpzFile) -> dict:
    """Calculate statistics for key variables in a dataset."""
    stats = {}
    actions = data["actions"]
    states = data["states"]
    
    stats['accel'] = {'mean': np.mean(actions[:, :, 0]), 'std': np.std(actions[:, :, 0]), 'min': np.min(actions[:, :, 0]), 'max': np.max(actions[:, :, 0])}
    stats['steer'] = {'mean': np.mean(actions[:, :, 1]), 'std': np.std(actions[:, :, 1]), 'min': np.min(actions[:, :, 1]), 'max': np.max(actions[:, :, 1])}
    stats['vel'] = {'mean': np.mean(states[:, :, 3]), 'std': np.std(states[:, :, 3]), 'min': np.min(states[:, :, 3]), 'max': np.max(states[:, :, 3])}
    
    return stats

def compare_stats(ref_stats: dict, new_stats: dict, tolerance_ratio=0.5) -> bool:
    """Compare stats dicts and print warnings for large deviations."""
    is_consistent = True
    for key in ref_stats:
        for metric in ['mean', 'std']:
            ref_val = ref_stats[key][metric]
            new_val = new_stats[key][metric]
            if abs(ref_val) > 1e-6 and abs((new_val - ref_val) / ref_val) > tolerance_ratio:
                print(f"    ⚠️  Warning: Large deviation in '{key}' {metric}. "
                      f"Ref: {ref_val:.3f}, Current: {new_val:.3f} (ratio > {tolerance_ratio})")
                is_consistent = False
    return is_consistent

def print_summary(paths: list[str], all_data: dict, total_trajectories: int):
    print("\n==========================================")
    print("✅ All Checks Passed: Comprehensive Dataset Summary")
    print("==========================================")
    print(f"Source files: {len(paths)}")
    for p in paths:
        print(f"  - {os.path.basename(p)}")
    print(f"Total trajectories: {total_trajectories}")
    print("\n--- Data Content ---")

    key_descriptions = {
        "states": "Vehicle states. Shape: (N, 101, 4). Dims: [x_pix, y_pix, theta_rad, velocity_pix_per_s]",
        "actions": "Control inputs. Shape: (N, 100, 2). Dims: [acceleration, steering_angle_rad]",
        "x0s": "Initial states. Shape: (N, 4). Same dims as states[0]",
        "starts": "Start points on reference line. Shape: (N, 2). Dims: [x_pix, y_pix]",
        "ends": "End points on reference line. Shape: (N, 2). Dims: [x_pix, y_pix]",
        "refs": "Reference trajectory segments. Shape: (N, 101, 2). Dims: [x_pix, y_pix]",
        "vmaxs": "Maximum velocity for the segment. Shape: (N,)"
    }

    for key, desc in key_descriptions.items():
        if key in all_data:
            d = all_data[key]
            print(f"- {key}:")
            print(f"  - Description: {desc}")
            print(f"  - Shape: {d.shape}")
            print(f"  - Dtype: {d.dtype}")
    
    print("\n--- Combined Statistics ---")
    combined_stats = get_dataset_stats(all_data)
    for key, stats in combined_stats.items():
        print(f"- {key.capitalize()}:")
        print(f"  - Mean: {stats['mean']:.3f}")
        print(f"  - Std Dev: {stats['std']:.3f}")
        print(f"  - Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print("==========================================\n")

def visualize_combined_plots(paths: list[str], map_data: MapData, n_samples_per_ds=20):
    print("\n--- Generating Combined Visualization ---")
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    fig.suptitle(f'Combined Dataset Validation - {len(paths)} sources', fontsize=18)
    colors = plt.cm.jet(np.linspace(0, 1, len(paths)))

    # Plot 1: Trajectories on map
    ax1 = axes[0, 0]
    ax1.imshow(map_data.grid_map, cmap='gray', origin='lower', alpha=0.6)
    ax1.plot(map_data.raceline_grid[:, 0], map_data.raceline_grid[:, 1], 'g-', linewidth=2, label='Reference Line')
    
    legend_elements = [Line2D([0], [0], color='g', lw=2, label='Reference Line')]

    all_velocities = []
    all_accel = []
    all_steering = []

    for i, path in enumerate(paths):
        with np.load(path) as data:
            n_traj = data['states'].shape[0]
            n_samples = min(n_samples_per_ds, n_traj)
            sample_indices = np.random.choice(n_traj, n_samples, replace=False)
            
            for idx in sample_indices:
                traj_states = data["states"][idx]
                ax1.plot(traj_states[:, 0], traj_states[:, 1], alpha=0.6, linewidth=1.0, color=colors[i])
            
            all_velocities.append(data["states"][..., 3].flatten())
            all_accel.append(data["actions"][..., 0].flatten())
            all_steering.append(data["actions"][..., 1].flatten())
        
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Src {i+1}: {os.path.basename(path)}'))

    ax1.legend(handles=legend_elements, fontsize=8)
    ax1.set_title('Trajectory Samples from All Datasets')
    ax1.set_xlabel('X (pixels)'); ax1.set_ylabel('Y (pixels)'); ax1.grid(True, alpha=0.3)

    # Plot 2: Velocity distributions
    ax2 = axes[0, 1]
    ax2.hist(all_velocities, bins=50, alpha=0.7, stacked=True, color=[c for c in colors], label=[os.path.basename(p) for p in paths])
    ax2.set_title('Velocity Distribution'); ax2.set_xlabel('Velocity (pixels/s)'); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Plot 3: Control distributions
    ax3 = axes[1, 0]
    for i in range(len(paths)):
        ax3.scatter(all_accel[i], all_steering[i], alpha=0.2, s=2, color=colors[i], label=os.path.basename(paths[i]))
    ax3.set_title('Control Input Distribution'); ax3.set_xlabel('Acceleration'); ax3.set_ylabel('Steering Angle')
    ax3.grid(True, alpha=0.3)
    ax3.legend(markerscale=4, fontsize=8)

    # Plot 4: Empty for now, or another plot
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Verification Complete', horizontalalignment='center', verticalalignment='center', fontsize=15)
    ax4.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f'dataset_combination_verification_{len(paths)}_sources.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Combined verification plot saved: {save_path}")
    plt.show()


def verify_and_combine_datasets(paths: list[str], output_path: str, map_file: str, full_check: bool):
    """
    Verify that multiple datasets can be combined and optionally combine them.
    """
    if not paths:
        print("No dataset paths provided.")
        return False

    print(f"=== Verifying {len(paths)} datasets for combination (Full Check: {full_check}) ===")
    
    map_data = load_map(map_file)
    config = OCPConfig()
    
    try:
        ref_data = np.load(paths[0], allow_pickle=False)
    except Exception as e:
        print(f"❌ Could not read reference dataset: {paths[0]} ({e})")
        return False

    required_keys = ["actions", "states", "x0s", "starts", "ends", "refs", "vmaxs"]
    missing_keys = [key for key in required_keys if key not in ref_data.keys()]
    if missing_keys:
        print(f"❌ Reference dataset missing keys: {missing_keys}")
        ref_data.close()
        return False

    ref_shapes = {key: ref_data[key].shape[1:] for key in required_keys}
    ref_dtypes = {key: ref_data[key].dtype for key in required_keys}
    ref_stats = get_dataset_stats(ref_data) if full_check else None
    
    total_trajectories = ref_data["actions"].shape[0]
    print(f"✅ Reference dataset: {os.path.basename(paths[0])} ({total_trajectories} trajectories)")
    ref_data.close()

    is_compatible = True
    all_stats = [ref_stats] if ref_stats else []
    for path in paths[1:]:
        try:
            data = np.load(path, allow_pickle=False)
            n_traj = data["actions"].shape[0]
            print(f"--- Verifying: {os.path.basename(path)} ({n_traj} trajectories) ---")

            if any(key not in data.keys() for key in required_keys):
                print(f"❌ Missing keys")
                is_compatible = False
            else:
                for key in required_keys:
                    if data[key].shape[1:] != ref_shapes[key] or data[key].dtype != ref_dtypes[key]:
                        print(f"❌ Shape or Dtype mismatch for '{key}'")
                        is_compatible = False
                
                if full_check and is_compatible:
                    if np.any(np.isnan(data["actions"])) or np.any(np.isnan(data["states"])):
                        print(f"❌ NaN values found in data.")
                        is_compatible = False
                    
                    n_dyn_checks = min(5, n_traj)
                    indices = np.random.choice(n_traj, n_dyn_checks, replace=False)
                    print(f"    Running {n_dyn_checks} dynamics checks...")
                    dyn_ok = all(check_dynamics(data["states"][i], data["actions"][i], map_data, config) for i in indices)
                    if not dyn_ok:
                        is_compatible = False
                    
                    current_stats = get_dataset_stats(data)
                    compare_stats(ref_stats, current_stats)
                    all_stats.append(current_stats)

            if not is_compatible:
                data.close()
                break
            
            print(f"✅ Compatible: {os.path.basename(path)}")
            total_trajectories += n_traj
            data.close()

        except Exception as e:
            print(f"❌ Error reading file {path}: {e}")
            is_compatible = False
            break

    if not is_compatible:
        print("\n❌ Incompatible datasets found. Combination aborted.")
        return False
    
    print("\n==========================================")
    print(f"✅ All {len(paths)} datasets are compatible.")
    print(f"   Total trajectories: {total_trajectories}")
    print("==========================================")
    
    all_data = None
    if full_check or output_path: # Need to load all data for summary or saving
        all_data_list = {key: [] for key in required_keys}
        for path in paths:
            with np.load(path) as data:
                for key in required_keys:
                    all_data_list[key].append(data[key])
        all_data = {key: np.concatenate(all_data_list[key], axis=0) for key in required_keys}

    if full_check:
        visualize_combined_plots(paths, map_data)
        print_summary(paths, all_data, total_trajectories)

    if output_path:
        print(f"\nCombining datasets into '{output_path}'...")
        try:
            np.savez_compressed(output_path, **all_data)
            print(f"✅ Successfully saved combined dataset to '{output_path}'")
        except Exception as e:
            print(f"❌ An error occurred during file combination: {e}")
            return False

    return True

def verify_dataset(path: str, n_samples: int = 20, map_file: str = "nuerburgring_segment_map.npz"):
    """
    Verify dataset integrity and visualize randomly sampled trajectories
    """
    print(f"\n=== Verifying dataset: {path} ===")
    
    # 1. Basic checks
    if not os.path.exists(path):
        print(f"❌ File does not exist: {path}")
        return False
    
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"❌ Cannot read file: {e}")
        return False
    
    # 2. Data integrity check
    required_keys = ["actions", "states", "x0s", "starts", "ends", "refs", "vmaxs"]
    missing_keys = [key for key in required_keys if key not in data.keys()]
    if missing_keys:
        print(f"❌ Missing required data keys: {missing_keys}")
        data.close()
        return False
    
    # 3. Shape consistency check
    n_traj = data["actions"].shape[0]
    control_steps = data["actions"].shape[1]
    state_steps = data["states"].shape[1]
    
    print(f"✅ Data keys check passed")
    print(f"✅ Number of trajectories: {n_traj}")
    print(f"✅ Control steps: {control_steps}, State steps: {state_steps}")
    
    if state_steps != control_steps + 1:
        print(f"❌ State steps should be control steps + 1")
        data.close()
        return False
    
    # 4. Value range check
    actions = data["actions"]
    states = data["states"]
    
    if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
        print(f"❌ Actions contain NaN or Inf values")
        data.close()
        return False
    
    if np.any(np.isnan(states)) or np.any(np.isinf(states)):
        print(f"❌ States contain NaN or Inf values")
        data.close()
        return False
    
    print(f"✅ Value range check passed")
    print(f"   Actions range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"   States range: [{states.min():.3f}, {states.max():.3f}]")
    
    # 5. Visualization of random samples
    try:
        map_data: MapData = load_map(map_file)
        
        n_samples = min(n_samples, n_traj)
        if n_samples == 0:
            print("No trajectories to visualize.")
            return True
            
        sample_indices = np.random.choice(n_traj, n_samples, replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dataset Validation - {n_samples} Random Trajectories from {os.path.basename(path)}', fontsize=16)
        
        ax1 = axes[0, 0]
        ax1.imshow(map_data.grid_map, cmap='gray', origin='lower', alpha=0.6)
        ax1.plot(map_data.raceline_grid[:, 0], map_data.raceline_grid[:, 1], 'g-', linewidth=2, label='Reference Line')
        
        for i, idx in enumerate(sample_indices[:10]):
            traj_states = data["states"][idx]
            p = ax1.plot(traj_states[:, 0], traj_states[:, 1], alpha=0.7, linewidth=1.5)
            line_color = p[0].get_color()
            ax1.plot(traj_states[0, 0], traj_states[0, 1], '^', color=line_color, markersize=8, alpha=0.7)
            ax1.plot(traj_states[-1, 0], traj_states[-1, 1], 'o', color=line_color, markersize=8, alpha=0.7)
        
        ax1.set_title('Trajectory Distribution on Map')
        ax1.set_xlabel('X (pixels)'); ax1.set_ylabel('Y (pixels)')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        all_velocities = data["states"][sample_indices, :, 3].flatten()
        ax2.hist(all_velocities, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Velocity Distribution'); ax2.set_xlabel('Velocity (pixels/s)'); ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        sample_actions = data["actions"][sample_indices]
        accel = sample_actions[:, :, 0].flatten()
        steering = sample_actions[:, :, 1].flatten()
        ax3.scatter(accel, steering, alpha=0.5, s=1)
        ax3.set_title('Control Input Distribution'); ax3.set_xlabel('Acceleration'); ax3.set_ylabel('Steering Angle')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        sample_idx = sample_indices[0]
        sample_traj = data["states"][sample_idx]
        sample_actions_single = data["actions"][sample_idx]
        time_steps = np.arange(len(sample_traj))
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(time_steps, sample_traj[:, 3], 'b-', label='Velocity', linewidth=2)
        line2 = ax4_twin.plot(time_steps[:-1], sample_actions_single[:, 0], 'r-', label='Acceleration', linewidth=2)
        line3 = ax4_twin.plot(time_steps[:-1], sample_actions_single[:, 1], 'g-', label='Steering', linewidth=2)
        
        ax4.set_xlabel('Time Steps'); ax4.set_ylabel('Velocity', color='b')
        ax4_twin.set_ylabel('Control Inputs'); ax4.set_title(f'Single Trajectory Example (Traj #{sample_idx})')
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right'); ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = path.replace('.npz', '_verification.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Verification plot saved: {save_path}")
        plt.show()
        
    except FileNotFoundError:
        print(f"❌ Error: Map file not found at '{map_file}'. Cannot generate visualizations.")
    except Exception as e:
        print(f"⚠️ An unexpected error occurred during visualization: {e}")
    
    finally:
        data.close()
    
    print(f"✅ Dataset verification complete!")
    return True

def main():
    ap = argparse.ArgumentParser(
        description="Standalone tool to verify/combine and visualize OCP trajectory datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
# Verify and visualize a single dataset
python research/UniCon/raceline/verify_dataset.py \\
    --datasets ./outputs/ocp_dataset_inverse_xxxx.npz

# Check compatibility of multiple datasets with a quick check
python research/UniCon/raceline/verify_dataset.py \\
    --datasets ./outputs/ocp_dataset_1.npz ./outputs/ocp_dataset_2.npz

# Run a full, in-depth check on multiple datasets
python research/UniCon/raceline/verify_dataset.py \\
    --datasets ./outputs/ocp_dataset_1.npz ./outputs/ocp_dataset_2.npz \\
    --full-check

# Combine multiple datasets into a new file after a full check
python research/UniCon/raceline/verify_dataset.py \\
    --datasets ./outputs/ocp_dataset_1.npz ./outputs/ocp_dataset_2.npz \\
    --output combined_dataset.npz --full-check
"""
    )
    ap.add_argument("--datasets", type=str, nargs='+', required=True, 
                    help="Path(s) to the .npz dataset file(s) to process.")
    ap.add_argument("--map", type=str, default="nuerburgring_segment_map.npz",
                    help="Path to the corresponding map file for visualization.")
    ap.add_argument("--samples", type=int, default=20,
                    help="Number of random trajectories to visualize for single file verification.")
    ap.add_argument("--output", type=str, default=None,
                    help="If provided, combines compatible datasets into this new .npz file.")
    ap.add_argument("--full-check", action="store_true",
                    help="Enable comprehensive checks (dynamics, stats, visualization) for multiple datasets.")
    args = ap.parse_args()

    if len(args.datasets) > 1:
        verify_and_combine_datasets(args.datasets, args.output, args.map, args.full_check)
    elif len(args.datasets) == 1:
        if args.output:
            print("Warning: --output is only used when multiple datasets are provided. "
                  "To copy a single dataset, use standard file system commands.")
        verify_dataset(args.datasets[0], args.samples, args.map)
    else:
        print("Error: No dataset files provided.")

if __name__ == "__main__":
    main() 
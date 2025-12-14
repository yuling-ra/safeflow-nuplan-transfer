# make_dataset.py
import os, time, argparse, uuid
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from raceline_core import (
    MapData, OCPConfig, load_map, build_sdf_from_grid,
    nearest_index, sample_in_box, resample_polyline_to_N,
    continuation_solve, unpack_controls, simulate_vehicle_bc
)

# Global variables for multiprocessing workers
md = None
cfg = None
control_steps = None
sdf_map = None
L_pix = None
START_A = None
START_B = None
END_A = None
END_B = None

def solve_one(job_seed: int):
    """Worker function defined in global scope to support pickle"""
    rng = np.random.default_rng(job_seed)
    start_sample = sample_in_box(START_A, START_B, rng)
    end_sample   = sample_in_box(END_A,   END_B,   rng)

    idx_s = nearest_index(md.raceline_grid, start_sample)
    idx_e = nearest_index(md.raceline_grid, end_sample)
    if idx_s == idx_e: idx_e = min(idx_s + 1, len(md.raceline_grid) - 1)
    sub_poly = md.raceline_grid[idx_s:idx_e+1] if idx_s < idx_e else md.raceline_grid[idx_e:idx_s+1][::-1]

    raceline_ref_seg, seg_len_px = resample_polyline_to_N(sub_poly, cfg.state_steps)
    raceline_ref_seg[-1] = end_sample
    dxy_seg = np.diff(raceline_ref_seg, axis=0, append=raceline_ref_seg[-1:])
    theta_ref_seg = np.arctan2(dxy_seg[:,1], dxy_seg[:,0])

    avg_speed_pix_seg = seg_len_px / (control_steps * cfg.dt + 1e-12)
    v_max_seg = 3.5 * avg_speed_pix_seg
    v_ref_seg = np.clip(np.linspace(0, 2.0 * avg_speed_pix_seg, cfg.state_steps), 0, v_max_seg)

    theta0_seg = np.arctan2(raceline_ref_seg[1,1]-raceline_ref_seg[0,1],
                            raceline_ref_seg[1,0]-raceline_ref_seg[0,0])
    x0_seg = np.array([start_sample[0], start_sample[1], theta0_seg, 0.0], dtype=np.float64)

    z_init = np.zeros(control_steps * 2, dtype=np.float64)
    # Continuous continuation
    u_opt, states_opt, _ = continuation_solve(z_init, x0_seg, raceline_ref_seg, theta_ref_seg, v_ref_seg,
                                              cfg, v_max_seg, L_pix, sdf_map)

    return dict(
        actions=u_opt.astype(np.float64),
        states=states_opt.astype(np.float64),
        x0=x0_seg,
        start=start_sample.astype(np.float64),
        end=end_sample.astype(np.float64),
        raceline_ref=raceline_ref_seg.astype(np.float64),
        v_max_seg=float(v_max_seg)
    )

def save_npz(path: str, buf: list, meta: dict):
    if not buf: return
    actions = np.stack([b["actions"] for b in buf], axis=0)  # [N,T,2]
    states  = np.stack([b["states"]  for b in buf], axis=0)  # [N,S,4]
    x0s     = np.stack([b["x0"]      for b in buf], axis=0)  # [N,4]
    starts  = np.stack([b["start"]   for b in buf], axis=0)  # [N,2]
    ends    = np.stack([b["end"]     for b in buf], axis=0)  # [N,2]
    refs    = np.stack([b["raceline_ref"] for b in buf], axis=0)  # [N,S,2]
    vmaxs   = np.array([b["v_max_seg"] for b in buf], dtype=np.float64)

    if os.path.isfile(path):
        old = np.load(path, allow_pickle=False)
        actions = np.concatenate([old["actions"], actions], axis=0)
        states  = np.concatenate([old["states"],  states],  axis=0)
        x0s     = np.concatenate([old["x0s"],     x0s],     axis=0)
        starts  = np.concatenate([old["starts"],  starts],  axis=0)
        ends    = np.concatenate([old["ends"],    ends],    axis=0)
        refs    = np.concatenate([old["refs"],    refs],    axis=0)
        vmaxs   = np.concatenate([old["vmaxs"],   vmaxs],   axis=0)
        old.close()

    np.savez_compressed(
        path,
        actions=actions, states=states, x0s=x0s, starts=starts, ends=ends, refs=refs, vmaxs=vmaxs,
        **meta
    )

def sanity_check(path: str):
    data = np.load(path, allow_pickle=False)
    print("=== NPZ keys:", list(data.keys()))
    print("actions:", data["actions"].shape, "states:", data["states"].shape, "refs:", data["refs"].shape)
    n = data["actions"].shape[0]
    control_steps = data["actions"].shape[1]
    state_steps = data["states"].shape[1]
    print(f"traj={n}, control_steps={control_steps}, state_steps={state_steps} (expect S=T+1)")
    print("Actions range:", data["actions"].min(), data["actions"].max())
    print("vmaxs range:", data["vmaxs"].min(), data["vmaxs"].max())
    ok = (state_steps == control_steps + 1)
    print("shape ok:", ok)
    data.close()

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
    
    # Check for NaN or Inf
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
        # Load map for background
        map_data = load_map(map_file)
        
        # Random trajectory selection
        n_samples = min(n_samples, n_traj)
        sample_indices = np.random.choice(n_traj, n_samples, replace=False)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dataset Validation - {n_samples} Random Trajectories', fontsize=16)
        
        # Subplot 1: Trajectories on map
        ax1 = axes[0, 0]
        ax1.imshow(map_data.grid_map, cmap='gray', origin='lower', alpha=0.6)
        ax1.plot(map_data.raceline_grid[:, 0], map_data.raceline_grid[:, 1], 'g-', linewidth=2, label='Reference Line')
        
        for i, idx in enumerate(sample_indices[:10]):  # Show only first 10 to avoid clutter
            traj_states = data["states"][idx]
            p = ax1.plot(traj_states[:, 0], traj_states[:, 1], alpha=0.7, linewidth=1.5)
            line_color = p[0].get_color()
            ax1.plot(traj_states[0, 0], traj_states[0, 1], '^', color=line_color, markersize=8, alpha=0.7)
            ax1.plot(traj_states[-1, 0], traj_states[-1, 1], 'o', color=line_color, markersize=8, alpha=0.7)
        
        ax1.set_title('Trajectory Distribution on Map')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Velocity distribution
        ax2 = axes[0, 1]
        all_velocities = []
        for idx in sample_indices:
            velocities = data["states"][idx][:, 3]  # Assuming 4th column is velocity
            all_velocities.extend(velocities)
        
        ax2.hist(all_velocities, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Velocity Distribution')
        ax2.set_xlabel('Velocity (pixels/s)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Control input distribution
        ax3 = axes[1, 0]
        sample_actions = data["actions"][sample_indices]
        accel = sample_actions[:, :, 0].flatten()
        steering = sample_actions[:, :, 1].flatten()
        
        ax3.scatter(accel, steering, alpha=0.5, s=1)
        ax3.set_title('Control Input Distribution')
        ax3.set_xlabel('Acceleration')
        ax3.set_ylabel('Steering Angle')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Single trajectory details
        ax4 = axes[1, 1]
        sample_idx = sample_indices[0]
        sample_traj = data["states"][sample_idx]
        sample_actions_single = data["actions"][sample_idx]
        
        time_steps = np.arange(len(sample_traj))
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(time_steps, sample_traj[:, 3], 'b-', label='Velocity', linewidth=2)
        line2 = ax4_twin.plot(time_steps[:-1], sample_actions_single[:, 0], 'r-', label='Acceleration', linewidth=2)
        line3 = ax4_twin.plot(time_steps[:-1], sample_actions_single[:, 1], 'g-', label='Steering', linewidth=2)
        
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Velocity', color='b')
        ax4_twin.set_ylabel('Control Inputs', color='r')
        ax4.set_title(f'Single Trajectory Example (Traj #{sample_idx})')
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save verification plot
        save_path = path.replace('.npz', '_verification.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Verification plot saved: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Error during visualization: {e}")
    
    finally:
        data.close()
    
    print(f"✅ Dataset verification complete!")
    return True

def main():
    global md, cfg, control_steps, sdf_map, L_pix, START_A, START_B, END_A, END_B
    
    ap = argparse.ArgumentParser("Parallel OCP dataset generator (incremental save)")
    ap.add_argument("--map", type=str, default="nuerburgring_segment_map.npz")
    ap.add_argument("--state_steps", type=int, default=101)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--a_max", type=float, default=35.0)
    ap.add_argument("--delta_max", type=float, default=1.0)
    ap.add_argument("--safety_pix", type=float, default=10.0)
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    
    # Generate random ID to avoid filename conflicts
    random_id = str(uuid.uuid4())[:8]
    default_save_path = f"./outputs/ocp_dataset_incremental_{random_id}.npz"
    ap.add_argument("--save", type=str, default=default_save_path)
    
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--start_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[349.3, 572.1, 382.7, 561.9])
    ap.add_argument("--end_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[120.70, 153.80, 95.90, 178.50])
    
    # Verification parameters
    ap.add_argument("--verify", action="store_true", 
                    help="Only verify existing dataset, no generation")
    ap.add_argument("--verify_samples", type=int, default=20,
                    help="Number of samples to visualize during verification")
    
    args = ap.parse_args()

    # If in verification mode, proceed directly to verification
    if args.verify:
        if os.path.exists(args.save):
            success = verify_dataset(args.save, args.verify_samples, args.map)
            if success:
                print("✅ Dataset verification successful!")
            else:
                print("❌ Dataset verification failed!")
            return
        else:
            print(f"❌ Specified dataset file does not exist: {args.save}")
            # Try to find latest dataset file
            output_dir = os.path.dirname(args.save)
            if os.path.exists(output_dir):
                npz_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
                if npz_files:
                    latest_file = max(npz_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
                    latest_path = os.path.join(output_dir, latest_file)
                    print(f"🔍 Found latest dataset file: {latest_path}")
                    print("Verify this file? (Enter y to continue, any other key to exit)")
                    user_input = input().strip().lower()
                    if user_input == 'y':
                        success = verify_dataset(latest_path, args.verify_samples, args.map)
                        if success:
                            print("✅ Dataset verification successful!")
                        else:
                            print("❌ Dataset verification failed!")
                    return
                else:
                    print(f"❌ No dataset files found in {output_dir}")
                    return
            else:
                print(f"❌ Output directory does not exist: {output_dir}")
                return

    # Normal data generation mode
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # --- Initialize global variables (inherited by child processes via fork) ---
    md = load_map(args.map)
    cfg = OCPConfig(
        state_steps=args.state_steps,
        dt=args.dt,
        a_max=args.a_max,
        delta_max=args.delta_max,
        safety_margin_pix=args.safety_pix
    )
    control_steps = cfg.state_steps - 1
    sdf_map, _ = build_sdf_from_grid(md.grid_map)
    L_pix = cfg.wheelbase_m * md.resolution
    START_A = np.array(args.start_box[:2]); START_B = np.array(args.start_box[2:])
    END_A   = np.array(args.end_box[:2]);   END_B   = np.array(args.end_box[2:])

    rng_master = np.random.default_rng(args.seed)

    # Metadata
    meta = dict(
        dt=np.float64(cfg.dt),
        state_steps=np.int64(cfg.state_steps),
        control_steps=np.int64(control_steps),
        a_max=np.float64(cfg.a_max),
        delta_max=np.float64(cfg.delta_max),
        resolution=np.float64(md.resolution),
        safety_margin_pix=np.float64(cfg.safety_margin_pix)
    )

    # ---- Scheduling and incremental saving ----
    t0 = time.time()
    produced_total, buffer = 0, []
    print(f"[Start] target={args.total}, chunk={args.chunk}, workers={args.workers}")
    print(f"[Info] Dataset will be saved to: {args.save}")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        in_flight = {}
        submit_n = min(args.total, args.workers * 2)
        for _ in range(submit_n):
            fut = ex.submit(solve_one, int(rng_master.integers(0, 2**31-1)))
            in_flight[fut] = True

        while produced_total < args.total:
            done_fut = next(as_completed(in_flight))
            in_flight.pop(done_fut, None)
            try:
                res = done_fut.result()
                buffer.append(res)
                produced_total += 1
                remaining = args.total - (len(in_flight) + produced_total)
                if remaining > 0:
                    fut = ex.submit(solve_one, int(rng_master.integers(0, 2**31-1)))
                    in_flight[fut] = True
            except Exception as e:
                print(f"[warn] one job failed: {e!r}")

            if len(buffer) >= args.chunk or (produced_total == args.total):
                save_npz(args.save, buffer, meta)
                print(f"[save] total={produced_total}/{args.total}  file={args.save}  (+{len(buffer)} saved)")
                buffer = []

    t1 = time.time()
    print(f"[Done] generated {produced_total} trajs in {t1-t0:.1f}s -> {args.save}")
    sanity_check(args.save)
    
    # Ask user about verification
    print("\n🔍 Verify and visualize the generated dataset? (Enter y to verify, any other key to skip)")
    user_input = input().strip().lower()
    if user_input == 'y':
        print("Starting dataset verification...")
        success = verify_dataset(args.save, args.verify_samples, args.map)
        if success:
            print("✅ Dataset verification successful!")
        else:
            print("❌ Dataset verification failed!")
    else:
        print("⏭️ Skipping verification step")
        print(f"💡 You can verify the dataset later using:")
        print(f"   python make_dataset.py --verify --save {args.save}")

if __name__ == "__main__":
    main()

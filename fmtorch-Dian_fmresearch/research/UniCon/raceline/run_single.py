# run_single.py
import argparse
import numpy as np
from raceline_core import (
    MapData, OCPConfig, load_map, build_reference_from_raceline,
    build_sdf_from_grid, continuation_solve, mppi_local_refine,
    plot_results, plot_mppi_results, sample_in_box, nearest_index,
    resample_polyline_to_N
)

def main():
    ap = argparse.ArgumentParser("Single OCP solve (+ optional MPPI polish)")
    ap.add_argument("--map", type=str, default="nuerburgring_segment_map.npz")
    ap.add_argument("--state_steps", type=int, default=101)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--a_max", type=float, default=35.0)
    ap.add_argument("--delta_max", type=float, default=1.0)
    ap.add_argument("--safety_pix", type=float, default=10.0)
    ap.add_argument("--run_mppi", action="store_true")
    ap.add_argument("--mppi_iters", type=int, default=30)
    ap.add_argument("--mppi_N", type=int, default=2048)
    ap.add_argument("--segment_demo", action="store_true",
                    help="Run one randomized segment OCP solve using START/END boxes.")
    ap.add_argument("--start_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[349.3, 572.1, 382.7, 561.9])
    ap.add_argument("--end_box", type=float, nargs=4, metavar=("Ax","Ay","Bx","By"),
                    default=[120.70, 153.80, 95.90, 178.50])
    # 在 argparse 定义后面补一行
    ap.add_argument("--inverse", action="store_true",
                    help="Swap sampled start/end (only affects --segment_demo).")
    ap.add_argument("--full_from_boxes", action="store_true",default=True,
                help="For the full-track OCP, override the raceline start/end with the midpoints of start_box/end_box.")

    args = ap.parse_args()




    # --- Load map ---
    md: MapData = load_map(args.map)
    cfg = OCPConfig(
        state_steps=args.state_steps,
        dt=args.dt,
        a_max=args.a_max,
        delta_max=args.delta_max,
        safety_margin_pix=args.safety_pix
    )

    # --- Reference on full raceline ---
    START_A = np.array(args.start_box[:2]); START_B = np.array(args.start_box[2:])
    END_A   = np.array(args.end_box[:2]);   END_B   = np.array(args.end_box[2:])
    start_mid = 0.5 * (START_A + START_B)
    end_mid   = 0.5 * (END_A   + END_B)


    raceline_full = md.raceline_grid.copy()

    if args.full_from_boxes:
        print("Full OCP: using the centers of start/end boxes as full-track start/end.")
        raceline_full[0]  = start_mid
        raceline_full[-1] = end_mid

    if args.inverse:
        print("Full OCP: [inverse] building reference on full raceline.")
        raceline_full = raceline_full[::-1].copy() 
        raceline_ref, theta_ref, path_len_px = build_reference_from_raceline(raceline_full, cfg.state_steps)
    else:
        print("Full OCP: building reference on full raceline.")
        raceline_ref, theta_ref, path_len_px = build_reference_from_raceline(raceline_full, cfg.state_steps)

    control_steps = cfg.state_steps - 1
    total_time = control_steps * cfg.dt
    avg_speed_pixels = path_len_px / (total_time + 1e-12)
    v_max = 3.5 * avg_speed_pixels
    v_ref = np.clip(np.linspace(0, 2.0 * avg_speed_pixels, cfg.state_steps), 0, v_max)

    # --- Initial state from first segment tangent ---
    dx, dy = raceline_ref[1, 0] - raceline_ref[0, 0], raceline_ref[1, 1] - raceline_ref[0, 1]
    theta0 = np.arctan2(dy, dx)
    x0 = np.array([raceline_ref[0, 0], raceline_ref[0, 1], theta0, 0.0])

    # --- SDF ---
    sdf_map, _ = build_sdf_from_grid(md.grid_map)
    L_pix = cfg.wheelbase_m * md.resolution

    # --- OCP continuation solve ---
    print("Solving OCP with continuation...")
    z0 = np.zeros(control_steps * 2)
    u_opt, states_opt, info = continuation_solve(z0, x0, raceline_ref, theta_ref, v_ref, cfg, v_max, L_pix, sdf_map)

    # --- Visualize OCP ---
    plot_results(states_opt, u_opt, raceline_ref, v_ref, md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max)

    # --- Optional MPPI polish ---
    if args.run_mppi:
        print("Polishing with MPPI...")
        u_mppi, states_mppi, J_mppi = mppi_local_refine(
            u_opt, x0, cfg.dt, L_pix, v_max, cfg.delta_rate_max,
            raceline_ref, theta_ref, v_ref, sdf_map, cfg.safety_margin_pix,
            cfg.a_max, cfg.delta_max,
            iters=args.mppi_iters, N=args.mppi_N, avg_speed_pixels=avg_speed_pixels
        )
        pre_clear = None  # 可按需计算 clearance 对比
        plot_mppi_results(states_opt, u_opt, states_mppi, u_mppi,
                          raceline_ref, v_ref, md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max)

    # --- Optional: one randomized segment demo ---
    if args.segment_demo:
        rng = np.random.default_rng(None)
        START_A = np.array(args.start_box[:2]); START_B = np.array(args.start_box[2:])
        END_A   = np.array(args.end_box[:2]);   END_B   = np.array(args.end_box[2:])
        start_sample = sample_in_box(START_A, START_B, rng)
        end_sample   = sample_in_box(END_A,   END_B,   rng)

        if args.inverse:
            print("[inverse] swapping sampled start/end points.")
            start_sample, end_sample = end_sample, start_sample

        idx_s = nearest_index(md.raceline_grid, start_sample)
        idx_e = nearest_index(md.raceline_grid, end_sample)
        if idx_s == idx_e: idx_e = min(idx_s + 1, len(md.raceline_grid) - 1)
        sub_poly = md.raceline_grid[idx_s:idx_e+1] if idx_s < idx_e else md.raceline_grid[idx_e:idx_s+1][::-1]
        raceline_seg, seg_len_px = resample_polyline_to_N(sub_poly, cfg.state_steps)
        raceline_seg[-1] = end_sample
        dxy_seg = np.diff(raceline_seg, axis=0, append=raceline_seg[-1:])
        theta_ref_seg = np.arctan2(dxy_seg[:,1], dxy_seg[:,0])

        avg_speed_seg = seg_len_px / (control_steps * cfg.dt + 1e-12)
        v_max_seg = 3.5 * avg_speed_seg
        v_ref_seg = np.clip(np.linspace(0, 2.0 * avg_speed_seg, cfg.state_steps), 0, v_max_seg)

        theta0_seg = np.arctan2(raceline_seg[1,1] - raceline_seg[0,1], raceline_seg[1,0] - raceline_seg[0,0])
        x0_seg = np.array([start_sample[0], start_sample[1], theta0_seg, 0.0])
        z_init = np.zeros(control_steps * 2)
        u_seg, states_seg, _ = continuation_solve(z_init, x0_seg, raceline_seg, theta_ref_seg, v_ref_seg,
                                                  cfg, v_max_seg, L_pix, sdf_map)
        plot_results(states_seg, u_seg, raceline_seg, v_ref_seg, md.grid_map, cfg.dt, cfg.a_max, cfg.delta_max, v_max_seg)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FR3 Trajectory Replay Script (MuJoCo) — Open-loop only-once alignment

- Align ONLY the first frame to recorded pre-step state.
- Then run forward OPEN-LOOP:
  * High-res: apply recorded ctrl/xfrc every step (100%).
  * Low-res:   sample by --percentage (e.g., 1%), expand by ZOH to every step, keep original dt.

Usage examples:
  # High-res open-loop (full resolution)
  python replay_trajectory_openloop.py --xml scene.xml --replay-path draw_eight_record.npz

  # Low-res open-loop (ZOH to full sim rate), 1% by default
  python replay_trajectory_openloop.py --xml scene.xml --replay-path draw_eight_record.npz --low-res

  # Low-res with custom percentage, e.g., 5%
  python replay_trajectory_openloop.py --xml scene.xml --replay-path draw_eight_record.npz --low-res --percentage 5

  # Headless + no plots
  python replay_trajectory_openloop.py --xml scene.xml --replay-path draw_eight_record.npz --hide-viewer --no-plot
"""

import argparse
import time
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt


def restore_pre_step(model, data, pre_state, set_time=None):
    """Restore simulation state to specified pre-step state (one-time at i=0)."""
    data.qpos[:] = pre_state["qpos"]
    data.qvel[:] = pre_state["qvel"]
    if model.na > 0 and pre_state.get("act") is not None and len(pre_state["act"]) == model.na:
        data.act[:] = pre_state["act"]
    if model.nmocap > 0 and pre_state.get("mocap_pos") is not None:
        data.mocap_pos[:] = pre_state["mocap_pos"]
        data.mocap_quat[:] = pre_state["mocap_quat"]
    data.qacc_warmstart[:] = pre_state["qacc_warmstart"]
    if set_time is not None:
        data.time = set_time
    mujoco.mj_forward(model, data)


def build_zoh_series(actions, xfrcs, keep_percentage: int):
    """
    Build ZOH-expanded series at the original simulation step rate.

    actions, xfrcs: shape (N, ...)
    keep_percentage in [1, 100]
    Returns arrays of the same length N with piecewise-constant segments.
    """
    N = actions.shape[0]
    if keep_percentage >= 100:
        return actions.copy(), xfrcs.copy()

    K = max(2, int(np.ceil(N * keep_percentage / 100.0)))
    keep_idx = np.linspace(0, N - 1, K, dtype=int)

    actions_zoh = np.empty_like(actions)
    xfrcs_zoh   = np.empty_like(xfrcs)

    # fill segments [keep_idx[j-1], keep_idx[j])
    for j in range(1, len(keep_idx)):
        a = actions[keep_idx[j-1]]
        f = xfrcs[keep_idx[j-1]]
        actions_zoh[keep_idx[j-1]:keep_idx[j]] = a
        xfrcs_zoh[keep_idx[j-1]:keep_idx[j]]   = f

    # tail
    actions_zoh[keep_idx[-1]:] = actions[keep_idx[-1]]
    xfrcs_zoh[keep_idx[-1]:]   = xfrcs[keep_idx[-1]]
    return actions_zoh, xfrcs_zoh


def main():
    parser = argparse.ArgumentParser("FR3 Trajectory Replay Script (Open-loop, one-time alignment)")
    parser.add_argument("--xml", type=str, default="scene.xml", help="MuJoCo XML file path")
    parser.add_argument("--replay_path", type=str, default="draw_eight_record.npz", help="Trajectory data file path (.npz)")
    parser.add_argument("--low_res", action="store_true", help="Use low resolution control with ZOH expansion (keep original dt)")
    parser.add_argument("--percentage", type=int, default=50, help="Low-res keep percentage (1-100). Only used if --low-res.")
    parser.add_argument("--hide_viewer", action="store_true", help="Hide visualization window")
    parser.add_argument("--plot_results", action="store_true", default=True, help="Show result plots")
    parser.add_argument("--no_plot", action="store_true", help="Don't show plots")
    args = parser.parse_args()

    print(f"Loading MuJoCo model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    print(f"Loading trajectory data from {args.replay_path}")
    try:
        arr = np.load(args.replay_path, allow_pickle=True)
        N = len(arr["times"])
        print(f"Loaded {N} timesteps, duration {arr['times'][-1]:.3f}s")
    except Exception as e:
        print(f"Failed to load trajectory data: {e}")
        return

    # Inputs at recording rate
    actions_raw = arr["ctrl"].copy()   # (N, nu_used)
    xfrcs_raw   = arr["xfrc"].copy()   # (N, nbody, 6)

    # Build driving signals for open-loop
    if args.low_res:
        pct = int(np.clip(args.percentage, 1, 100))
        print(f"Low-res open-loop with ZOH to full sim rate: {pct}% of frames kept.")
        actions, xfrcs = build_zoh_series(actions_raw, xfrcs_raw, pct)
        resolution_name = f"low-res ZOH ({pct}%)"
    else:
        actions, xfrcs = actions_raw, xfrcs_raw
        resolution_name = "high-res (100%)"

    # Create a separate data for replay
    data2 = mujoco.MjData(model)

    # ---------- Only-once alignment at i=0 ----------
    pre0 = {
        "qpos": arr["pre_qpos"][0],
        "qvel": arr["pre_qvel"][0],
        "act":  arr["pre_act"][0] if model.na > 0 else None,
        "mocap_pos": arr["pre_mocap_pos"][0] if model.nmocap > 0 else None,
        "mocap_quat": arr["pre_mocap_quat"][0] if model.nmocap > 0 else None,
        "qacc_warmstart": arr["pre_qacc_warmstart"][0],
    }
    restore_pre_step(model, data2, pre0, set_time=arr["times"][0])
    print("Aligned to the first pre-step state. Start open-loop forward ...")

    # ---------- Open-loop forward ----------
    errors = []
    times2 = []
    replayed_qpos = []
    replayed_qvel = []
    replayed_tau  = []

    start_time = time.time()
    print(f"Resolution mode: {resolution_name}")

    def step_once(i: int):
        # Apply ctrl/xfrc for this step (NO restore after i=0)
        data2.ctrl[:] = 0.0
        data2.ctrl[:actions.shape[1]] = actions[i]
        data2.xfrc_applied[:] = xfrcs[i]

        mujoco.mj_step(model, data2)

        # Compare to recorded post-step at the same index i
        e = data2.qpos.copy() - arr["post_qpos"][i]
        errors.append(e)
        times2.append(data2.time)
        replayed_qpos.append(data2.qpos.copy())
        replayed_qvel.append(data2.qvel.copy())
        replayed_tau.append(actions[i].copy())

        if i % 200 == 0:
            print(f"Open-loop progress: {i}/{len(actions)} ({100*i/len(actions):.1f}%)")

    if args.hide_viewer:
        for i in range(N):
            step_once(i)
    else:
        try:
            with viewer.launch_passive(model, data2) as v:
                i = 0
                while v.is_running() and i < N:
                    step_once(i)
                    v.sync()
                    time.sleep(0.0004)
                    i += 1
        except Exception:
            try:
                with viewer.launch(model, data2) as v:
                    i = 0
                    while v.is_running() and i < N:
                        step_once(i)
                        v.sync()
                        time.sleep(0.0004)
                        i += 1
            except Exception as e:
                print(f"Viewer failed, continue headless: {e}")
                for i in range(N):
                    step_once(i)

    replay_time = time.time() - start_time
    errors = np.stack(errors, axis=0)
    replayed_qpos = np.stack(replayed_qpos, axis=0)
    replayed_qvel = np.stack(replayed_qvel, axis=0)
    replayed_tau  = np.stack(replayed_tau, axis=0)
    times2 = np.array(times2)
    mse = float(np.mean(errors**2))

    print("\n=== Open-loop Replay Complete ===")
    print(f"Replay duration (wall): {replay_time:.2f} s")
    print(f"Replay steps: {len(times2)} / {N}")
    print(f"Position MSE vs recorded post-step: {mse:.6e}")
    print(f"Mode: {resolution_name}")
    print(f"States shape:  {replayed_qpos.shape}")
    print(f"Controls shape:{actions.shape}")

    # ---------- Plots ----------
    if args.plot_results and not args.no_plot:
        num_replayed_steps = len(times2)

        plt.figure(figsize=(15, 12))

        # 1) q
        plt.subplot(3, 2, 1)
        for j in range(min(7, replayed_qpos.shape[1])):
            plt.plot(times2, replayed_qpos[:, j], label=f"J{j+1}")
        plt.legend(); plt.title("Joint Positions (q)")
        plt.xlabel("Time (s)"); plt.ylabel("rad"); plt.grid(True)

        # 2) dq
        plt.subplot(3, 2, 2)
        for j in range(min(7, replayed_qvel.shape[1])):
            plt.plot(times2, replayed_qvel[:, j], label=f"J{j+1}")
        plt.legend(); plt.title("Joint Velocities (dq)")
        plt.xlabel("Time (s)"); plt.ylabel("rad/s"); plt.grid(True)

        # 3) tau
        plt.subplot(3, 2, 3)
        for j in range(min(7, replayed_tau.shape[1])):
            plt.plot(times2, replayed_tau[:, j], label=f"J{j+1}")
        plt.legend(); plt.title("Joint Torques (tau)")
        plt.xlabel("Time (s)"); plt.ylabel("Nm"); plt.grid(True)

        # 4) q vs recorded post
        plt.subplot(3, 2, 4)
        for j in range(min(7, replayed_qpos.shape[1])):
            plt.plot(times2, replayed_qpos[:, j],
                     label="Replay" if j == 0 else "", color=f"C{j}", alpha=0.75)
        for j in range(min(7, replayed_qpos.shape[1])):
            plt.plot(arr["times"][:num_replayed_steps],
                     arr["post_qpos"][:num_replayed_steps, j],
                     label="Recorded" if j == 0 else "",
                     linestyle='--', alpha=0.8, color=f"C{j}")
        plt.legend(); plt.title("Joint Position: Replay vs Recorded")
        plt.xlabel("Time (s)"); plt.ylabel("rad"); plt.grid(True)

        # 5) ||error||
        plt.subplot(3, 2, 5)
        err_norm = np.linalg.norm(errors, axis=1)
        plt.plot(times2, err_norm, 'r-', linewidth=2)
        plt.title("Position Error Norm"); plt.xlabel("Time (s)"); plt.ylabel("rad"); plt.grid(True)

        # 6) per-joint error
        plt.subplot(3, 2, 6)
        for j in range(min(7, errors.shape[1])):
            plt.plot(times2, errors[:, j], label=f"J{j+1}")
        plt.legend(); plt.title("Per-joint Position Errors")
        plt.xlabel("Time (s)"); plt.ylabel("rad"); plt.grid(True)

        plt.tight_layout(); plt.show()

    print("Replay complete!")


if __name__ == "__main__":
    main()

# SafeFlow ReFlow Ablation

This directory runs the same four SafeFlow scenarios with the Rectified Flow
checkpoint. The original launchers, checkpoint, and saved NuBoard files are
left untouched.

The default checkpoint is:

`$SAFEFLOW_PROJECT_ROOT/rectified/checkpoints/model_reflow_round1.pt`

The default `FM_NUM_SEGMENTS=1` is the ReFlow low-NFE setting. Set
`FM_NUM_SEGMENTS=2`, `4`, or `8` when comparing sampling steps. All other
scenario parameters remain those of the corresponding SafeFlow launcher.

## Run

From `flow_planner/run_script`:

```bash
./safeflow_reflow_ablation/launch_sim_reflow_straight.sh
./safeflow_reflow_ablation/launch_sim_reflow_curve.sh
./safeflow_reflow_ablation/launch_sim_reflow_multi_vehicle.sh
./safeflow_reflow_ablation/launch_sim_reflow_pedestrian.sh
./safeflow_reflow_ablation/launch_sim_reflow_turn.sh
```

Results are copied to `saved_nuboards/` in this directory. After the runs:

```bash
./safeflow_reflow_ablation/launch_nuboard_reflow_ablation.sh
```

The turn experiment uses a `starting_right_turn` route from the Las Vegas Strip
NuPlan map and adds two vehicles that follow the curved route geometry.

For a first smoke test, run the straight scene with `SCENARIO_LIMIT=1` and
`PLANNER_DEVICE=cpu`. A successful load does not by itself prove that ReFlow
has learned obstacle avoidance: the trained coupling data came from the
available obstacle-free ring-track dataset, while CBF remains the runtime
safety layer.

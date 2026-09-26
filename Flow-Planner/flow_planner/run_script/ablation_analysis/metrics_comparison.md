# Ablation Metrics Comparison

This directory contains the current comparison of the saved SafeFlow,
ReFlow, Flow-Planner, PDM, and MPC-CBF runs. Each method has one saved run
for each semantic scene label: Straight, Curve, Multi-vehicle, and
Pedestrian.

The figures are:

- `route_ratio.png`: route progress ratio.
- `ego_progress.png`: ego progress in meters.
- `planning_latency.png`: mean `compute_trajectory_runtimes` in milliseconds.
- `collision_free.png`: no-ego-at-fault collision pass/fail per cell.
- `drivable_area.png`: drivable-area compliance.
- `comfort_compliance.png`: comfort compliance.
- `composite_score.png`: aggregated NuPlan score.

## Mean Across Four Saved Runs

| Method | Route ratio | Progress (m) | Collision-free cells | Drivable area | Comfort | Score | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SafeFlow | 0.616 | 68.37 | 4/4 | 1.000 | 0.500 | 0.590 | 161.5 |
| ReFlow | 0.514 | 63.25 | 3/4 | 0.750 | 0.250 | 0.188 | 54.5 |
| Flow-Planner | 0.531 | 60.46 | 4/4 | 1.000 | 0.750 | 0.804 | 253.1 |
| PDM | 0.843 | 100.87 | 4/4 | 1.000 | 0.500 | 0.881 | 94.3 |
| MPC-CBF | 0.914 | 109.23 | 4/4 | 1.000 | 1.000 | 0.787 | 544.0 |

These are descriptive summaries of one saved run per method and scene, not a
statistical collision probability. In particular, the current ReFlow Round-1
checkpoint is faster but has one collision and one curve run with zero
drivable-area compliance; it should not be presented as a final safe baseline
until it is retrained or its inference configuration is corrected.

The methods are grouped by the existing semantic scene labels. Before using
the results as a final paper table, verify that every method uses the same
scenario token, horizon, CPU/GPU setting, and planner frequency. The raw rows
are in `metrics.csv`; the aggregate table is in `summary_metrics.csv`.

Regenerate all figures with:

```bash
MPLCONFIGDIR=/tmp/ablation_matplotlib \
  /new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python \
  ablation_analysis/generate_report.py
```

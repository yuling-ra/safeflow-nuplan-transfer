# PDM Ablation Experiments

This directory runs the official `PDMClosedPlanner` implementation from
`tuplan_garage` against the same four local scenarios used by the
Flow-Planner ablation. It does not load a Flow-Planner checkpoint.

The vendored PDM implementation is from `tuplan_garage` and targets
`nuplan-devkit==1.2.2`, matching the local nuPlan environment. The four
scenario launchers are:

| Experiment | Launcher |
| --- | --- |
| Straight road, dense traffic and slowed lead vehicle with overtaking | `launch_sim_pdm_straight.sh` |
| Curved road, two moving obstacle vehicles with overtaking | `launch_sim_pdm_curve_two_vehicle.sh` |
| Combined dense multi-vehicle avoidance | `launch_sim_pdm_multi_vehicle.sh` |
| Right turn with straight-road pedestrian crossing | `launch_sim_pdm_pedestrian.sh` |

Run from this directory with the local CPU environment:

```bash
PLANNER_DEVICE=cpu ./launch_sim_pdm_straight.sh
PLANNER_DEVICE=cpu ./launch_sim_pdm_curve_two_vehicle.sh
PLANNER_DEVICE=cpu ./launch_sim_pdm_multi_vehicle.sh
PLANNER_DEVICE=cpu ./launch_sim_pdm_pedestrian.sh
```

Each successful run is copied to `saved_nuboards/`. Open all four results with:

```bash
./launch_nuboard_pdm_ablation.sh
```

The default launcher now uses a complex PDM configuration: 9 lateral paths x
9 IDM policies (81 proposals), a 6-second proposal horizon, and additional
bounded smoothness, curvature, acceleration, jerk, and lateral-acceleration
costs. This increases actual proposal generation, batch simulation, and cost
evaluation; it does not add an artificial delay. Set
`PDM_COMPLEX_MODE=false` to reproduce the original 15-proposal baseline.

The candidate count and cost weights are configurable through the
`PDM_*` environment variables in `run_pdm_ablation.sh`.

Complex results are saved with the `complex_` prefix and can be opened with
`./launch_nuboard_pdm_complex_ablation.sh`. Baseline results remain under the
original names when running with `PDM_COMPLEX_MODE=false`.

The synthetic obstacle speeds and route progress can be overridden with the
same `SYNTHETIC_*` environment variables used by the Flow-Planner launchers.

The straight launcher uses a tuned PDM-Closed configuration for the intended
overtaking case: the selected lead vehicle runs at 35% of its recorded speed,
the candidate lateral offsets reach the neighboring lane, and the fastest IDM
proposal is capped at the road speed limit. The other three experiments retain
their conservative avoidance settings.

The multi-vehicle launcher keeps the original recorded traffic and adds the two
controlled route-aligned vehicles, so that experiment contains dense traffic
instead of only three total vehicles.

The pedestrian launcher places the crossing on the straight approach before
the right turn. Both pedestrians start crossing at the scenario start and move
to the far sidewalk before the ego proceeds through the crossing area. The
pedestrians remain on the sidewalks after crossing and do not turn back.
The accompanying vehicles use normal moving speeds so they do not become
stationary visual blockers at the end of the scenario.

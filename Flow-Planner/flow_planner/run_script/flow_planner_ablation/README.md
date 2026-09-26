# Flow-Planner Ablation Experiments

The straight-road experiment uses the native `changing_lane` scenario
`f6f9afda75e251ae`. It keeps dense recorded traffic and slows the selected lead
vehicle consistently to 35% velocity and 80% trajectory progress. The validated
result is collision-free and stays in the drivable area, but the current pure
Flow-Planner checkpoint follows the lead vehicle rather than guaranteeing an
overtake.

These launchers use the same maps and scenario semantics as the four local
SafeFlow experiments, while replacing the planner with the released
Flow-Planner checkpoint. The straight and combined-vehicle defaults use
Flow-Planner-compatible traffic motion: the lead vehicle remains on its
recorded trajectory, and the two combined obstacles travel at 4.0 m/s. This
avoids turning the learned-planner input into a near-static blocker, which is
not the checkpoint's normal data distribution.

The synthetic three-vehicle scenes are intentionally kept separate from
recorded replay traffic. Replay agents are open-loop: if the planner selects a
different lane, a replay vehicle can later cross the ego trajectory without
reacting. `SYNTHETIC_KEEP_RECORDED_VEHICLES=true` is therefore an explicit
stress-test option, not the default evaluation setup.

These are pure checkpoint ablations. Flow-Planner has no explicit overtake
objective or collision-checking trajectory selector, so reducing an obstacle's
speed can make it follow or stop instead of pass. A guaranteed overtake requires
a separately named safety/overtake wrapper and should not be reported as the
pure checkpoint result.

To reproduce the original SafeFlow stress parameters for comparison, override
the values explicitly, for example:

```bash
SINGLE_AGENT_VEHICLE_SPEED_SCALE=0.35 SINGLE_AGENT_VEHICLE_POSITION_SCALE=0.35 \
  ./launch_sim_flow_planner_straight.sh
SYNTHETIC_VEHICLE_1_SPEED_MPS=0.5 SYNTHETIC_VEHICLE_2_SPEED_MPS=2.5 \
  ./launch_sim_flow_planner_multi_vehicle.sh
```

| Experiment | Launcher | SafeFlow counterpart |
| --- | --- | --- |
| Straight road, multi-car traffic and one slowed lead vehicle | `launch_sim_flow_planner_straight.sh` | `launch_sim_safeflow_straight_overtake.sh` |
| Straight-road overtake with explicit collision-checked wrapper | `launch_sim_flow_planner_straight_overtake.sh` | n/a |
| Curved road, two synthetic obstacle vehicles | `launch_sim_flow_planner_curve_two_vehicle.sh` | `launch_sim_safeflow_three_vehicle_turn_avoidance.sh` |
| Combined multi-vehicle avoidance | `launch_sim_flow_planner_multi_vehicle.sh` | `launch_sim_safeflow_three_vehicle_combined_avoidance.sh` |
| Right turn with straight-road pedestrian crossing | `launch_sim_flow_planner_pedestrian.sh` | `launch_sim_safeflow_right_turn_pedestrian.sh` |

The Flow-Planner checkpoint defaults to:

`/new_world/cockatiel/TeleNas/DataExchange/yuling/flow_planner_ckpt/model.pth`

Run with `PLANNER_DEVICE=cpu` for the local CPU environment. Each completed run is copied into `saved_nuboards/`. Open all four results with:

```bash
./launch_nuboard_flow_planner_ablation.sh
```

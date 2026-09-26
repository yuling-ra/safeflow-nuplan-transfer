# MPC-CBF Ablation

This is an independent optimization-based baseline. It uses a finite-horizon
nonlinear kinematic-bicycle MPC with tracking, speed, control-smoothness and
terminal costs. Predicted vehicles, pedestrians, and bicycles are enforced
with discrete CBF constraints in the SLSQP problem. The nuPlan IDM planner supplies
only the nominal route/speed reference; it is not the safety mechanism.

Run the four matching scenes with:

```bash
./mpc_cbf_ablation/launch_sim_mpc_cbf_straight.sh
./mpc_cbf_ablation/launch_sim_mpc_cbf_curve.sh
./mpc_cbf_ablation/launch_sim_mpc_cbf_multi_vehicle.sh
./mpc_cbf_ablation/launch_sim_mpc_cbf_pedestrian.sh
```

The initial parameters are intentionally explicit and configurable through
environment variables such as `MPC_HORIZON_STEPS`, `MPC_SOLVER_MAXITER`,
`MPC_OBSTACLE_MARGIN_M`, and `MPC_OBSTACLE_PREDICTION_HORIZON_S`. These are
baseline parameters, not yet a tuned result. A run is not considered a valid
paper result unless collision, progress, drivable-area compliance, and solver
fallback counts are checked together. Open completed runs with:

```bash
./mpc_cbf_ablation/launch_nuboard_mpc_cbf.sh
```

Validation runs completed on one scenario per scene with
`MPC_SOLVER_MAXITER=10`, CPU execution, and the default `0.75 m` tracking-error
margin:

| Scene | No-ego-fault collisions | Route progress | Drivable area | Mean planner time |
| --- | ---: | ---: | ---: | ---: |
| Straight | 0 | 79.9% | 100% | 505 ms |
| Curve / two-vehicle | 0 | 100% | 100% | 664 ms |
| Multi-vehicle | 0 | 85.9% | 100% | 621 ms |
| Pedestrian | 0 | 100% | 100% | 386 ms |

These are smoke/validation results, not a statistical comparison. Run the same
scenario set and solver budget for every baseline before reporting a table.
The straight launcher also applies the same `SingleAgentScenario` lead-vehicle
slowdown used by the PDM and Flow-Planner straight experiments.
When IDM returns a route that is displaced from the current ego pose, the
planner uses the nearest heading-consistent map lane geometry instead of
following the invalid route.

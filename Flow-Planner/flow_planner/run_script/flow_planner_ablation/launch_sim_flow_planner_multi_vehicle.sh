#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same combined three-vehicle scene as SafeFlow's successful multi-vehicle run.
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE=combined
# The synthetic scene is closed-loop compatible.  Recorded traffic remains an
# opt-in stress test because replay agents do not react when Flow-Planner
# chooses a different lane and can then cut across the ego trajectory.
export SYNTHETIC_KEEP_RECORDED_VEHICLES="${SYNTHETIC_KEEP_RECORDED_VEHICLES:-false}"
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-150}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-32.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-48.0}"
# The checkpoint handles normal moving traffic, but the SafeFlow stress
# speeds (0.5/2.5 m/s) create near-stationary blockers that are out of
# distribution for this learned planner.
export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-2.5}"
export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-3.0}"
export SINGLE_AGENT_SCENARIO=false

exec "$SCRIPT_DIR/run_flow_planner_ablation.sh" multi_vehicle \
    planner.flow_planner.future_trajectory_sampling.time_horizon=8.0 \
    "$@"

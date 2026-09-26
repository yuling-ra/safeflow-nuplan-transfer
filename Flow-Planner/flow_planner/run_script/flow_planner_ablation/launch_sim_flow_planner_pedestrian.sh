#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same right-turn straight-approach pedestrian scene as SafeFlow.
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.10.11.02.57.41_veh-50_01522_02088.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_right_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-be47172809a953bd}"
export MAP_NAMES="${MAP_NAMES:-sg-one-north}"
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE=right_turn_pedestrian
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-120}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-30.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-40.0}"
export SYNTHETIC_PEDESTRIAN_SPEED_MPS="${SYNTHETIC_PEDESTRIAN_SPEED_MPS:-2.0}"
export SYNTHETIC_PEDESTRIAN_PROGRESS_M="${SYNTHETIC_PEDESTRIAN_PROGRESS_M:-7.0}"
export SYNTHETIC_PEDESTRIAN_SEED="${SYNTHETIC_PEDESTRIAN_SEED:-17}"
export SINGLE_AGENT_SCENARIO=false

exec "$SCRIPT_DIR/run_flow_planner_ablation.sh" pedestrian \
    planner.flow_planner.future_trajectory_sampling.time_horizon=8.0 \
    "$@"

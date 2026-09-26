#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SYNTHETIC_AVOIDANCE_MODE=intersection_turn
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-110}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-30.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-40.0}"

exec "$SCRIPT_DIR/launch_sim_safeflow_synthetic_three_vehicle.sh" \
    planner.safeflow_planner.local_goal_distance_m=50.0 \
    planner.safeflow_planner.avoidance_lateral_offset_m=3.4 \
    planner.safeflow_planner.avoidance_transition_distance_m=20.0 \
    "$@"

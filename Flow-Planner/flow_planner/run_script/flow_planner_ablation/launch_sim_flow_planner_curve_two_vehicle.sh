#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Same curved two-obstacle scene as SafeFlow's intersection-turn experiment.
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE=intersection_turn
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-149}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-35.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-50.0}"
export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-2.5}"
export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-3.0}"
export SINGLE_AGENT_SCENARIO=false

exec "$SCRIPT_DIR/run_flow_planner_ablation.sh" curve_two_vehicle "$@"

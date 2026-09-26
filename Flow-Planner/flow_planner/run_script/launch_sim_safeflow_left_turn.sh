#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SINGLE_AGENT_SCENARIO=false
unset SINGLE_AGENT_SCENARIO_TOKENS
unset SINGLE_AGENT_VEHICLE_SPEED_SCALE
unset SINGLE_AGENT_VEHICLE_POSITION_SCALE
unset SINGLE_AGENT_VEHICLE_STRAIGHT

"$SCRIPT_DIR/launch_sim_safeflow.sh" \
    planner.safeflow_planner.enable_overtake_demo=false \
    planner.safeflow_planner.force_overtake_demo=false \
    "$@"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export SINGLE_AGENT_VEHICLE_SPEED_SCALE="${SINGLE_AGENT_VEHICLE_SPEED_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_POSITION_SCALE="${SINGLE_AGENT_VEHICLE_POSITION_SCALE:-$SINGLE_AGENT_VEHICLE_SPEED_SCALE}"
export SINGLE_AGENT_VEHICLE_STRAIGHT="${SINGLE_AGENT_VEHICLE_STRAIGHT:-true}"

"$SCRIPT_DIR/launch_sim_safeflow.sh" \
    planner.safeflow_planner.enable_overtake_demo=true \
    planner.safeflow_planner.force_overtake_demo=false \
    planner.safeflow_planner.overtake_pass_side=left \
    planner.safeflow_planner.overtake_lateral_offset_m=4.5 \
    planner.safeflow_planner.overtake_min_lead_distance_m=12.0 \
    planner.safeflow_planner.overtake_max_lead_distance_m=30.0 \
    planner.safeflow_planner.overtake_distance_m=58.0 \
    planner.safeflow_planner.overtake_target_speed_mps=7.0 \
    planner.safeflow_planner.overtake_min_clearance_m=4.5 \
    planner.safeflow_planner.overtake_require_lane=true \
    planner.safeflow_planner.overtake_follow_distance_m=18.0 \
    planner.safeflow_planner.overtake_lane_change_distance_m=26.0 \
    planner.safeflow_planner.overtake_return_to_original_lane=false \
    planner.safeflow_planner.overtake_allow_lane_connector=true \
    "$@"

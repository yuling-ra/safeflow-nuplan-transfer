#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.05.25.14.16.10_veh-35_01690_02183.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-changing_lane}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f6f9afda75e251ae}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SINGLE_AGENT_SCENARIO=true
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-f6f9afda75e251ae}"
# Make the lead vehicle slow enough for a clear, reproducible overtake while
# retaining the original dense multi-car map.
export SINGLE_AGENT_VEHICLE_SPEED_SCALE="${SINGLE_AGENT_VEHICLE_SPEED_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_POSITION_SCALE="${SINGLE_AGENT_VEHICLE_POSITION_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_STRAIGHT="${SINGLE_AGENT_VEHICLE_STRAIGHT:-true}"
export SINGLE_AGENT_KEEP_OTHER_VEHICLES="${SINGLE_AGENT_KEEP_OTHER_VEHICLES:-true}"
unset SYNTHETIC_THREE_VEHICLE_SCENARIO SYNTHETIC_AVOIDANCE_MODE

exec "$SCRIPT_DIR/run_pdm_ablation.sh" straight \
    planner.pdm_closed_planner.idm_policies.speed_limit_fraction='[0.3,0.5,0.7,0.9,1.0]' \
    planner.pdm_closed_planner.idm_policies.fallback_target_velocity=13.5 \
    planner.pdm_closed_planner.idm_policies.accel_max=2.0 \
    planner.pdm_closed_planner.idm_policies.min_gap_to_lead_agent=1.0 \
    planner.pdm_closed_planner.lateral_offsets='[-3.5,3.5]' \
    "$@"

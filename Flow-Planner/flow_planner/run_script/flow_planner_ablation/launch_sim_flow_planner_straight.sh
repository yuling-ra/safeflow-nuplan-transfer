#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Long straight changing-lane scene with dense recorded traffic. The selected
# lead vehicle is slowed consistently in both velocity and position progress.
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.05.25.14.16.10_veh-35_01690_02183.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-changing_lane}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f6f9afda75e251ae}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SINGLE_AGENT_SCENARIO=true
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-f6f9afda75e251ae}"
export SINGLE_AGENT_VEHICLE_SPEED_SCALE="${SINGLE_AGENT_VEHICLE_SPEED_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_POSITION_SCALE="${SINGLE_AGENT_VEHICLE_POSITION_SCALE:-0.80}"
export SINGLE_AGENT_VEHICLE_STRAIGHT="${SINGLE_AGENT_VEHICLE_STRAIGHT:-true}"
export SINGLE_AGENT_KEEP_OTHER_VEHICLES="${SINGLE_AGENT_KEEP_OTHER_VEHICLES:-true}"
unset SYNTHETIC_THREE_VEHICLE_SCENARIO
unset SYNTHETIC_AVOIDANCE_MODE

planner_overrides=()
if [[ "${FLOW_PLANNER_OVERTAKE_WRAPPER:-false}" == "true" ]]; then
    planner_overrides+=("planner.flow_planner._target_=flow_planner_overtake_wrapper.FlowPlannerOvertake")
fi

exec "$SCRIPT_DIR/run_flow_planner_ablation.sh" "${FLOW_PLANNER_EXPERIMENT_NAME:-straight}" \
    "${planner_overrides[@]}" "$@"

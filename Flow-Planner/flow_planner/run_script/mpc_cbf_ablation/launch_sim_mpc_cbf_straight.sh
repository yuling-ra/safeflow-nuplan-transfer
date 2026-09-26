#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.05.25.14.16.10_veh-35_01690_02183.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-changing_lane}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f6f9afda75e251ae}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SINGLE_AGENT_SCENARIO=true
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-f6f9afda75e251ae}"
export SINGLE_AGENT_VEHICLE_SPEED_SCALE="${SINGLE_AGENT_VEHICLE_SPEED_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_POSITION_SCALE="${SINGLE_AGENT_VEHICLE_POSITION_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_STRAIGHT="${SINGLE_AGENT_VEHICLE_STRAIGHT:-true}"
export SINGLE_AGENT_KEEP_OTHER_VEHICLES="${SINGLE_AGENT_KEEP_OTHER_VEHICLES:-true}"
exec "$SCRIPT_DIR/run_mpc_cbf_ablation.sh" straight "$@"

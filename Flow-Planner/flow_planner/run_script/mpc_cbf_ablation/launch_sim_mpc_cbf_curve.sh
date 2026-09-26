#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
exec "$SCRIPT_DIR/run_mpc_cbf_ablation.sh" curve_two_vehicle "$@"

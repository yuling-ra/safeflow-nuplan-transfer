#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.10.11.02.57.41_veh-50_01522_02088.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_right_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-be47172809a953bd}"
export MAP_NAMES="${MAP_NAMES:-sg-one-north}"
exec "$SCRIPT_DIR/run_mpc_cbf_ablation.sh" pedestrian "$@"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
exec "$SCRIPT_DIR/run_mpc_cbf_ablation.sh" multi_vehicle "$@"

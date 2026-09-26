#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$(cd -- "$SCRIPT_DIR/../../../nuplan-devkit" && pwd)}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMBINED="$SCRIPT_DIR/saved_nuboards/three_vehicle_combined_avoidance.nuboard"

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$COMBINED]" \
    "$@"

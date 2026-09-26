#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
YULING_ROOT="$(cd -- "$RUN_SCRIPT_DIR/../../.." && pwd)"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$YULING_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pdm_complex_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$RUN_SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

files=(
    "$SCRIPT_DIR/saved_nuboards/complex_straight.nuboard"
    "$SCRIPT_DIR/saved_nuboards/complex_curve_two_vehicle.nuboard"
    "$SCRIPT_DIR/saved_nuboards/complex_multi_vehicle.nuboard"
    "$SCRIPT_DIR/saved_nuboards/complex_pedestrian.nuboard"
)
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing complex PDM NuBoard file: %s\nRun the corresponding PDM launcher first.\n' "$file" >&2
        exit 2
    fi
done

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$(IFS=,; echo "${files[*]}")]" \
    "$@"

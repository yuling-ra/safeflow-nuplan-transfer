#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$PROJECT_ROOT/../nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/pdm_ablation_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pdm_ablation_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/..:$PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

files=(
    "$SCRIPT_DIR/saved_nuboards/straight.nuboard"
    "$SCRIPT_DIR/saved_nuboards/curve_two_vehicle.nuboard"
    "$SCRIPT_DIR/saved_nuboards/multi_vehicle.nuboard"
    "$SCRIPT_DIR/saved_nuboards/pedestrian.nuboard"
)
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing NuBoard file: %s\nRun the four PDM experiments first.\n' "$file" >&2
        exit 2
    fi
done

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$(IFS=,; echo "${files[*]}")]" \
    "$@"

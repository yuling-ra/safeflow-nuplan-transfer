#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$PROJECT_ROOT/../nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/flow_planner_ablation_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flow_planner_ablation_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

AB_ROOT="${FLOW_PLANNER_ABLATION_ROOT:-$SCRIPT_DIR}"
files=(
    "$AB_ROOT/saved_nuboards/straight.nuboard"
    "$AB_ROOT/saved_nuboards/straight_overtake.nuboard"
    "$AB_ROOT/saved_nuboards/curve_two_vehicle.nuboard"
    "$AB_ROOT/saved_nuboards/multi_vehicle.nuboard"
    "$AB_ROOT/saved_nuboards/pedestrian.nuboard"
)
for file in "${files[@]}"; do
    if [[ ! -f "$file" ]]; then
        printf 'Missing NuBoard file: %s\nRun the four Flow-Planner experiments first.\n' "$file" >&2
        exit 2
    fi
done

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$(IFS=,; echo "${files[*]}")]" \
    "$@"

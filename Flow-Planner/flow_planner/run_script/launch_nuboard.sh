#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$(dirname -- "$PROJECT_ROOT")/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/flow_planner_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flow_planner_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

NUBOARD_FILE="${NUBOARD_FILE:-}"
if (( $# > 0 )) && [[ "$1" != *=* ]]; then
    NUBOARD_FILE="$1"
    shift
fi

if [[ -z "$NUBOARD_FILE" ]]; then
    NUBOARD_FILE="$(find "$NUPLAN_EXP_ROOT" -type f -name '*.nuboard' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$NUBOARD_FILE" || ! -f "$NUBOARD_FILE" ]]; then
    printf 'Error: no NuBoard file found. Pass one as the first argument or set NUBOARD_FILE.\n' >&2
    exit 2
fi

printf 'Opening NuBoard result: %s\n' "$NUBOARD_FILE"
exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    simulation_path="[$NUBOARD_FILE]" \
    "$@"

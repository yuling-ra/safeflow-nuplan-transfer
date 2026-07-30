#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAFEFLOW_PROJECT_ROOT="${SAFEFLOW_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SAFEFLOW_REPO="${SAFEFLOW_REPO:-$(cd -- "$SAFEFLOW_PROJECT_ROOT/.." && pwd)}"
WORKSPACE_ROOT="$(cd -- "$SAFEFLOW_REPO/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    DEFAULT_PYTHON_BIN="/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python"
    if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
        PYTHON_BIN="$DEFAULT_PYTHON_BIN"
    else
        PYTHON_BIN="$(command -v python3 || command -v python || true)"
    fi
fi

export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$WORKSPACE_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_matplotlib}"
export PYTHONPATH="$SAFEFLOW_PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Error: Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -d "$NUPLAN_DEVKIT_ROOT" ]]; then
    printf 'Error: nuPlan devkit directory does not exist: %s\n' "$NUPLAN_DEVKIT_ROOT" >&2
    exit 2
fi

NUBOARD_FILE="${NUBOARD_FILE:-}"
if (( $# > 0 )) && [[ "$1" != -* && "$1" != *=* ]]; then
    NUBOARD_FILE="$1"
    shift
fi

if [[ -z "$NUBOARD_FILE" ]]; then
    NUBOARD_FILE="$(find "$NUPLAN_EXP_ROOT" -type f -name '*.nuboard' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "$NUBOARD_FILE" || ! -f "$NUBOARD_FILE" ]]; then
    printf 'Error: no SafeFlow NuBoard file found under %s. Run launch_sim_safeflow.sh first.\n' \
        "$NUPLAN_EXP_ROOT" >&2
    exit 2
fi

mkdir -p "$MPLCONFIGDIR"
printf 'Opening SafeFlow NuBoard result: %s\n' "$NUBOARD_FILE"
exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$NUBOARD_FILE]" \
    "$@"

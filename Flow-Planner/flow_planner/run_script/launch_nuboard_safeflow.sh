#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}"

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
    printf 'Error: no SafeFlow NuBoard file found under %s. Run launch_sim_safeflow.sh first.\n' \
        "$NUPLAN_EXP_ROOT" >&2
    exit 2
fi

exec "$SCRIPT_DIR/launch_nuboard.sh" "$NUBOARD_FILE" "$@"

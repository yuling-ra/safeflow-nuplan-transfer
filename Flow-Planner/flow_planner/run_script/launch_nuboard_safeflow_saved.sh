#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$(dirname -- "$PROJECT_ROOT")/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

STRAIGHT_NUBOARD="$SCRIPT_DIR/saved_nuboards/single_agent_straight.nuboard"
PRE_SINGLE_AGENT_LEFT_TURN_NUBOARD="$SCRIPT_DIR/saved_nuboards/moving_multiagent_left_turn.nuboard"
COMBINED_AVOIDANCE_NUBOARD="$SCRIPT_DIR/saved_nuboards/three_vehicle_combined_avoidance.nuboard"

for nuboard_file in "$STRAIGHT_NUBOARD" "$PRE_SINGLE_AGENT_LEFT_TURN_NUBOARD" "$COMBINED_AVOIDANCE_NUBOARD"; do
    if [[ ! -f "$nuboard_file" ]]; then
        printf 'Error: saved NuBoard file does not exist: %s\n' "$nuboard_file" >&2
        exit 2
    fi
done

printf 'Opening saved SafeFlow NuBoard results:\n'
printf '  [1] single-agent straight:                  %s\n' "$STRAIGHT_NUBOARD"
printf '  [2] moving multi-agent left turn:           %s\n' "$PRE_SINGLE_AGENT_LEFT_TURN_NUBOARD"
printf '  [3] combined three-vehicle avoidance:       %s\n' "$COMBINED_AVOIDANCE_NUBOARD"
printf '\n'

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
    "simulation_path=[$STRAIGHT_NUBOARD,$PRE_SINGLE_AGENT_LEFT_TURN_NUBOARD,$COMBINED_AVOIDANCE_NUBOARD]" \
    "$@"

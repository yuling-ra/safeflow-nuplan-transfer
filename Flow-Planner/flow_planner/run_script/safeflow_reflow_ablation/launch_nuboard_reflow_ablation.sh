#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
YULING_ROOT="$(cd -- "$RUN_SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$YULING_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_reflow_matplotlib}"
export PYTHONPATH="$RUN_SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

files=(
  "$SCRIPT_DIR/saved_nuboards/reflow_straight_overtake.nuboard"
  "$SCRIPT_DIR/saved_nuboards/reflow_curve_two_vehicle.nuboard"
  "$SCRIPT_DIR/saved_nuboards/reflow_multi_vehicle.nuboard"
  "$SCRIPT_DIR/saved_nuboards/reflow_right_turn_pedestrian.nuboard"
  "$SCRIPT_DIR/saved_nuboards/reflow_roundabout_turn.nuboard"
)
for file in "${files[@]}"; do
  if [[ ! -f "$file" ]]; then
    printf 'Missing ReFlow NuBoard file: %s\nRun the corresponding simulation first.\n' "$file" >&2
    exit 2
  fi
done

exec "$PYTHON_BIN" -m nuplan.planning.script.run_nuboard \
  "simulation_path=[$(IFS=,; echo "${files[*]}")]" \
  "$@"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_NAME="${1:?experiment name is required}"
shift

export PLANNER_DEVICE="${PLANNER_DEVICE:-cpu}"
export PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/flow_planner_ablation_exp}"
export CONFIG_FILE="${CONFIG_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/flow_planner_ckpt/model_config.yaml}"
export CKPT_FILE="${CKPT_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/flow_planner_ckpt/model.pth}"
export FLOW_PLANNER_ABLATION_ROOT="${FLOW_PLANNER_ABLATION_ROOT:-$SCRIPT_DIR}"
export FLOW_PLANNER_CFG_WEIGHT="${FLOW_PLANNER_CFG_WEIGHT:-1.8}"

mkdir -p "$FLOW_PLANNER_ABLATION_ROOT/saved_nuboards"

# Build a runtime config so the device and model-internal CFG guidance reach
# FlowODE. The planner wrapper's cfg_weight argument is not consumed by the
# current FlowODE implementation.
runtime_config="$(mktemp /tmp/flow_planner_ablation_config.XXXXXX.yaml)"
trap 'rm -f "$runtime_config"' EXIT
"$PYTHON_BIN" - "$CONFIG_FILE" "$runtime_config" "$PLANNER_DEVICE" "$FLOW_PLANNER_CFG_WEIGHT" <<'PY'
import sys

from omegaconf import OmegaConf

source, destination, device, cfg_weight = sys.argv[1:]
config = OmegaConf.load(source)
config.device = device
config.model.device = device
config.model.cfg_weight = float(cfg_weight)
OmegaConf.save(config, destination)
PY
export CONFIG_FILE="$runtime_config"

set +e
"$RUN_SCRIPT_DIR/launch_sim_nuplan.sh" "$@"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    latest_nuboard="$(find "$NUPLAN_EXP_ROOT" -type f -name '*.nuboard' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$latest_nuboard" && -f "$latest_nuboard" ]]; then
        cp -- "$latest_nuboard" "$FLOW_PLANNER_ABLATION_ROOT/saved_nuboards/${EXPERIMENT_NAME}.nuboard"
        printf 'Saved Flow-Planner NuBoard: %s\n' \
            "$FLOW_PLANNER_ABLATION_ROOT/saved_nuboards/${EXPERIMENT_NAME}.nuboard"
    else
        printf 'Flow-Planner simulation completed without a NuBoard file under %s\n' "$NUPLAN_EXP_ROOT" >&2
    fi
fi

exit "$status"

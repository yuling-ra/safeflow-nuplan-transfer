#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_PYTHON_BIN="/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Error: Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flow_planner_matplotlib}"

###################################
# User Configuration Section
###################################
# Defaults match the local mini dataset. Export a variable to override it.
LOCAL_NUPLAN_ROOT="$(dirname -- "$PROJECT_ROOT")/nuplan-devkit"
WORKING_FLOW_PLANNER_ROOT="$LOCAL_NUPLAN_ROOT/Flow-Planner"
FLOW_PLANNER_ROOT="${FLOW_PLANNER_ROOT:-$WORKING_FLOW_PLANNER_ROOT}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$LOCAL_NUPLAN_ROOT}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$LOCAL_NUPLAN_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$LOCAL_NUPLAN_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/flow_planner_exp}"

# The mini smoke test uses a DB containing both requested scenario types.
# Other supported values: "val14", "test14-random", "test14-hard".
SPLIT="${SPLIT:-all_scenarios}"
MINI_DB_FILE="${MINI_DB_FILE:-$NUPLAN_DATA_ROOT/data/cache/mini/2021.06.07.18.53.26_veh-26_00005_00427.db}"
SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
SCENARIO_TOKENS="${SCENARIO_TOKENS:-297baa6f81b05d01}"
MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
SCENARIOS_PER_TYPE="${SCENARIOS_PER_TYPE:-1}"
SCENARIO_LIMIT="${SCENARIO_LIMIT:-1}"
PLANNER_DEVICE="${PLANNER_DEVICE:-cuda}"
WORKER="${WORKER:-sequential}"
export SINGLE_AGENT_SCENARIO="${SINGLE_AGENT_SCENARIO:-true}"
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-297baa6f81b05d01}"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
###################################


BRANCH_NAME=flow_planner_release
CONFIG_FILE="${CONFIG_FILE:-}"
CKPT_FILE="${CKPT_FILE:-}"

for checkpoint_dir in \
    "$PROJECT_ROOT/checkpoints/flow_planner" \
    "$WORKING_FLOW_PLANNER_ROOT/checkpoints/flow_planner" \
    "$(dirname -- "$(dirname -- "$PROJECT_ROOT")")/flow_planner_ckpt"; do
    if [[ -z "$CONFIG_FILE" && -f "$checkpoint_dir/model_config.yaml" ]]; then
        CONFIG_FILE="$checkpoint_dir/model_config.yaml"
    fi
    if [[ -z "$CKPT_FILE" && -f "$checkpoint_dir/model.pth" ]]; then
        CKPT_FILE="$checkpoint_dir/model.pth"
    fi
done

missing_vars=()
for var_name in NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT SPLIT CHALLENGE CONFIG_FILE CKPT_FILE; do
    if [[ -z "${!var_name}" ]]; then
        missing_vars+=("$var_name")
    fi
done

for dir_path in "$FLOW_PLANNER_ROOT" "$NUPLAN_DATA_ROOT" "$NUPLAN_MAPS_ROOT"; do
    if [[ ! -d "$dir_path" ]]; then
        printf 'Error: required directory does not exist: %s\n' "$dir_path" >&2
        exit 2
    fi
done

if (( ${#missing_vars[@]} > 0 )); then
    printf 'Error: configure these variables before launching the simulation:\n' >&2
    printf '  %s\n' "${missing_vars[@]}" >&2
    exit 2
fi

export PYTHONPATH="$FLOW_PLANNER_ROOT:$PROJECT_ROOT${NUPLAN_DEVKIT_ROOT:+:$NUPLAN_DEVKIT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON_BIN" -c 'import importlib.util, flow_planner.planner; assert importlib.util.find_spec("nuplan.planning.script.run_simulation")' 2>/dev/null; then
    printf 'Error: flow_planner or nuplan-devkit is not importable by %s.\n' "$PYTHON_BIN" >&2
    exit 2
fi

for file_path in "$CONFIG_FILE" "$CKPT_FILE"; do
    if [[ ! -f "$file_path" ]]; then
        printf 'Error: required file does not exist: %s\n' "$file_path" >&2
        exit 2
    fi
done

scenario_args=()
case "$SPLIT" in
    all_scenarios)
        SCENARIO_BUILDER="nuplan_mini"
        if [[ ! -f "$MINI_DB_FILE" ]]; then
            printf 'Error: mini database file does not exist: %s\n' "$MINI_DB_FILE" >&2
            exit 2
        fi
        scenario_args+=("scenario_builder.db_files=$MINI_DB_FILE")
        scenario_args+=("scenario_filter.scenario_types=[$SCENARIO_TYPES]")
        if [[ -n "$SCENARIO_TOKENS" ]]; then
            scenario_args+=("scenario_filter.scenario_tokens=[$SCENARIO_TOKENS]")
        fi
        if [[ -n "$MAP_NAMES" ]]; then
            scenario_args+=("scenario_filter.map_names=[$MAP_NAMES]")
        fi
        scenario_args+=("scenario_filter.num_scenarios_per_type=$SCENARIOS_PER_TYPE")
        scenario_args+=("scenario_filter.limit_total_scenarios=$SCENARIO_LIMIT")
        ;;
    val14)
        SCENARIO_BUILDER="nuplan"
        ;;
    test14-random|test14-hard)
        SCENARIO_BUILDER="nuplan_challenge"
        ;;
    *)
        printf 'Error: unsupported SPLIT: %s\n' "$SPLIT" >&2
        exit 2
        ;;
esac

if [[ "$PLANNER_DEVICE" == "cuda" ]] && ! "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    printf 'Error: CUDA is unavailable. Run on a GPU host or set PLANNER_DEVICE=cpu for a slow smoke test.\n' >&2
    exit 2
fi

CONFIG_FILE_FOR_RUN="$CONFIG_FILE"
if [[ "$PLANNER_DEVICE" != "cuda" ]]; then
    mkdir -p "$NUPLAN_EXP_ROOT"
    CONFIG_FILE_FOR_RUN="$(mktemp "$NUPLAN_EXP_ROOT/flow_planner_config.XXXXXX.yaml")"
    trap 'rm -f "$CONFIG_FILE_FOR_RUN"' EXIT
    "$PYTHON_BIN" - "$CONFIG_FILE" "$CONFIG_FILE_FOR_RUN" "$PLANNER_DEVICE" <<'PY'
import sys

from omegaconf import OmegaConf

source, destination, device = sys.argv[1:]
config = OmegaConf.load(source)
config.device = device
OmegaConf.save(config, destination)
PY
fi

worker_args=()
case "$WORKER" in
    sequential)
        worker_args+=("worker=sequential")
        worker_args+=("number_of_cpus_allocated_per_simulation=1")
        worker_args+=("number_of_gpus_allocated_per_simulation=0")
        ;;
    ray_distributed)
        worker_args+=("worker=ray_distributed")
        worker_args+=("worker.threads_per_node=${WORKER_THREADS:-64}")
        worker_args+=("number_of_gpus_allocated_per_simulation=${GPUS_PER_SIMULATION:-0.15}")
        ;;
    *)
        printf 'Error: unsupported WORKER: %s\n' "$WORKER" >&2
        exit 2
        ;;
esac

echo "Processing $CKPT_FILE..."
echo "Scenario filter: $SCENARIO_TYPES (up to $SCENARIOS_PER_TYPE per type)"
echo "Scenario tokens: ${SCENARIO_TOKENS:-<none>}"
echo "Single-agent simplification: $SINGLE_AGENT_SCENARIO (${SINGLE_AGENT_SCENARIO_TOKENS:-all selected scenarios})"
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=flow_planner

"$PYTHON_BIN" "$SCRIPT_DIR/run_simulation.py" \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.flow_planner.config_path=$CONFIG_FILE_FOR_RUN \
    planner.flow_planner.ckpt_path=$CKPT_FILE \
    planner.flow_planner.device=$PLANNER_DEVICE \
    scenario_builder=$SCENARIO_BUILDER \
    "${scenario_args[@]}" \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    "${worker_args[@]}" \
    distributed_mode='SINGLE_NODE' \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://flow_planner.nuplan_simulation.scenario_filter, pkg://flow_planner.nuplan_simulation, pkg://flow_planner.script, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    "$@"

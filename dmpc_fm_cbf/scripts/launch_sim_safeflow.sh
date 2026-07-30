#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SAFEFLOW_PROJECT_ROOT="${SAFEFLOW_PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SAFEFLOW_REPO="${SAFEFLOW_REPO:-$(cd -- "$SAFEFLOW_PROJECT_ROOT/.." && pwd)}"
WORKSPACE_ROOT="$(cd -- "$SAFEFLOW_REPO/.." && pwd)"
CONFIG_ROOT="$SAFEFLOW_PROJECT_ROOT/config"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    DEFAULT_PYTHON_BIN="/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python"
    if [[ -x "$DEFAULT_PYTHON_BIN" ]]; then
        PYTHON_BIN="$DEFAULT_PYTHON_BIN"
    else
        PYTHON_BIN="$(command -v python3 || command -v python || true)"
    fi
fi

CKPT_FILE="${CKPT_FILE:-$SAFEFLOW_PROJECT_ROOT/notebooks/cache/model_vel_5ch_canonical_r.pt}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$WORKSPACE_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_matplotlib}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$SAFEFLOW_PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
    printf 'Error: Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi

for path in "$SAFEFLOW_PROJECT_ROOT" "$CONFIG_ROOT" "$NUPLAN_DEVKIT_ROOT" "$NUPLAN_DATA_ROOT" "$NUPLAN_MAPS_ROOT"; do
    if [[ ! -d "$path" ]]; then
        printf 'Error: required directory does not exist: %s\n' "$path" >&2
        exit 2
    fi
done

if [[ ! -f "$CKPT_FILE" ]]; then
    printf 'Error: SafeFlow checkpoint does not exist: %s\n' "$CKPT_FILE" >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c 'import dmpc_fm_cbf, fmtorch, nuplan; from dmpc_fm_cbf.safeflow_nuplan_planner import SafeFlowNuPlanPlanner' 2>/dev/null; then
    printf 'Error: SafeFlow, nuPlan, or the planner adapter is not importable by %s.\n' "$PYTHON_BIN" >&2
    exit 2
fi

PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"
if [[ "$PLANNER_DEVICE" == "auto" ]]; then
    if "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        PLANNER_DEVICE=cuda
    else
        PLANNER_DEVICE=cpu
    fi
fi
if [[ "$PLANNER_DEVICE" != "cpu" && "$PLANNER_DEVICE" != "cuda" ]]; then
    printf 'Error: PLANNER_DEVICE must be auto, cpu, or cuda; got %s\n' "$PLANNER_DEVICE" >&2
    exit 2
fi

MINI_DB_FILE="${MINI_DB_FILE:-$NUPLAN_DATA_ROOT/data/cache/mini/2021.06.07.18.53.26_veh-26_00005_00427.db}"
SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn,starting_straight_traffic_light_intersection_traversal}"
SCENARIOS_PER_TYPE="${SCENARIOS_PER_TYPE:-1}"
SCENARIO_LIMIT="${SCENARIO_LIMIT:-2}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
FM_NUM_SEGMENTS="${FM_NUM_SEGMENTS:-8}"
ODE_METHOD="${ODE_METHOD:-rk4}"
MODEL_TRAJECTORY_STEPS="${MODEL_TRAJECTORY_STEPS:-60}"
USE_CBF="${USE_CBF:-true}"

case "$USE_CBF" in
    true)
        PLANNER_VARIANT=fm_cbf
        ;;
    false)
        PLANNER_VARIANT=pure_fm
        ;;
    *)
        printf 'Error: USE_CBF must be true or false; got %s\n' "$USE_CBF" >&2
        exit 2
        ;;
esac

if [[ ! -f "$MINI_DB_FILE" ]]; then
    printf 'Error: mini database does not exist: %s\n' "$MINI_DB_FILE" >&2
    exit 2
fi

mkdir -p "$NUPLAN_EXP_ROOT" "$MPLCONFIGDIR"

checkpoint_name="$(basename -- "$CKPT_FILE")"
checkpoint_name="${checkpoint_name%.*}"
experiment_uid="safeflow/all_scenarios/${PLANNER_VARIANT}/${checkpoint_name}_$(date '+%Y-%m-%d-%H-%M-%S')"

printf 'SafeFlow checkpoint: %s\n' "$CKPT_FILE"
printf 'Device: %s, scenarios: %s, limit: %s\n' "$PLANNER_DEVICE" "$SCENARIO_TYPES" "$SCENARIO_LIMIT"
printf 'Experiment root: %s\n' "$NUPLAN_EXP_ROOT"

simulation_command=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_nuplan_simulation.py"
    "+simulation=$CHALLENGE"
    "planner=safeflow_planner"
    "planner.safeflow_planner.ckpt_path=$CKPT_FILE"
    "planner.safeflow_planner.safeflow_project_root=$SAFEFLOW_PROJECT_ROOT"
    "planner.safeflow_planner.device=$PLANNER_DEVICE"
    "planner.safeflow_planner.fm_num_segments=$FM_NUM_SEGMENTS"
    "planner.safeflow_planner.ode_method=$ODE_METHOD"
    "planner.safeflow_planner.model_trajectory_steps=$MODEL_TRAJECTORY_STEPS"
    "planner.safeflow_planner.use_cbf=$USE_CBF"
    "scenario_builder=nuplan_mini"
    "scenario_builder.db_files=$MINI_DB_FILE"
    "scenario_filter=all_scenarios"
    "scenario_filter.scenario_types=[$SCENARIO_TYPES]"
    "scenario_filter.num_scenarios_per_type=$SCENARIOS_PER_TYPE"
    "scenario_filter.limit_total_scenarios=$SCENARIO_LIMIT"
    "experiment_uid=$experiment_uid"
    "worker=sequential"
    "number_of_cpus_allocated_per_simulation=1"
    "number_of_gpus_allocated_per_simulation=0"
    "distributed_mode=SINGLE_NODE"
    "enable_simulation_progress_bar=true"
    "verbose=true"
    "hydra.searchpath=[file://$CONFIG_ROOT,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${simulation_command[@]}" "$@"
    printf '\n'
    exit 0
fi

exec "${simulation_command[@]}" "$@"

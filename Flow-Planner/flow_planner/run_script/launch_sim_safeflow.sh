#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
YULING_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
SAFEFLOW_REPO="${SAFEFLOW_REPO:-$YULING_ROOT/safeflow-nuplan-transfer}"
SAFEFLOW_PROJECT_ROOT="${SAFEFLOW_PROJECT_ROOT:-$SAFEFLOW_REPO/dmpc_fm_cbf}"
CKPT_FILE="${CKPT_FILE:-$SAFEFLOW_PROJECT_ROOT/notebooks/cache/model_vel_5ch_canonical_r.pt}"

export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$YULING_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/safeflow_matplotlib}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="$SCRIPT_DIR:$SAFEFLOW_PROJECT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Error: Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi

for path in "$SAFEFLOW_PROJECT_ROOT" "$NUPLAN_DEVKIT_ROOT" "$NUPLAN_DATA_ROOT" "$NUPLAN_MAPS_ROOT"; do
    if [[ ! -d "$path" ]]; then
        printf 'Error: required directory does not exist: %s\n' "$path" >&2
        exit 2
    fi
done

if [[ ! -f "$CKPT_FILE" ]]; then
    printf 'Error: SafeFlow checkpoint does not exist: %s\n' "$CKPT_FILE" >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c 'import dmpc_fm_cbf, fmtorch, nuplan, safeflow_nuplan_planner' 2>/dev/null; then
    printf 'Error: SafeFlow, nuPlan, or the planner adapter is not importable.\n' >&2
    exit 2
fi

PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"
if [[ "$PLANNER_DEVICE" == "auto" ]]; then
    if "$PYTHON_BIN" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
        PLANNER_DEVICE=cuda
    else
        PLANNER_DEVICE=cpu
    fi
fi
if [[ "$PLANNER_DEVICE" != "cpu" && "$PLANNER_DEVICE" != "cuda" ]]; then
    printf 'Error: PLANNER_DEVICE must be auto, cpu, or cuda; got %s\n' "$PLANNER_DEVICE" >&2
    exit 2
fi

MINI_DB_FILE="${MINI_DB_FILE:-$NUPLAN_DATA_ROOT/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
SCENARIOS_PER_TYPE="${SCENARIOS_PER_TYPE:-1}"
SCENARIO_LIMIT="${SCENARIO_LIMIT:-1}"
CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
export SINGLE_AGENT_SCENARIO="${SINGLE_AGENT_SCENARIO:-true}"
export SINGLE_AGENT_SCENARIO_TOKENS="${SINGLE_AGENT_SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export SINGLE_AGENT_VEHICLE_SPEED_SCALE="${SINGLE_AGENT_VEHICLE_SPEED_SCALE:-0.35}"
export SINGLE_AGENT_VEHICLE_POSITION_SCALE="${SINGLE_AGENT_VEHICLE_POSITION_SCALE:-$SINGLE_AGENT_VEHICLE_SPEED_SCALE}"
export SINGLE_AGENT_VEHICLE_STRAIGHT="${SINGLE_AGENT_VEHICLE_STRAIGHT:-true}"
FM_NUM_SEGMENTS="${FM_NUM_SEGMENTS:-8}"
ODE_METHOD="${ODE_METHOD:-rk4}"
MODEL_TRAJECTORY_STEPS="${MODEL_TRAJECTORY_STEPS:-60}"
USE_CBF="${USE_CBF:-true}"

if [[ ! -f "$MINI_DB_FILE" ]]; then
    printf 'Error: mini database does not exist: %s\n' "$MINI_DB_FILE" >&2
    exit 2
fi

mkdir -p "$NUPLAN_EXP_ROOT"

checkpoint_name="$(basename -- "$CKPT_FILE")"
checkpoint_name="${checkpoint_name%.*}"
experiment_uid="safeflow/all_scenarios/fm_cbf/${checkpoint_name}_$(date '+%Y-%m-%d-%H-%M-%S')"

printf 'SafeFlow checkpoint: %s\n' "$CKPT_FILE"
printf 'Device: %s, scenarios: %s, limit: %s\n' "$PLANNER_DEVICE" "$SCENARIO_TYPES" "$SCENARIO_LIMIT"
printf 'Scenario tokens: %s\n' "${SCENARIO_TOKENS:-<none>}"
printf 'Map names: %s\n' "${MAP_NAMES:-<none>}"
if [[ "${SYNTHETIC_THREE_VEHICLE_SCENARIO:-false}" == "true" ]]; then
    printf 'Synthetic three-vehicle avoidance: enabled (%s)\n' "${SYNTHETIC_AVOIDANCE_MODE:-straight_lane_change}"
    printf 'Obstacle route progress: %sm, %sm\n' \
        "${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-default}" "${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-default}"
elif [[ "${THREE_VEHICLE_SCENARIO:-false}" == "true" ]]; then
    printf 'Strict three-vehicle scenario: enabled (ego + oncoming + left-turn)\n'
    printf 'Three-vehicle role tokens: oncoming=%s, left_turn=%s\n' \
        "${THREE_VEHICLE_ONCOMING_TOKEN:-auto}" "${THREE_VEHICLE_LEFT_TURN_TOKEN:-auto}"
else
    printf 'Single-agent simplification: %s (%s)\n' "$SINGLE_AGENT_SCENARIO" "${SINGLE_AGENT_SCENARIO_TOKENS:-all selected scenarios}"
    printf 'Single-agent vehicle speed scale: %s, position scale: %s, straight: %s\n' "$SINGLE_AGENT_VEHICLE_SPEED_SCALE" "$SINGLE_AGENT_VEHICLE_POSITION_SCALE" "$SINGLE_AGENT_VEHICLE_STRAIGHT"
fi

scenario_args=()
if [[ -n "$SCENARIO_TOKENS" ]]; then
    scenario_args+=("scenario_filter.scenario_tokens=[$SCENARIO_TOKENS]")
fi
if [[ -n "$MAP_NAMES" ]]; then
    scenario_args+=("scenario_filter.map_names=[$MAP_NAMES]")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/run_simulation.py" \
    +simulation="$CHALLENGE" \
    planner=safeflow_planner \
    planner.safeflow_planner.ckpt_path="$CKPT_FILE" \
    planner.safeflow_planner.safeflow_project_root="$SAFEFLOW_PROJECT_ROOT" \
    planner.safeflow_planner.device="$PLANNER_DEVICE" \
    planner.safeflow_planner.fm_num_segments="$FM_NUM_SEGMENTS" \
    planner.safeflow_planner.ode_method="$ODE_METHOD" \
    planner.safeflow_planner.model_trajectory_steps="$MODEL_TRAJECTORY_STEPS" \
    planner.safeflow_planner.use_cbf="$USE_CBF" \
    scenario_builder=nuplan_mini \
    scenario_builder.db_files="$MINI_DB_FILE" \
    scenario_filter=all_scenarios \
    "scenario_filter.scenario_types=[$SCENARIO_TYPES]" \
    "${scenario_args[@]}" \
    scenario_filter.num_scenarios_per_type="$SCENARIOS_PER_TYPE" \
    scenario_filter.limit_total_scenarios="$SCENARIO_LIMIT" \
    experiment_uid="$experiment_uid" \
    worker=sequential \
    number_of_cpus_allocated_per_simulation=1 \
    number_of_gpus_allocated_per_simulation=0 \
    distributed_mode=SINGLE_NODE \
    enable_simulation_progress_bar=true \
    verbose=true \
    "hydra.searchpath=[file://$SCRIPT_DIR,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
    "$@"

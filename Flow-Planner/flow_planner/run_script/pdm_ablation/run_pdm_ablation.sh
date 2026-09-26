#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_NAME="${1:?experiment name is required}"
shift

export PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$(cd -- "$RUN_SCRIPT_DIR/../.." && pwd)/../nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
PDM_COMPLEX_MODE="${PDM_COMPLEX_MODE:-true}"
export PDM_COMPLEX_MODE
if [[ -z "${NUPLAN_EXP_ROOT:-}" ]]; then
    if [[ "$PDM_COMPLEX_MODE" == "true" ]]; then
        export NUPLAN_EXP_ROOT="/tmp/pdm_complex_ablation_exp"
    else
        export NUPLAN_EXP_ROOT="/tmp/pdm_ablation_exp"
    fi
else
    export NUPLAN_EXP_ROOT
fi
export MINI_DB_FILE="${MINI_DB_FILE:-$NUPLAN_DATA_ROOT/data/cache/mini/2021.06.07.18.53.26_veh-26_00005_00427.db}"
export SCENARIOS_PER_TYPE="${SCENARIOS_PER_TYPE:-1}"
export SCENARIO_LIMIT="${SCENARIO_LIMIT:-1}"
export SPLIT="${SPLIT:-all_scenarios}"
export CHALLENGE="${CHALLENGE:-closed_loop_nonreactive_agents}"
export WORKER="${WORKER:-sequential}"
export PLANNER_DEVICE="cpu"

mkdir -p "$NUPLAN_EXP_ROOT" "$SCRIPT_DIR/saved_nuboards"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Error: Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi

export PYTHONPATH="$SCRIPT_DIR:$RUN_SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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

scenario_args=(
    "scenario_builder.db_files=$MINI_DB_FILE"
    "scenario_filter.scenario_types=[$SCENARIO_TYPES]"
    "scenario_filter.num_scenarios_per_type=$SCENARIOS_PER_TYPE"
    "scenario_filter.limit_total_scenarios=$SCENARIO_LIMIT"
)
if [[ -n "${SCENARIO_TOKENS:-}" ]]; then
    scenario_args+=("scenario_filter.scenario_tokens=[$SCENARIO_TOKENS]")
fi
if [[ -n "${MAP_NAMES:-}" ]]; then
    scenario_args+=("scenario_filter.map_names=[$MAP_NAMES]")
fi

# Complex PDM evaluates substantially more real candidates and motion costs:
# 9 lateral paths x 9 IDM policies over a 6-second proposal horizon. Set
# PDM_COMPLEX_MODE=false to reproduce the original 15-proposal baseline.
pdm_complex_overrides=()
if [[ "$PDM_COMPLEX_MODE" == "true" ]]; then
    pdm_complex_overrides+=(
        "planner.pdm_closed_planner.proposal_sampling.num_poses=${PDM_PROPOSAL_POSES:-60}"
        "planner.pdm_closed_planner.lateral_offsets=${PDM_LATERAL_OFFSETS:-[-3.5,-2.5,-1.5,-0.75,0.75,1.5,2.5,3.5]}"
        "planner.pdm_closed_planner.idm_policies.speed_limit_fraction=${PDM_SPEED_FRACTIONS:-[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}"
        "planner.pdm_closed_planner.idm_policies.fallback_target_velocity=${PDM_FALLBACK_TARGET_VELOCITY:-15.0}"
        "planner.pdm_closed_planner.complex_cost_weight=${PDM_COMPLEX_COST_WEIGHT:-0.20}"
        "planner.pdm_closed_planner.smoothness_cost_weight=${PDM_SMOOTHNESS_COST_WEIGHT:-1.0}"
        "planner.pdm_closed_planner.curvature_cost_weight=${PDM_CURVATURE_COST_WEIGHT:-1.0}"
        "planner.pdm_closed_planner.acceleration_cost_weight=${PDM_ACCELERATION_COST_WEIGHT:-1.0}"
        "planner.pdm_closed_planner.jerk_cost_weight=${PDM_JERK_COST_WEIGHT:-1.0}"
        "planner.pdm_closed_planner.lateral_acceleration_cost_weight=${PDM_LATERAL_ACCELERATION_COST_WEIGHT:-1.0}"
    )
fi

echo "Planner: PDMClosedPlanner"
echo "Scenario filter: $SCENARIO_TYPES (up to $SCENARIOS_PER_TYPE per type)"
echo "Scenario tokens: ${SCENARIO_TOKENS:-<none>}"
echo "Synthetic three-vehicle scenario: ${SYNTHETIC_THREE_VEHICLE_SCENARIO:-false}"
if [[ "$PDM_COMPLEX_MODE" == "true" ]]; then
    echo "Complex PDM: enabled (81 proposals, ${PDM_PROPOSAL_POSES:-60}-pose proposal horizon, motion-quality cost)"
else
    echo "Complex PDM: disabled (baseline proposal and scoring configuration)"
fi

set +e
"$PYTHON_BIN" "$RUN_SCRIPT_DIR/run_simulation.py" \
    +simulation="$CHALLENGE" \
    planner=pdm_closed_planner \
    scenario_builder=nuplan_mini \
    "${scenario_args[@]}" \
    scenario_filter="$SPLIT" \
    experiment_uid=pdm_closed_planner/$SPLIT/pdm_ablation/PDMClosedPlanner_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    "${worker_args[@]}" \
    distributed_mode='SINGLE_NODE' \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.simulation,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
    "$@" \
    "${pdm_complex_overrides[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    latest_nuboard="$(find "$NUPLAN_EXP_ROOT" -type f -name '*.nuboard' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$latest_nuboard" && -f "$latest_nuboard" ]]; then
        result_name="${EXPERIMENT_NAME}.nuboard"
        if [[ "$PDM_COMPLEX_MODE" == "true" ]]; then
            result_name="complex_${result_name}"
        fi
        cp -- "$latest_nuboard" "$SCRIPT_DIR/saved_nuboards/$result_name"
        printf 'Saved PDM NuBoard: %s\n' "$SCRIPT_DIR/saved_nuboards/$result_name"
    else
        printf 'PDM simulation completed without a NuBoard file under %s\n' "$NUPLAN_EXP_ROOT" >&2
    fi
fi

exit "$status"

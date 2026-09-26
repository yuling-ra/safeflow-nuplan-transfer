#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
YULING_ROOT="$(cd -- "$RUN_SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export NUPLAN_DEVKIT_ROOT="${NUPLAN_DEVKIT_ROOT:-$YULING_ROOT/nuplan-devkit}"
export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-v1.1_mini}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$NUPLAN_DEVKIT_ROOT/nuplan-maps-v1.0/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/mpc_cbf_ablation_exp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpc_cbf_matplotlib}"
export PYTHONPATH="$SCRIPT_DIR:$RUN_SCRIPT_DIR:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENT_NAME="${1:?experiment name is required}"
shift
SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
MINI_DB_FILE="${MINI_DB_FILE:-$NUPLAN_DATA_ROOT/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
SCENARIOS_PER_TYPE="${SCENARIOS_PER_TYPE:-1}"
SCENARIO_LIMIT="${SCENARIO_LIMIT:-1}"

mkdir -p "$NUPLAN_EXP_ROOT"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi

printf 'Planner: MPC-CBF\n'
printf 'Horizon: %s steps, obstacle radius: %sm, margin: %sm\n' \
    "${MPC_HORIZON_STEPS:-12}" "${MPC_OBSTACLE_QUERY_RADIUS_M:-45.0}" "${MPC_OBSTACLE_MARGIN_M:-0.5}"
printf 'Tracking-error margin: %sm, CBF gamma: %s\n' \
    "${MPC_TRACKING_ERROR_MARGIN_M:-0.75}" "${MPC_CBF_GAMMA:-0.2}"

set +e
"$PYTHON_BIN" "$RUN_SCRIPT_DIR/run_simulation.py" \
    +simulation=closed_loop_nonreactive_agents \
    planner=mpc_cbf_planner \
    planner.mpc_cbf_planner.trajectory_steps="${MPC_TRAJECTORY_STEPS:-13}" \
    planner.mpc_cbf_planner.mpc_horizon_steps="${MPC_HORIZON_STEPS:-12}" \
    planner.mpc_cbf_planner.max_agents="${MPC_MAX_AGENTS:-8}" \
    planner.mpc_cbf_planner.obstacle_query_radius_m="${MPC_OBSTACLE_QUERY_RADIUS_M:-45.0}" \
    planner.mpc_cbf_planner.obstacle_margin_m="${MPC_OBSTACLE_MARGIN_M:-0.5}" \
    planner.mpc_cbf_planner.tracking_error_margin_m="${MPC_TRACKING_ERROR_MARGIN_M:-0.75}" \
    planner.mpc_cbf_planner.obstacle_prediction_horizon_s="${MPC_OBSTACLE_PREDICTION_HORIZON_S:-2.0}" \
    planner.mpc_cbf_planner.cbf_gamma="${MPC_CBF_GAMMA:-0.2}" \
    planner.mpc_cbf_planner.solver_maxiter="${MPC_SOLVER_MAXITER:-35}" \
    scenario_builder=nuplan_mini \
    scenario_builder.db_files="$MINI_DB_FILE" \
    scenario_filter=all_scenarios \
    "scenario_filter.scenario_types=[$SCENARIO_TYPES]" \
    "scenario_filter.scenario_tokens=[$SCENARIO_TOKENS]" \
    "scenario_filter.map_names=[$MAP_NAMES]" \
    scenario_filter.num_scenarios_per_type="$SCENARIOS_PER_TYPE" \
    scenario_filter.limit_total_scenarios="$SCENARIO_LIMIT" \
    experiment_uid="mpc_cbf/all_scenarios/${EXPERIMENT_NAME}_$(date '+%Y-%m-%d-%H-%M-%S')" \
    worker=sequential \
    number_of_cpus_allocated_per_simulation=1 \
    number_of_gpus_allocated_per_simulation=0 \
    distributed_mode=SINGLE_NODE \
    enable_simulation_progress_bar=true \
    verbose=true \
    "hydra.searchpath=[file://$SCRIPT_DIR,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]" \
    "$@"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    result_root="$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/mpc_cbf/all_scenarios"
    latest_run="$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -name "${EXPERIMENT_NAME}_*" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    latest_nuboard="$(find "$latest_run" -maxdepth 1 -type f -name '*.nuboard' -print -quit 2>/dev/null)"
    if [[ -n "$latest_nuboard" ]]; then
        mkdir -p "$SCRIPT_DIR/saved_nuboards"
        cp -- "$latest_nuboard" "$SCRIPT_DIR/saved_nuboards/${EXPERIMENT_NAME}.nuboard"
        printf 'Saved MPC-CBF NuBoard: %s\n' "$SCRIPT_DIR/saved_nuboards/${EXPERIMENT_NAME}.nuboard"
    fi
fi
exit "$status"

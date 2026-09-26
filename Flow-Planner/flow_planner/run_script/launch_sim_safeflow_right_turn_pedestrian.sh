#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Real nuPlan right-turn route with two deterministic, opposing pedestrians
# crossing at the strongest right-turn point.  Existing vehicles are retained
# by the synthetic wrapper and pedestrians are added to the same detections.
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.10.11.02.57.41_veh-50_01522_02088.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_right_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-be47172809a953bd}"
export MAP_NAMES="${MAP_NAMES:-sg-one-north}"
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE=right_turn_pedestrian
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-120}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-30.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-40.0}"
export SYNTHETIC_PEDESTRIAN_SPEED_MPS="${SYNTHETIC_PEDESTRIAN_SPEED_MPS:-2.0}"
export SYNTHETIC_PEDESTRIAN_PROGRESS_M="${SYNTHETIC_PEDESTRIAN_PROGRESS_M:-7.0}"
export SYNTHETIC_PEDESTRIAN_SEED="${SYNTHETIC_PEDESTRIAN_SEED:-17}"
export SINGLE_AGENT_SCENARIO=false

set +e
"$SCRIPT_DIR/launch_sim_safeflow.sh" \
    planner.safeflow_planner.enable_overtake_demo=false \
    planner.safeflow_planner.force_overtake_demo=false \
    planner.safeflow_planner.use_cbf=true \
    planner.safeflow_planner.use_obstacle_free_reference=true \
    planner.safeflow_planner.ignore_traffic_lights_in_reference=false \
    planner.safeflow_planner.use_scenario_route_reference=true \
    planner.safeflow_planner.enable_route_avoidance_fallback=true \
    planner.safeflow_planner.max_agents=8 \
    planner.safeflow_planner.obstacle_query_radius_m=45.0 \
    planner.safeflow_planner.obstacle_margin_m=0.75 \
    planner.safeflow_planner.obstacle_prediction_horizon_s=1.5 \
    planner.safeflow_planner.obstacle_prediction_steps=4 \
    planner.safeflow_planner.avoidance_transition_distance_m=12.0 \
    planner.safeflow_planner.local_goal_distance_m=35.0 \
    "$@"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
    result_root="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}/exp/simulation/closed_loop_nonreactive_agents/safeflow/all_scenarios/fm_cbf"
    latest_run="$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -name 'model_vel_5ch_canonical_r_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    latest_nuboard="$(find "$latest_run" -maxdepth 1 -type f -name '*.nuboard' -print -quit 2>/dev/null)"
    latest_log="$(find "$latest_run/simulation_log" -type f -name '*.msgpack.xz' -print -quit 2>/dev/null)"
    if [[ -n "$latest_nuboard" && -n "$latest_log" ]]; then
        cp -- "$latest_nuboard" "$SCRIPT_DIR/saved_nuboards/right_turn_pedestrian_avoidance.nuboard"
        printf 'Saved right-turn pedestrian NuBoard: %s\n' "$SCRIPT_DIR/saved_nuboards/right_turn_pedestrian_avoidance.nuboard"
    fi
fi
exit "$status"

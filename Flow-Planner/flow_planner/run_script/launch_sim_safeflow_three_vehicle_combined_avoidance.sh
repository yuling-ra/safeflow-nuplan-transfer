#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# One experiment, three vehicles:
# Ego passes one slow vehicle on the straight and another in the left-hand curve.
export SYNTHETIC_AVOIDANCE_MODE=combined
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-150}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-24.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-34.0}"
export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-0.5}"
export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-2.5}"

set +e
"$SCRIPT_DIR/launch_sim_safeflow_synthetic_three_vehicle.sh" \
    planner.safeflow_planner.avoidance_transition_distance_m=8.0 \
    planner.safeflow_planner.follow_scenario_route_after_passing_agents=true \
    planner.safeflow_planner.route_follow_min_speed_mps=10.0 \
    planner.safeflow_planner.route_rejoin_distance_m=20.0 \
    planner.safeflow_planner.sequential_agent_overtake=false \
    planner.safeflow_planner.obstacle_query_radius_m=32.0 \
    planner.safeflow_planner.obstacle_margin_m=0.1 \
    planner.safeflow_planner.obstacle_prediction_horizon_s=1.0 \
    planner.safeflow_planner.obstacle_prediction_steps=2 \
    "$@"
status=$?
set -e

# Keep the saved NuBoard synchronized only with a completed simulation log.
if [[ "$status" -eq 0 ]]; then
    result_root="${NUPLAN_EXP_ROOT:-/tmp/safeflow_exp}/exp/simulation/closed_loop_nonreactive_agents/safeflow/all_scenarios/fm_cbf"
    latest_run="$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -name 'model_vel_5ch_canonical_r_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
    latest_nuboard="$(find "$latest_run" -maxdepth 1 -type f -name '*.nuboard' -print -quit 2>/dev/null)"
    latest_log="$(find "$latest_run/simulation_log" -type f -name '*.msgpack.xz' -print -quit 2>/dev/null)"
    if [[ -n "$latest_nuboard" && -n "$latest_log" ]]; then
        cp -- "$latest_nuboard" "$SCRIPT_DIR/saved_nuboards/three_vehicle_combined_avoidance.nuboard"
        printf 'Saved latest combined NuBoard: %s\n' "$SCRIPT_DIR/saved_nuboards/three_vehicle_combined_avoidance.nuboard"
    else
        printf 'Simulation did not produce a complete log; saved NuBoard was left unchanged.\n' >&2
    fi
fi
exit "$status"

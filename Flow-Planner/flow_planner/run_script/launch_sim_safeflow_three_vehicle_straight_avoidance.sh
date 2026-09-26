#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SYNTHETIC_AVOIDANCE_MODE=straight_lane_change
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-80}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-24.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-26.0}"

exec "$SCRIPT_DIR/launch_sim_safeflow_synthetic_three_vehicle.sh" \
    planner.safeflow_planner.avoidance_transition_distance_m=20.0 \
    "$@"

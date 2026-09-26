#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MINI_DB_FILE="${MINI_DB_FILE:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE=intersection_turn
# The 110-frame version ended just before PDM completed the second overtake.
# Use the full local scenario horizon (149 frames) instead.
export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-150}"
export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-30.0}"
export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-40.0}"
export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-4.0}"
export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-4.0}"
export SINGLE_AGENT_SCENARIO=false

exec "$SCRIPT_DIR/run_pdm_ablation.sh" curve_two_vehicle \
    planner.pdm_closed_planner.idm_policies.speed_limit_fraction='[0.3,0.5,0.7,0.9,1.0]' \
    planner.pdm_closed_planner.idm_policies.fallback_target_velocity=13.5 \
    planner.pdm_closed_planner.idm_policies.accel_max=2.0 \
    planner.pdm_closed_planner.idm_policies.min_gap_to_lead_agent=1.0 \
    planner.pdm_closed_planner.lateral_offsets='[-3.5,3.5]' \
    "$@"

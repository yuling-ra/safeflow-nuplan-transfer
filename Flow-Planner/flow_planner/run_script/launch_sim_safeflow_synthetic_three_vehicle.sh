#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 兼容入口：默认运行组合三车避障（直行车 + 路口左转车）。

export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.16.18.06.21_veh-38_04471_04922.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-medium_magnitude_speed}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-f2e82a3aaccb5777}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"

# 启用合成三车场景模式
export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
export SYNTHETIC_AVOIDANCE_MODE="${SYNTHETIC_AVOIDANCE_MODE:-combined}"
export SINGLE_AGENT_SCENARIO=false
export MODEL_TRAJECTORY_STEPS="${MODEL_TRAJECTORY_STEPS:-160}"
unset THREE_VEHICLE_SCENARIO
unset SINGLE_AGENT_SCENARIO_TOKENS

"$SCRIPT_DIR/launch_sim_safeflow.sh" \
    planner.safeflow_planner.enable_overtake_demo=false \
    planner.safeflow_planner.force_overtake_demo=false \
    planner.safeflow_planner.use_cbf=true \
    planner.safeflow_planner.use_obstacle_free_reference=true \
    planner.safeflow_planner.ignore_traffic_lights_in_reference=true \
    planner.safeflow_planner.use_scenario_route_reference=true \
    planner.safeflow_planner.enable_route_avoidance_fallback=true \
    planner.safeflow_planner.avoidance_lateral_offset_m=4.5 \
    planner.safeflow_planner.avoidance_transition_distance_m=16.0 \
    planner.safeflow_planner.local_goal_distance_m=50.0 \
    planner.safeflow_planner.max_agents=2 \
    planner.safeflow_planner.obstacle_query_radius_m=50.0 \
    planner.safeflow_planner.obstacle_margin_m=0.25 \
    planner.safeflow_planner.obstacle_prediction_horizon_s=0.0 \
    planner.safeflow_planner.obstacle_prediction_steps=1 \
    planner.safeflow_planner.overtake_allow_lane_connector=true \
    "$@"

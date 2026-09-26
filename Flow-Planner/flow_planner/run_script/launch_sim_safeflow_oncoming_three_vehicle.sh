#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 严格三车实验：自车 + 真实对向直行车 + 真实左转车。
# 两个 NPC 都来自 nuPlan 记录轨迹，因此沿当前地图车道合法行驶。
export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-/new_world/cockatiel/TeleNas/DataExchange/yuling/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_straight_traffic_light_intersection_traversal}"
export SCENARIO_TOKENS="${SCENARIO_TOKENS:-793afaec74a75a5a}"
export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
export THREE_VEHICLE_SCENARIO=true
export THREE_VEHICLE_SCENARIO_TOKENS="${THREE_VEHICLE_SCENARIO_TOKENS:-793afaec74a75a5a}"
export THREE_VEHICLE_ONCOMING_TOKEN="${THREE_VEHICLE_ONCOMING_TOKEN:-31cdd11c1ab85e26}"
export THREE_VEHICLE_LEFT_TURN_TOKEN="${THREE_VEHICLE_LEFT_TURN_TOKEN:-d50a53a49aea5487}"
export SINGLE_AGENT_SCENARIO=false
unset TWO_VEHICLE_INTERACTION
unset SINGLE_AGENT_SCENARIO_TOKENS
unset SINGLE_AGENT_VEHICLE_SPEED_SCALE
unset SINGLE_AGENT_VEHICLE_POSITION_SCALE
unset SINGLE_AGENT_VEHICLE_STRAIGHT

"$SCRIPT_DIR/launch_sim_safeflow.sh" \
    planner.safeflow_planner.enable_overtake_demo=false \
    planner.safeflow_planner.force_overtake_demo=false \
    planner.safeflow_planner.max_agents=2 \
    planner.safeflow_planner.local_goal_distance_m=12.0 \
    "$@"

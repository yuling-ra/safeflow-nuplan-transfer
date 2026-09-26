#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
YULING_ROOT="$(cd -- "$RUN_SCRIPT_DIR/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/new_world/cockatiel/miniconda3/envs/nuplan_clean/bin/python}"
export SAFEFLOW_PROJECT_ROOT="${SAFEFLOW_PROJECT_ROOT:-$YULING_ROOT/safeflow-nuplan-transfer/dmpc_fm_cbf}"
export CKPT_FILE="${CKPT_FILE:-$SAFEFLOW_PROJECT_ROOT/rectified/checkpoints/model_reflow_round1.pt}"
export FM_NUM_SEGMENTS="${FM_NUM_SEGMENTS:-1}"
export MODEL_TRAJECTORY_STEPS="${MODEL_TRAJECTORY_STEPS:-60}"
export PLANNER_DEVICE="${PLANNER_DEVICE:-auto}"

scenario="${1:-}"
shift || true

case "$scenario" in
  straight)
    output_name="reflow_straight_overtake.nuboard"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_reflow_ablation/straight}"
    command=("$RUN_SCRIPT_DIR/launch_sim_safeflow_straight_overtake.sh")
    ;;
  curve)
    output_name="reflow_curve_two_vehicle.nuboard"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_reflow_ablation/curve}"
    export MINI_DB_FILE="${MINI_DB_FILE:-${NUPLAN_DATA_ROOT:-$YULING_ROOT/nuplan-devkit/nuplan-v1.1_mini}/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
    export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_left_turn}"
    export SCENARIO_TOKENS="${SCENARIO_TOKENS:-d9ee9cf40a84520f}"
    export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
    export SYNTHETIC_AVOIDANCE_MODE=intersection_turn
    export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-110}"
    export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-18.0}"
    export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-28.0}"
    command=("$RUN_SCRIPT_DIR/launch_sim_safeflow_synthetic_three_vehicle.sh"
      planner.safeflow_planner.local_goal_distance_m=50.0
      planner.safeflow_planner.avoidance_lateral_offset_m=3.4
      planner.safeflow_planner.avoidance_transition_distance_m=20.0)
    ;;
  multi)
    output_name="reflow_multi_vehicle.nuboard"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_reflow_ablation/multi_vehicle}"
    export SYNTHETIC_AVOIDANCE_MODE=combined
    export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-150}"
    export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-30.0}"
    export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-40.0}"
    export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-0.5}"
    export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-2.5}"
    command=("$RUN_SCRIPT_DIR/launch_sim_safeflow_synthetic_three_vehicle.sh"
      planner.safeflow_planner.avoidance_transition_distance_m=8.0
      planner.safeflow_planner.follow_scenario_route_after_passing_agents=true
      planner.safeflow_planner.route_follow_min_speed_mps=10.0
      planner.safeflow_planner.route_rejoin_distance_m=20.0
      planner.safeflow_planner.sequential_agent_overtake=false
      planner.safeflow_planner.obstacle_query_radius_m=32.0
      planner.safeflow_planner.obstacle_margin_m=0.1
      planner.safeflow_planner.obstacle_prediction_horizon_s=1.0
      planner.safeflow_planner.obstacle_prediction_steps=2)
    ;;
  pedestrian)
    output_name="reflow_right_turn_pedestrian.nuboard"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_reflow_ablation/pedestrian}"
    export MINI_DB_FILE="${MINI_DB_FILE:-$YULING_ROOT/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.10.11.02.57.41_veh-50_01522_02088.db}"
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
    command=("$RUN_SCRIPT_DIR/launch_sim_safeflow.sh"
      planner.safeflow_planner.enable_overtake_demo=false
      planner.safeflow_planner.force_overtake_demo=false
      planner.safeflow_planner.use_cbf=true
      planner.safeflow_planner.use_obstacle_free_reference=true
      planner.safeflow_planner.ignore_traffic_lights_in_reference=false
      planner.safeflow_planner.use_scenario_route_reference=true
      planner.safeflow_planner.enable_route_avoidance_fallback=true
      planner.safeflow_planner.max_agents=8
      planner.safeflow_planner.obstacle_query_radius_m=45.0
      planner.safeflow_planner.obstacle_margin_m=0.75
      planner.safeflow_planner.obstacle_prediction_horizon_s=1.5
      planner.safeflow_planner.obstacle_prediction_steps=4
      planner.safeflow_planner.avoidance_transition_distance_m=12.0
      planner.safeflow_planner.local_goal_distance_m=35.0)
    ;;
  turn)
    output_name="reflow_roundabout_turn.nuboard"
    export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/tmp/safeflow_reflow_ablation/roundabout_turn}"
    export MINI_DB_FILE="${MINI_DB_FILE:-$YULING_ROOT/nuplan-devkit/nuplan-v1.1_mini/data/cache/mini/2021.07.24.20.37.45_veh-17_00015_00375.db}"
    export SCENARIO_TYPES="${SCENARIO_TYPES:-starting_right_turn}"
    export SCENARIO_TOKENS="${SCENARIO_TOKENS:-e3cfe893945d5636}"
    export MAP_NAMES="${MAP_NAMES:-us-nv-las-vegas-strip}"
    export SYNTHETIC_THREE_VEHICLE_SCENARIO=true
    export SYNTHETIC_AVOIDANCE_MODE=roundabout_turn
    export SYNTHETIC_SCENARIO_MAX_ITERATIONS="${SYNTHETIC_SCENARIO_MAX_ITERATIONS:-300}"
    export SYNTHETIC_OBSTACLE_1_PROGRESS_M="${SYNTHETIC_OBSTACLE_1_PROGRESS_M:-45.0}"
    export SYNTHETIC_OBSTACLE_2_PROGRESS_M="${SYNTHETIC_OBSTACLE_2_PROGRESS_M:-65.0}"
    export SYNTHETIC_VEHICLE_1_SPEED_MPS="${SYNTHETIC_VEHICLE_1_SPEED_MPS:-0.35}"
    export SYNTHETIC_VEHICLE_2_SPEED_MPS="${SYNTHETIC_VEHICLE_2_SPEED_MPS:-0.25}"
    export SINGLE_AGENT_SCENARIO=false
    command=("$RUN_SCRIPT_DIR/launch_sim_safeflow.sh"
      planner.safeflow_planner.enable_overtake_demo=false
      planner.safeflow_planner.force_overtake_demo=false
      planner.safeflow_planner.use_cbf=true
      planner.safeflow_planner.use_obstacle_free_reference=true
      planner.safeflow_planner.ignore_traffic_lights_in_reference=true
      planner.safeflow_planner.use_scenario_route_reference=true
      planner.safeflow_planner.route_geometry_guidance_weight=1.0
      planner.safeflow_planner.enable_route_avoidance_fallback=true
      planner.safeflow_planner.fm_num_segments="${TURN_FM_NUM_SEGMENTS:-2}"
      planner.safeflow_planner.local_goal_distance_m=50.0
      planner.safeflow_planner.max_agents=8
      planner.safeflow_planner.obstacle_query_radius_m=20.0
      planner.safeflow_planner.obstacle_margin_m=0.5
      planner.safeflow_planner.obstacle_prediction_horizon_s=1.5
      planner.safeflow_planner.obstacle_prediction_steps=4
      planner.safeflow_planner.avoidance_transition_distance_m=10.0)
    ;;
  *)
    printf 'Usage: %s {straight|curve|multi|pedestrian|turn} [hydra overrides...]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ ! -f "$CKPT_FILE" ]]; then
  printf 'ReFlow checkpoint not found: %s\n' "$CKPT_FILE" >&2
  exit 2
fi

printf 'Running ReFlow SafeFlow ablation: %s\n' "$scenario"
printf 'Checkpoint: %s\n' "$CKPT_FILE"
printf 'FM_NUM_SEGMENTS: %s\n' "$FM_NUM_SEGMENTS"
printf 'NUPLAN_EXP_ROOT: %s\n' "$NUPLAN_EXP_ROOT"

set +e
"${command[@]}" "$@"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  result_root="$NUPLAN_EXP_ROOT/exp/simulation/closed_loop_nonreactive_agents/safeflow/all_scenarios/fm_cbf"
  latest_run="$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -name 'model_reflow_round1_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
  latest_nuboard="$(find "$latest_run" -maxdepth 1 -type f -name '*.nuboard' -print -quit 2>/dev/null)"
  latest_log="$(find "$latest_run/simulation_log" -type f -name '*.msgpack.xz' -print -quit 2>/dev/null)"
  if [[ -n "$latest_nuboard" && -n "$latest_log" ]]; then
    mkdir -p "$SCRIPT_DIR/saved_nuboards"
    cp -- "$latest_nuboard" "$SCRIPT_DIR/saved_nuboards/$output_name"
    printf 'Saved ReFlow NuBoard: %s\n' "$SCRIPT_DIR/saved_nuboards/$output_name"
  else
    printf 'Simulation completed without a complete NuBoard/log pair; no saved result was copied.\n' >&2
    status=1
  fi
fi

exit "$status"

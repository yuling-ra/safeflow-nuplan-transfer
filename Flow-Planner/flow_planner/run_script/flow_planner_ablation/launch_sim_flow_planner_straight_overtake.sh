#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FLOW_PLANNER_OVERTAKE_WRAPPER=true
export FLOW_PLANNER_EXPERIMENT_NAME=straight_overtake

exec "$SCRIPT_DIR/launch_sim_flow_planner_straight.sh" "$@"

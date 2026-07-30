#!/usr/bin/env python3
"""Run nuPlan without serializing the learned planner into every scene log."""

import runpy
from typing import Any

from nuplan.planning.simulation.simulation_log import SimulationLog


_save_to_file = SimulationLog.save_to_file


def _save_without_runtime_objects(self: SimulationLog) -> Any:
    planner = self.planner
    map_api = self.simulation_history.map_api
    try:
        # nuBoard reads the scenario and history. The model and map API make each
        # serialized scene unnecessarily large and can require CUDA on reload.
        self.planner = None
        self.simulation_history.map_api = None
        return _save_to_file(self)
    finally:
        self.planner = planner
        self.simulation_history.map_api = map_api


def main() -> None:
    SimulationLog.save_to_file = _save_without_runtime_objects
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")


if __name__ == "__main__":
    main()

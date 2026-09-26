#!/usr/bin/env python3
"""Run nuPlan simulation with small local compatibility patches."""

import runpy
import os
from typing import Any, List, Set

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_log import SimulationLog

from single_agent_scenario import SingleAgentScenario
from three_vehicle_scenario import ThreeVehicleScenario
from three_vehicle_synthetic_scenario import ThreeVehicleSyntheticScenario


_save_to_file = SimulationLog.save_to_file


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _csv_set(name: str) -> Set[str]:
    return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}


def _patch_scenarios() -> None:
    three_vehicle_enabled = _env_enabled("THREE_VEHICLE_SCENARIO") or _env_enabled("TWO_VEHICLE_INTERACTION")
    synthetic_three_vehicle_enabled = _env_enabled("SYNTHETIC_THREE_VEHICLE_SCENARIO")
    single_agent_enabled = _env_enabled("SINGLE_AGENT_SCENARIO")
    if not three_vehicle_enabled and not synthetic_three_vehicle_enabled and not single_agent_enabled:
        return

    from nuplan.common.utils.distributed_scenario_filter import DistributedScenarioFilter

    original_get_scenarios = DistributedScenarioFilter.get_scenarios
    single_agent_tokens = _csv_set("SINGLE_AGENT_SCENARIO_TOKENS")
    three_vehicle_tokens = _csv_set("THREE_VEHICLE_SCENARIO_TOKENS")

    def get_scenarios(self: Any) -> List[AbstractScenario]:
        scenarios = original_get_scenarios(self)
        patched: List[AbstractScenario] = []
        for scenario in scenarios:
            if synthetic_three_vehicle_enabled:
                patched.append(ThreeVehicleSyntheticScenario(scenario))
            elif three_vehicle_enabled and (not three_vehicle_tokens or scenario.token in three_vehicle_tokens):
                patched.append(ThreeVehicleScenario(scenario))
            elif single_agent_enabled and (not single_agent_tokens or scenario.token in single_agent_tokens):
                patched.append(SingleAgentScenario(scenario))
            else:
                patched.append(scenario)
        return patched

    DistributedScenarioFilter.get_scenarios = get_scenarios


def _save_without_planner(self: SimulationLog) -> Any:
    planner = self.planner
    map_api = self.simulation_history.map_api
    try:
        # nuBoard only reads scenario and simulation_history.  A learned
        # planner can contain a full CUDA model, making every log huge and
        # impossible to open in a CPU-only nuBoard process.
        self.planner = None
        self.simulation_history.map_api = None
        return _save_to_file(self)
    finally:
        self.planner = planner
        self.simulation_history.map_api = map_api


def main() -> None:
    SimulationLog.save_to_file = _save_without_planner
    _patch_scenarios()
    runpy.run_module("nuplan.planning.script.run_simulation", run_name="__main__")


if __name__ == "__main__":
    main()

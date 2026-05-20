from __future__ import annotations

import importlib
import importlib.util
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class OfficialPlannerSpec:
    planner_key: str
    hydra_planner_name: str
    source_project: str
    requires_tuplan_garage: bool = False
    description: str = ""


@dataclass
class NuPlanSimulationConfig:
    devkit_root: Path
    data_root: Path
    map_root: Path
    output_dir: Path
    planner_key: str
    simulation: str = "closed_loop_nonreactive_agents"
    scenario_builder: str = "nuplan"
    scenario_filter: str = "mini"
    split_filter_yaml: Optional[Path] = None
    sensor_root: Optional[Path] = None
    hydra_searchpath: List[str] = field(default_factory=list)
    extra_overrides: List[str] = field(default_factory=list)
    python_bin: str = "python"

    def __post_init__(self) -> None:
        self.devkit_root = Path(self.devkit_root).resolve()
        self.data_root = Path(self.data_root).resolve()
        self.map_root = Path(self.map_root).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.sensor_root is not None:
            self.sensor_root = Path(self.sensor_root).resolve()


PLANNER_REGISTRY: Dict[str, OfficialPlannerSpec] = {
    "official_idm": OfficialPlannerSpec(
        planner_key="official_idm",
        hydra_planner_name="idm_planner",
        source_project="nuplan-devkit",
        description="Official IDM baseline shipped with nuplan-devkit.",
    ),
    "official_pdm_closed": OfficialPlannerSpec(
        planner_key="official_pdm_closed",
        hydra_planner_name="pdm_closed_planner",
        source_project="tuplan_garage",
        requires_tuplan_garage=True,
        description="PDM-Closed baseline from autonomousvision/tuplan_garage.",
    ),
}


def get_planner_spec(planner_key: str) -> OfficialPlannerSpec:
    if planner_key not in PLANNER_REGISTRY:
        raise KeyError(f"Unknown planner key: {planner_key}")
    return PLANNER_REGISTRY[planner_key]


def is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def is_nuplan_sdk_available() -> bool:
    return is_module_available("nuplan")


def is_tuplan_garage_available() -> bool:
    return is_module_available("tuplan_garage")


def validate_sdk_for_planner(planner_key: str) -> Dict[str, bool]:
    spec = get_planner_spec(planner_key)
    status = {
        "nuplan": is_nuplan_sdk_available(),
        "tuplan_garage": is_tuplan_garage_available(),
    }
    if not status["nuplan"]:
        raise ImportError(
            "nuplan-devkit is not installed. Install `nuplan-devkit` first. "
            "Official sources: Motional/HorizonRobotics nuplan-devkit."
        )
    if spec.requires_tuplan_garage and not status["tuplan_garage"]:
        raise ImportError(
            "tuplan_garage is not installed. `official_pdm_closed` depends on autonomousvision/tuplan_garage."
        )
    return status


def build_idm_planner(
    *,
    target_velocity: float = 10.0,
    min_gap_to_lead_agent: float = 1.0,
    headway_time: float = 1.5,
    accel_max: float = 1.0,
    decel_max: float = 2.0,
    planned_trajectory_samples: int = 40,
    planned_trajectory_sample_interval: float = 0.1,
    occupancy_map_radius: float = 40.0,
):
    validate_sdk_for_planner("official_idm")
    from nuplan.planning.simulation.planner.idm_planner import IDMPlanner

    return IDMPlanner(
        target_velocity=target_velocity,
        min_gap_to_lead_agent=min_gap_to_lead_agent,
        headway_time=headway_time,
        accel_max=accel_max,
        decel_max=decel_max,
        planned_trajectory_samples=planned_trajectory_samples,
        planned_trajectory_sample_interval=planned_trajectory_sample_interval,
        occupancy_map_radius=occupancy_map_radius,
    )


def build_pdm_closed_planner(**kwargs):
    validate_sdk_for_planner("official_pdm_closed")

    candidate_imports = [
        "tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner",
        "tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner",
    ]
    last_error = None
    for module_name in candidate_imports:
        try:
            mod = importlib.import_module(module_name)
            for cls_name in ["PDMClosedPlanner", "PDMClosed", "AbstractPDMClosedPlanner"]:
                if hasattr(mod, cls_name):
                    return getattr(mod, cls_name)(**kwargs)
        except Exception as exc:
            last_error = exc

    raise ImportError(
        "Could not import a PDM-Closed planner class from tuplan_garage. "
        "Check the installed tuplan_garage version and planner module path."
    ) from last_error


def build_nuplan_scenario_builder(
    *,
    data_root: Path,
    map_root: Path,
    sensor_root: Optional[Path] = None,
    db_files: Optional[Sequence[str]] = None,
    map_version: str = "nuplan-maps-v1.0",
    include_cameras: bool = False,
):
    validate_sdk_for_planner("official_idm")
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

    return NuPlanScenarioBuilder(
        data_root=str(data_root),
        map_root=str(map_root),
        sensor_root="" if sensor_root is None else str(sensor_root),
        db_files=None if db_files is None else list(db_files),
        map_version=map_version,
        include_cameras=include_cameras,
    )


def build_scenario_filter(
    *,
    scenario_types: Optional[List[str]] = None,
    num_scenarios_per_type: Optional[int] = None,
    limit_total_scenarios: Optional[int] = None,
    shuffle: bool = False,
    remove_invalid_goals: bool = False,
    ego_route_radius: Optional[float] = 30.0,
):
    validate_sdk_for_planner("official_idm")
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    return ScenarioFilter(
        scenario_types=scenario_types,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=num_scenarios_per_type,
        limit_total_scenarios=limit_total_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=remove_invalid_goals,
        shuffle=shuffle,
        ego_route_radius=ego_route_radius,
    )


def default_hydra_searchpath(planner_key: str) -> List[str]:
    spec = get_planner_spec(planner_key)
    base = [
        "pkg://nuplan.planning.script.config.common",
        "pkg://nuplan.planning.script.experiments",
    ]
    if spec.requires_tuplan_garage:
        return [
            "pkg://tuplan_garage.planning.script.config.common",
            "pkg://tuplan_garage.planning.script.config.simulation",
            *base,
        ]
    return base


def build_simulation_overrides(cfg: NuPlanSimulationConfig) -> List[str]:
    spec = get_planner_spec(cfg.planner_key)
    overrides = [
        f"+simulation={cfg.simulation}",
        f"planner={spec.hydra_planner_name}",
        f"scenario_filter={cfg.scenario_filter}",
        f"scenario_builder={cfg.scenario_builder}",
        f"group={spec.hydra_planner_name}",
        f"output_dir={cfg.output_dir}",
    ]

    searchpath = cfg.hydra_searchpath or default_hydra_searchpath(cfg.planner_key)
    if searchpath:
        quoted = ", ".join(searchpath)
        overrides.append(f'hydra.searchpath="[{quoted}]"')

    if cfg.split_filter_yaml is not None:
        overrides.append(f"+scenario_filter_path={cfg.split_filter_yaml}")

    overrides.extend(cfg.extra_overrides)
    return overrides


def build_run_simulation_command(cfg: NuPlanSimulationConfig) -> List[str]:
    validate_sdk_for_planner(cfg.planner_key)
    run_script = cfg.devkit_root / "nuplan" / "planning" / "script" / "run_simulation.py"
    if not run_script.exists():
        raise FileNotFoundError(f"run_simulation.py not found under {cfg.devkit_root}")

    cmd = [cfg.python_bin, str(run_script), *build_simulation_overrides(cfg)]
    return cmd


def shell_command_string(cfg: NuPlanSimulationConfig) -> str:
    return " ".join(shlex.quote(part) for part in build_run_simulation_command(cfg))


def recommended_env(cfg: NuPlanSimulationConfig) -> Dict[str, str]:
    env = {
        "NUPLAN_DATA_ROOT": str(cfg.data_root),
        "NUPLAN_MAPS_ROOT": str(cfg.map_root),
        "NUPLAN_DEVKIT_ROOT": str(cfg.devkit_root),
    }
    if cfg.sensor_root is not None:
        env["NUPLAN_SENSOR_ROOT"] = str(cfg.sensor_root)
    return env


def format_env_exports(cfg: NuPlanSimulationConfig) -> str:
    env = recommended_env(cfg)
    return "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())

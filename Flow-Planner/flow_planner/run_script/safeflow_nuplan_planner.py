from __future__ import annotations

import logging
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class SafeFlowNuPlanPlanner(AbstractPlanner):
    """Adapter that exposes the canonical 5-channel SafeFlow model to nuPlan."""

    requires_scenario = True

    def __init__(
        self,
        ckpt_path: str,
        safeflow_project_root: str,
        device: str = "cpu",
        model_trajectory_steps: int = 60,
        trajectory_sample_interval: float = 0.1,
        local_goal_distance_m: float = 3.0,
        minimum_goal_distance_m: float = 0.5,
        max_agents: int = 8,
        obstacle_query_radius_m: float = 15.0,
        obstacle_margin_m: float = 0.75,
        fm_num_segments: int = 8,
        ode_method: str = "rk4",
        fm_noise_scale: float = 1.0,
        start_guidance_weight: float = 0.4,
        goal_guidance_weight: float = 1.2,
        use_cbf: bool = True,
        cbf_tau0: float = 0.5,
        cbf_tau1: float = 0.9,
        cbf_passes: int = 8,
        cbf_extra_margin_m: float = 0.25,
        cbf_alpha: float = 8.0,
        cbf_max_corr_norm: float = 2.0,
        obstacle_prediction_horizon_s: float = 3.0,
        obstacle_prediction_steps: int = 5,
        model_speed_scale: float = 0.06,
        max_output_speed_mps: float = 15.0,
        emergency_decel_mps2: float = 6.0,
        seed: int = 7,
        idm_target_velocity: float = 8.0,
        idm_min_gap_to_lead_agent: float = 1.0,
        idm_headway_time: float = 1.5,
        idm_accel_max: float = 1.0,
        idm_decel_max: float = 3.0,
        idm_occupancy_map_radius: float = 40.0,
        use_obstacle_free_reference: bool = False,
        ignore_traffic_lights_in_reference: bool = False,
        use_scenario_route_reference: bool = False,
        route_geometry_guidance_weight: float = 0.0,
        enable_route_avoidance_fallback: bool = False,
        avoidance_lateral_offset_m: float = 4.5,
        avoidance_transition_distance_m: float = 7.0,
        follow_scenario_route_after_passing_agents: bool = False,
        route_follow_min_speed_mps: float = 0.0,
        route_rejoin_distance_m: float = 20.0,
        sequential_agent_overtake: bool = False,
        enable_overtake_demo: bool = True,
        force_overtake_demo: bool = True,
        overtake_pass_side: str = "left",
        overtake_lateral_offset_m: float = 3.5,
        overtake_min_lead_distance_m: float = 4.0,
        overtake_max_lead_distance_m: float = 30.0,
        overtake_same_lane_width_m: float = 2.2,
        overtake_distance_m: float = 28.0,
        overtake_target_speed_mps: float = 5.0,
        overtake_min_clearance_m: float = 4.0,
        overtake_require_lane: bool = True,
        overtake_lane_sample_stride: int = 4,
        overtake_follow_distance_m: float = 18.0,
        overtake_lane_change_distance_m: float = 26.0,
        overtake_return_to_original_lane: bool = False,
        overtake_allow_lane_connector: bool = False,
        debug_output_dir: str = "",
        debug_max_steps: int = 8,
        scenario: AbstractScenario | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported device: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SafeFlow planner requested CUDA, but CUDA is unavailable")
        if model_trajectory_steps < 4:
            raise ValueError("model_trajectory_steps must be at least 4")
        if trajectory_sample_interval <= 0:
            raise ValueError("trajectory_sample_interval must be positive")

        self._ckpt_path = Path(ckpt_path).expanduser().resolve()
        self._safeflow_project_root = Path(safeflow_project_root).expanduser().resolve()
        self._device = device
        self._model_trajectory_steps = int(model_trajectory_steps)
        self._trajectory_sample_interval = float(trajectory_sample_interval)
        self._local_goal_distance_m = float(local_goal_distance_m)
        self._minimum_goal_distance_m = float(minimum_goal_distance_m)
        self._max_agents = int(max_agents)
        self._obstacle_query_radius_m = float(obstacle_query_radius_m)
        self._obstacle_margin_m = float(obstacle_margin_m)
        self._fm_num_segments = int(fm_num_segments)
        self._ode_method = str(ode_method)
        self._fm_noise_scale = float(fm_noise_scale)
        self._start_guidance_weight = float(start_guidance_weight)
        self._goal_guidance_weight = float(goal_guidance_weight)
        self._use_cbf = bool(use_cbf)
        self._cbf_tau0 = float(cbf_tau0)
        self._cbf_tau1 = float(cbf_tau1)
        self._cbf_kwargs = {
            "passes": int(cbf_passes),
            "extra_margin": float(cbf_extra_margin_m),
            "alpha": float(cbf_alpha),
            "max_corr_norm": float(cbf_max_corr_norm),
        }
        self._obstacle_prediction_horizon_s = max(0.0, float(obstacle_prediction_horizon_s))
        self._obstacle_prediction_steps = max(1, int(obstacle_prediction_steps))
        self._model_speed_scale = max(0.0, float(model_speed_scale))
        self._max_output_speed_mps = float(max_output_speed_mps)
        self._emergency_decel_mps2 = max(0.1, float(emergency_decel_mps2))
        self._seed = int(seed)
        self._use_obstacle_free_reference = bool(use_obstacle_free_reference)
        self._ignore_traffic_lights_in_reference = bool(ignore_traffic_lights_in_reference)
        self._use_scenario_route_reference = bool(use_scenario_route_reference)
        self._route_geometry_guidance_weight = float(np.clip(route_geometry_guidance_weight, 0.0, 1.0))
        self._scenario = scenario
        if self._use_scenario_route_reference and self._scenario is None:
            raise ValueError("use_scenario_route_reference requires a scenario")
        self._enable_route_avoidance_fallback = bool(enable_route_avoidance_fallback)
        self._avoidance_lateral_offset_m = max(0.1, float(avoidance_lateral_offset_m))
        self._avoidance_transition_distance_m = max(2.0, float(avoidance_transition_distance_m))
        self._follow_scenario_route_after_passing_agents = bool(follow_scenario_route_after_passing_agents)
        self._route_follow_min_speed_mps = max(0.0, float(route_follow_min_speed_mps))
        self._route_rejoin_distance_m = max(2.0, float(route_rejoin_distance_m))
        self._sequential_agent_overtake = bool(sequential_agent_overtake)
        if overtake_pass_side not in {"left", "right"}:
            raise ValueError(f"Unsupported overtake_pass_side: {overtake_pass_side}")
        self._enable_overtake_demo = bool(enable_overtake_demo)
        self._force_overtake_demo = bool(force_overtake_demo)
        self._overtake_side_sign = 1.0 if overtake_pass_side == "left" else -1.0
        self._overtake_lateral_offset_m = float(overtake_lateral_offset_m)
        self._overtake_min_lead_distance_m = float(overtake_min_lead_distance_m)
        self._overtake_max_lead_distance_m = float(overtake_max_lead_distance_m)
        self._overtake_same_lane_width_m = float(overtake_same_lane_width_m)
        self._overtake_distance_m = float(overtake_distance_m)
        self._overtake_target_speed_mps = float(overtake_target_speed_mps)
        self._overtake_min_clearance_m = float(overtake_min_clearance_m)
        self._overtake_require_lane = bool(overtake_require_lane)
        self._overtake_lane_sample_stride = max(1, int(overtake_lane_sample_stride))
        self._overtake_follow_distance_m = float(overtake_follow_distance_m)
        self._overtake_lane_change_distance_m = float(overtake_lane_change_distance_m)
        self._overtake_return_to_original_lane = bool(overtake_return_to_original_lane)
        self._overtake_allow_lane_connector = bool(overtake_allow_lane_connector)
        self._debug_output_dir = Path(debug_output_dir).expanduser() if debug_output_dir else None
        self._debug_max_steps = int(debug_max_steps)

        self._reference_planner = IDMPlanner(
            target_velocity=float(idm_target_velocity),
            min_gap_to_lead_agent=float(idm_min_gap_to_lead_agent),
            headway_time=float(idm_headway_time),
            accel_max=float(idm_accel_max),
            decel_max=float(idm_decel_max),
            planned_trajectory_samples=self._model_trajectory_steps,
            planned_trajectory_sample_interval=self._trajectory_sample_interval,
            occupancy_map_radius=float(idm_occupancy_map_radius),
        )
        self._model: Any = None
        self._build_can_tf_5d: Any = None
        self._sample_fm: Any = None
        self._successful_inferences = 0
        self._successful_overtakes = 0
        self._avoidance_triggers = 0
        self._successful_avoidances = 0
        self._route_avoidance_fallbacks = 0
        self._route_avoidance_attempts = 0
        self._debug_dump_count = 0
        self._overtake_path_world: np.ndarray | None = None
        self._overtake_start_iteration = -1
        self._avoidance_path_world: np.ndarray | None = None
        # Keep the last trajectory that passed collision validation.  SafeFlow
        # can occasionally sample a loopy path for one frame; reusing the
        # last valid path prevents that transient failure from becoming a
        # permanent emergency-stop loop.
        self._last_safe_path_world: np.ndarray | None = None
        self._route_follow_after_pass = False
        self._first_agent_passed = False
        self._map_api: AbstractMap | None = None
        self._scenario_route_cache: np.ndarray | None = None
        self._scenario_route_progress_cache: np.ndarray | None = None

    def name(self) -> str:
        return "safeflow_fm_cbf" if self._use_cbf else "safeflow_fm"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._reference_planner.initialize(initialization)
        self._map_api = initialization.map_api
        self._cache_scenario_route()
        self._load_model()

    def _cache_scenario_route(self) -> None:
        """Cache route geometry used by sequential passing and route following."""
        if self._scenario is None:
            return
        route_getter = getattr(self._scenario, "get_route_reference_states", None)
        if route_getter is None:
            return
        route_states = route_getter(0)
        if len(route_states) < 2:
            return
        route = np.asarray([state.center.array[:2] for state in route_states], dtype=np.float64)
        self._scenario_route_cache = route
        self._scenario_route_progress_cache = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        )

    def _load_model(self) -> None:
        if not self._ckpt_path.is_file():
            raise FileNotFoundError(f"SafeFlow checkpoint not found: {self._ckpt_path}")
        if not self._safeflow_project_root.is_dir():
            raise FileNotFoundError(f"SafeFlow project root not found: {self._safeflow_project_root}")

        project_root = str(self._safeflow_project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from dmpc_fm_cbf.canonical import build_can_tf_5d
        from dmpc_fm_cbf.sampler_5h import piecewise_sample_fm_5ch_with_cbf
        from fmtorch.models.simple1d_unet import SimpleUNet1D

        checkpoint = torch.load(self._ckpt_path, map_location=self._device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported SafeFlow checkpoint type: {type(checkpoint).__name__}")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_config: Dict[str, int] = dict(
            checkpoint.get(
                "model_config",
                {
                    "time_emb_dim": 128,
                    "hidden_dim": 128,
                    "cond_dim": 5,
                    "in_channels": 5,
                    "out_channels": 5,
                },
            )
        )
        if model_config.get("in_channels") != 5 or model_config.get("out_channels") != 5:
            raise ValueError(
                "SafeFlow nuPlan integration requires the 5-channel checkpoint; "
                f"got {model_config.get('in_channels')}->{model_config.get('out_channels')}"
            )

        model = SimpleUNet1D(**model_config)
        model.load_state_dict(state_dict, strict=True)
        self._model = model.to(self._device).eval()
        self._build_can_tf_5d = build_can_tf_5d
        self._sample_fm = piecewise_sample_fm_5ch_with_cbf
        logger.info("Loaded SafeFlow checkpoint: %s", self._ckpt_path)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        reference_input = self._without_obstacles(current_input) if self._use_obstacle_free_reference else current_input
        reference = self._reference_planner.compute_planner_trajectory(reference_input)
        reference_states = reference.get_sampled_trajectory()
        ego_state, observation = current_input.history.current_state
        if self._use_scenario_route_reference and self._scenario is not None:
            if self._scenario_route_cache is not None:
                # Use spatial progress on the complete route. Indexing the
                # recorded route by simulation time lets the ego outrun the
                # reference in closed loop and creates a lateral shortcut at
                # turns.
                reference_states = self._resample_cached_route_reference(ego_state)
            else:
                route_getter = getattr(self._scenario, "get_route_reference_states", None)
                if route_getter is not None:
                    reference_states = route_getter(current_input.iteration.index)
                else:
                    start_iteration = min(current_input.iteration.index, self._scenario.get_number_of_iterations() - 1)
                    reference_states = [
                        self._scenario.get_ego_state_at_iteration(index)
                        for index in range(start_iteration, self._scenario.get_number_of_iterations())
                    ]
                reference_states = self._resample_route_reference(ego_state, reference_states)
            if len(reference_states) >= 2:
                reference = InterpolatedTrajectory(reference_states)
        agents = self._extract_agents(ego_state, observation)
        if self._sequential_agent_overtake and isinstance(observation, DetectionsTracks):
            tracked_vehicles = observation.tracked_objects.tracked_objects
            first_vehicle = next(
                (vehicle for vehicle in tracked_vehicles if str(vehicle.track_token).endswith("obstacle_1")),
                None,
            )
            if not self._first_agent_passed and first_vehicle is not None:
                ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
                ego_heading = float(ego_state.center.heading)
                forward = np.asarray([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
                relative_longitudinal = float(
                    (np.asarray(first_vehicle.center.array[:2], dtype=np.float64) - ego_xy) @ forward
                )
                pass_clearance = 0.5 * (
                    float(first_vehicle.box.length) + float(ego_state.car_footprint.oriented_box.length)
                )
                self._first_agent_passed = relative_longitudinal < -pass_clearance
            active_suffix = "obstacle_2" if self._first_agent_passed else "obstacle_1"
            agents = [agent for agent in agents if str(agent["track_token"]).endswith(active_suffix)]
        reference_xy = np.asarray(
            [state.center.array[:2] for state in reference_states[: self._model_trajectory_steps]],
            dtype=np.float64,
        )

        if self._follow_scenario_route_after_passing_agents:
            tracked_vehicles = (
                observation.tracked_objects.tracked_objects
                if isinstance(observation, DetectionsTracks)
                else []
            )
            if not self._route_follow_after_pass and len(tracked_vehicles) >= 2:
                ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
                ego_route_progress = self._scenario_route_progress(ego_xy)
                vehicle_route_progress = [
                    self._scenario_route_progress(tracked_vehicle.center.array[:2])
                    for tracked_vehicle in tracked_vehicles
                ]
                if ego_route_progress is not None and all(
                    progress is not None for progress in vehicle_route_progress
                ):
                    self._route_follow_after_pass = all(
                        ego_route_progress
                        > float(progress)
                        + 0.5
                        * (
                            float(tracked_vehicle.box.length)
                            + float(ego_state.car_footprint.oriented_box.length)
                        )
                        for tracked_vehicle, progress in zip(tracked_vehicles, vehicle_route_progress)
                    )
                else:
                    ego_heading = float(ego_state.center.heading)
                    forward = np.asarray([math.cos(ego_heading), math.sin(ego_heading)], dtype=np.float64)
                    self._route_follow_after_pass = all(
                        float(
                            (
                                np.asarray(tracked_vehicle.center.array[:2], dtype=np.float64)
                                - ego_xy
                            )
                            @ forward
                        )
                        < -0.5
                        * (
                            float(tracked_vehicle.box.length)
                            + float(ego_state.car_footprint.oriented_box.length)
                        )
                        for tracked_vehicle in tracked_vehicles
                    )
            if self._route_follow_after_pass:
                self._avoidance_path_world = None
                self._overtake_path_world = None
                spatial_route = self._build_spatial_route_follow_path(ego_state)
                if spatial_route is not None:
                    self._last_safe_path_world = np.asarray(spatial_route, dtype=np.float64).copy()
                    return self._build_trajectory(ego_state, spatial_route)
                return reference
        if self._enable_route_avoidance_fallback and self._avoidance_path_world is not None:
            cached_avoidance = self._slice_cached_avoidance_path(ego_state)
            if cached_avoidance is not None and self._path_is_collision_free(cached_avoidance, agents):
                self._last_safe_path_world = np.asarray(cached_avoidance, dtype=np.float64).copy()
                return self._build_trajectory(ego_state, cached_avoidance)
            self._avoidance_path_world = None

        try:
            local_goal, reference_progress = self._select_local_goal(ego_state, reference_states)
            if local_goal is None:
                return reference

            avoidance_triggered = bool(agents) and not self._path_is_collision_free(reference_xy, agents)
            if avoidance_triggered:
                self._avoidance_triggers += 1
                if self._avoidance_triggers == 1:
                    logger.info("SafeFlow avoidance trigger active: nominal route intersects a predicted obstacle")

            if self._enable_overtake_demo:
                overtake_xy = self._sample_overtake_path(ego_state, agents, current_input.iteration.index)
                if overtake_xy is not None:
                    self._successful_overtakes += 1
                    if self._successful_overtakes == 1:
                        logger.info("Overtake demo active: generated a left/right pass trajectory")
                    self._last_safe_path_world = np.asarray(overtake_xy, dtype=np.float64).copy()
                    return self._build_trajectory(ego_state, overtake_xy)

            world_xy = self._sample_world_path(ego_state, local_goal, agents, current_input.iteration.index)
            if self._route_geometry_guidance_weight >= 1.0:
                world_xy = reference_xy.copy()
            else:
                if self._route_geometry_guidance_weight > 0.0:
                    world_xy = self._blend_route_geometry(world_xy, reference_xy)
                world_xy = self._time_parameterize(world_xy, reference_progress)
            if not self._path_is_collision_free(world_xy, agents):
                if any(agent.get("is_pedestrian", False) for agent in agents):
                    logger.warning("Pedestrian crossing intersects planned path; holding ego before the crosswalk")
                    self._last_safe_path_world = None
                    return self._build_pedestrian_yield_trajectory(ego_state, reference_xy, agents)
                fallback_xy = self._build_route_avoidance_path(ego_state, reference_xy, agents)
                if fallback_xy is None:
                    raise ValueError("SafeFlow path intersects a predicted moving vehicle")
                self._route_avoidance_fallbacks += 1
                if self._route_avoidance_fallbacks == 1:
                    logger.warning(
                        "SafeFlow CBF path remained colliding; route-level avoidance fallback generated a clear path"
                    )
                self._last_safe_path_world = np.asarray(fallback_xy, dtype=np.float64).copy()
                return self._build_trajectory(ego_state, fallback_xy)
            if avoidance_triggered:
                self._successful_avoidances += 1
                if self._successful_avoidances == 1:
                    logger.info("SafeFlow avoidance succeeded: CBF path cleared the nominal-route obstacle")
            trajectory = self._build_trajectory(ego_state, world_xy)
            self._last_safe_path_world = np.asarray(world_xy, dtype=np.float64).copy()
            self._successful_inferences += 1
            if self._successful_inferences == 1:
                logger.info(
                    "SafeFlow inference active: steps=%d, segments=%d, CBF=%s",
                    self._model_trajectory_steps,
                    self._fm_num_segments,
                    self._use_cbf,
                )
            return trajectory
        except Exception:
            logger.exception("SafeFlow inference failed; using a collision-checked safety fallback for this step")
            if any(agent.get("is_pedestrian", False) for agent in agents):
                self._last_safe_path_world = None
                return self._build_pedestrian_yield_trajectory(ego_state, reference_xy, agents)
            cached_safe = self._slice_world_path(self._last_safe_path_world, ego_state)
            if cached_safe is not None and self._path_is_collision_free(cached_safe, agents):
                logger.warning("Reusing the last collision-free trajectory after SafeFlow failure")
                return self._build_trajectory(ego_state, cached_safe)
            if len(reference_xy) >= 2 and self._path_is_collision_free(reference_xy, agents):
                self._last_safe_path_world = reference_xy.copy()
                return reference
            fallback_xy = self._build_route_avoidance_path(ego_state, reference_xy, agents)
            if fallback_xy is not None:
                self._route_avoidance_fallbacks += 1
                if self._route_avoidance_fallbacks == 1:
                    logger.warning(
                        "SafeFlow inference failed; route-level avoidance fallback generated a clear path"
                    )
                self._last_safe_path_world = np.asarray(fallback_xy, dtype=np.float64).copy()
                return self._build_trajectory(ego_state, fallback_xy)
            logger.warning("IDM reference intersects a predicted vehicle; applying emergency braking")
            return self._build_emergency_stop_trajectory(ego_state)

    def _blend_route_geometry(self, world_xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
        """Keep generated offsets while preserving the local map-route shape."""
        generated = np.asarray(world_xy, dtype=np.float64)
        route = np.asarray(reference_xy, dtype=np.float64)
        if len(generated) < 2 or len(route) < 2:
            return generated
        route_index = np.linspace(0.0, float(len(route) - 1), len(generated))
        route_resampled = np.column_stack(
            [
                np.interp(route_index, np.arange(len(route)), route[:, 0]),
                np.interp(route_index, np.arange(len(route)), route[:, 1]),
            ]
        )
        weight = self._route_geometry_guidance_weight
        blended = (1.0 - weight) * generated + weight * route_resampled
        blended[0] = generated[0]
        return blended

    def _without_obstacles(self, current_input: PlannerInput) -> PlannerInput:
        """Clone planner history with empty detections while preserving ego and traffic lights."""
        history = current_input.history
        empty_observations = deque(
            (DetectionsTracks(TrackedObjects([])) for _ in history.observations),
            maxlen=history.size,
        )
        free_history = SimulationHistoryBuffer(
            ego_state_buffer=deque(history.ego_states, maxlen=history.size),
            observations_buffer=empty_observations,
            sample_interval=history.sample_interval,
        )
        return PlannerInput(
            iteration=current_input.iteration,
            history=free_history,
            traffic_light_data=[] if self._ignore_traffic_lights_in_reference else current_input.traffic_light_data,
        )

    def _resample_route_reference(
        self, ego_state: EgoState, reference_states: List[EgoState]
    ) -> List[EgoState]:
        """Resample recorded route geometry at the current closed-loop speed."""
        if len(reference_states) < 2:
            return reference_states
        route = np.asarray([state.center.array[:2] for state in reference_states], dtype=np.float64)
        progress = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))])
        if progress[-1] < 1e-3:
            return reference_states
        speed = max(
            1.0,
            float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()),
            self._route_follow_min_speed_mps if self._route_follow_after_pass else 0.0,
        )
        query = np.minimum(
            np.arange(self._model_trajectory_steps, dtype=np.float64)
            * speed
            * self._trajectory_sample_interval,
            progress[-1],
        )
        sampled_xy = np.column_stack(
            [np.interp(query, progress, route[:, axis]) for axis in range(2)]
        )
        sampled_xy[0] = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        return self._build_trajectory(ego_state, sampled_xy).get_sampled_trajectory()

    def _resample_cached_route_reference(self, ego_state: EgoState) -> List[EgoState]:
        """Sample the full scenario route from the ego's spatial projection."""
        if self._scenario_route_cache is None or self._scenario_route_progress_cache is None:
            return []

        route = self._scenario_route_cache
        progress = self._scenario_route_progress_cache
        segments = np.diff(route, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid_lengths = np.maximum(lengths, 1e-6)
        current = np.asarray(ego_state.center.array[:2], dtype=np.float64)

        # Project the current pose onto the nearest route segment. This keeps
        # the route reference continuous when closed-loop speed differs from
        # the recorded expert timing.
        relative = current[None, :] - route[:-1]
        ratios = np.clip(
            np.sum(relative * segments, axis=1) / (valid_lengths * valid_lengths),
            0.0,
            1.0,
        )
        projections = route[:-1] + ratios[:, None] * segments
        segment_index = int(np.argmin(np.linalg.norm(projections - current[None, :], axis=1)))
        start_progress = float(
            progress[segment_index] + ratios[segment_index] * lengths[segment_index]
        )
        projected_start = projections[segment_index]

        speed = max(
            1.0,
            float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()),
            self._route_follow_min_speed_mps,
        )
        relative_progress = (
            np.arange(self._model_trajectory_steps, dtype=np.float64)
            * speed
            * self._trajectory_sample_interval
        )
        query = start_progress + relative_progress
        clipped_query = np.minimum(query, progress[-1])
        sampled_xy = np.column_stack(
            [np.interp(clipped_query, progress, route[:, axis]) for axis in range(2)]
        )

        # Continue along the final route tangent if the horizon reaches the
        # end of the recorded route.
        beyond_route = query > progress[-1]
        if np.any(beyond_route):
            final_direction = segments[-1] / valid_lengths[-1]
            sampled_xy[beyond_route] += (
                query[beyond_route] - progress[-1]
            )[:, None] * final_direction[None, :]

        correction = current - projected_start
        rejoin_distance = max(self._route_rejoin_distance_m, 1.0)
        correction_weight = np.asarray(
            [1.0 - self._smoothstep(distance / rejoin_distance) for distance in relative_progress],
            dtype=np.float64,
        )
        sampled_xy += correction_weight[:, None] * correction[None, :]
        sampled_xy[0] = current
        return self._build_trajectory(ego_state, sampled_xy).get_sampled_trajectory()

    def _build_spatial_route_follow_path(self, ego_state: EgoState) -> np.ndarray | None:
        """Project onto the full map route and smoothly remove the overtake offset."""
        if self._scenario_route_cache is None or self._scenario_route_progress_cache is None:
            return None

        route = self._scenario_route_cache
        segments = np.diff(route, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid_lengths = np.maximum(lengths, 1e-6)
        progress = self._scenario_route_progress_cache
        current = np.asarray(ego_state.center.array[:2], dtype=np.float64)

        relative = current[None, :] - route[:-1]
        ratios = np.clip(np.sum(relative * segments, axis=1) / (valid_lengths * valid_lengths), 0.0, 1.0)
        projections = route[:-1] + ratios[:, None] * segments
        segment_index = int(np.argmin(np.linalg.norm(projections - current[None, :], axis=1)))
        start_progress = float(progress[segment_index] + ratios[segment_index] * lengths[segment_index])
        projected_start = projections[segment_index]

        speed = max(
            self._route_follow_min_speed_mps,
            float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()),
            1.0,
        )
        relative_progress = (
            np.arange(self._model_trajectory_steps, dtype=np.float64)
            * speed
            * self._trajectory_sample_interval
        )
        query = start_progress + relative_progress
        clipped_query = np.minimum(query, progress[-1])
        path = np.column_stack([np.interp(clipped_query, progress, route[:, axis]) for axis in range(2)])

        beyond_route = query > progress[-1]
        if np.any(beyond_route):
            final_direction = segments[-1] / valid_lengths[-1]
            path[beyond_route] += (query[beyond_route] - progress[-1])[:, None] * final_direction[None, :]

        correction = current - projected_start
        blend = np.asarray(
            [1.0 - self._smoothstep(distance / self._route_rejoin_distance_m) for distance in relative_progress],
            dtype=np.float64,
        )
        path += blend[:, None] * correction[None, :]
        path[0] = current
        return path

    def _scenario_route_progress(self, point: np.ndarray) -> float | None:
        """Project a point onto the full scenario route and return arc-length progress."""
        if self._scenario_route_cache is None or self._scenario_route_progress_cache is None:
            return None
        route = self._scenario_route_cache
        segments = np.diff(route, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        valid_lengths = np.maximum(lengths, 1e-6)
        progress = self._scenario_route_progress_cache
        relative = np.asarray(point, dtype=np.float64)[None, :] - route[:-1]
        ratios = np.clip(np.sum(relative * segments, axis=1) / (valid_lengths * valid_lengths), 0.0, 1.0)
        projections = route[:-1] + ratios[:, None] * segments
        segment_index = int(
            np.argmin(np.linalg.norm(projections - np.asarray(point, dtype=np.float64)[None, :], axis=1))
        )
        return float(progress[segment_index] + ratios[segment_index] * lengths[segment_index])

    def _build_route_avoidance_path(
        self, ego_state: EgoState, reference_xy: np.ndarray, agents: List[Dict[str, Any]]
    ) -> np.ndarray | None:
        """Build a smooth lane-level bypass around agents intersecting the nominal route."""
        if not self._enable_route_avoidance_fallback or len(reference_xy) < 4 or not agents:
            return None

        route = np.asarray(reference_xy, dtype=np.float64)
        segment_lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        progress = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if progress[-1] < self._avoidance_transition_distance_m:
            return None

        obstacle_progress = []
        obstacle_radii = []
        for agent in agents:
            distances = np.linalg.norm(route - np.asarray(agent["center"], dtype=np.float64)[None, :], axis=1)
            closest = int(np.argmin(distances))
            if float(distances[closest]) <= float(agent["radius"]) + 1.0:
                obstacle_progress.append(float(progress[closest]))
                obstacle_radii.append(float(agent["collision_half_length"]))
        if not obstacle_progress:
            return None

        first_obstacle = min(obstacle_progress)
        last_obstacle = max(obstacle_progress)
        transition_end = max(1.0, first_obstacle - max(obstacle_radii) - 0.75)
        transition_start = max(0.0, transition_end - self._avoidance_transition_distance_m)
        return_start = last_obstacle + max(obstacle_radii) + 2.0
        return_end = return_start + self._avoidance_transition_distance_m
        can_return = return_end <= float(progress[-1]) - 1.0

        tangents = np.empty_like(route)
        tangents[0] = route[1] - route[0]
        tangents[-1] = route[-1] - route[-2]
        tangents[1:-1] = route[2:] - route[:-2]
        norms = np.linalg.norm(tangents, axis=1)
        valid = norms > 1e-6
        tangents[valid] /= norms[valid, None]
        if not valid.all():
            heading = float(ego_state.center.heading)
            tangents[~valid] = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        left_normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

        lateral_profile = np.empty(len(route), dtype=np.float64)
        for index, distance in enumerate(progress):
            if distance <= transition_start:
                lateral_profile[index] = 0.0
            elif distance < transition_end:
                ratio = (distance - transition_start) / (transition_end - transition_start)
                lateral_profile[index] = self._smoothstep(ratio)
            elif can_return and distance > return_start:
                ratio = (distance - return_start) / (return_end - return_start)
                lateral_profile[index] = 1.0 - self._smoothstep(ratio)
            else:
                lateral_profile[index] = 1.0

        # Prefer the left lane, then try the right lane if the map or an agent blocks it.
        rejected: List[str] = []
        for side in (1.0, -1.0):
            candidate = route + (
                side * self._avoidance_lateral_offset_m * lateral_profile[:, None] * left_normals
            )
            candidate[0] = np.asarray(ego_state.center.array[:2], dtype=np.float64)
            if not self._path_is_collision_free(candidate, agents):
                rejected.append(f"{'left' if side > 0 else 'right'}: collision")
                continue
            if not self._path_stays_in_lane(candidate):
                rejected.append(f"{'left' if side > 0 else 'right'}: outside lane")
                continue
            self._avoidance_path_world = candidate
            return self._slice_cached_avoidance_path(ego_state)
        self._route_avoidance_attempts += 1
        if self._route_avoidance_attempts == 1:
            logger.warning("Route-level avoidance candidates rejected (%s)", ", ".join(rejected))
        return None

    def _slice_cached_avoidance_path(self, ego_state: EgoState) -> np.ndarray | None:
        if self._avoidance_path_world is None or len(self._avoidance_path_world) < 2:
            return None

        return self._slice_world_path(self._avoidance_path_world, ego_state)

    def _slice_world_path(self, world_path: np.ndarray, ego_state: EgoState) -> np.ndarray | None:
        """Slice a cached world path from the ego's nearest point and pad its horizon."""
        if world_path is None or len(world_path) < 2:
            return None

        current = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        nearest = int(np.argmin(np.linalg.norm(world_path - current[None, :], axis=1)))
        if nearest >= len(world_path) - 2:
            return None

        path = world_path[nearest:].copy()
        path[0] = current
        if len(path) < self._model_trajectory_steps:
            direction = path[-1] - path[-2]
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                return None
            direction /= norm
            step = max(0.1, float(np.median(np.linalg.norm(np.diff(path, axis=0), axis=1))))
            extra_count = self._model_trajectory_steps - len(path)
            extra = path[-1] + np.arange(1, extra_count + 1)[:, None] * step * direction
            path = np.vstack([path, extra])
        return path[: self._model_trajectory_steps]

    def _select_local_goal(
        self, ego_state: EgoState, reference_states: List[EgoState]
    ) -> tuple[np.ndarray | None, np.ndarray]:
        ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        reference_xy = np.asarray([state.center.array[:2] for state in reference_states], dtype=np.float64)
        if len(reference_xy) < 2:
            return None, np.zeros(1, dtype=np.float64)

        deltas = np.diff(reference_xy, axis=0)
        progress = np.concatenate([[0.0], np.cumsum(np.linalg.norm(deltas, axis=1))])
        reachable = min(self._local_goal_distance_m, float(progress[-1]))
        if reachable < self._minimum_goal_distance_m:
            return None, progress

        index = int(np.searchsorted(progress, reachable, side="left"))
        index = min(max(index, 1), len(reference_xy) - 1)
        p0, p1 = reference_xy[index - 1], reference_xy[index]
        s0, s1 = progress[index - 1], progress[index]
        weight = 0.0 if s1 <= s0 else (reachable - s0) / (s1 - s0)
        goal = p0 + weight * (p1 - p0)
        if float(np.linalg.norm(goal - ego_xy)) < self._minimum_goal_distance_m:
            return None, progress
        return goal.astype(np.float32), progress

    def _extract_agents(self, ego_state: EgoState, observation: Observation) -> List[Dict[str, Any]]:
        if not isinstance(observation, DetectionsTracks):
            raise TypeError(f"SafeFlow planner requires DetectionsTracks, got {type(observation).__name__}")

        ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        ego_box = ego_state.car_footprint.oriented_box
        candidates = []
        for tracked_object in observation.tracked_objects.tracked_objects:
            center = np.asarray(tracked_object.center.array[:2], dtype=np.float64)
            distance = float(np.linalg.norm(center - ego_xy))
            if distance > self._obstacle_query_radius_m:
                continue
            box = tracked_object.box
            collision_half_length = (
                0.5 * (float(box.length) + float(ego_box.length)) + self._obstacle_margin_m
            )
            collision_half_width = (
                0.5 * (float(box.width) + float(ego_box.width)) + self._obstacle_margin_m
            )
            velocity = np.asarray(
                [float(tracked_object.velocity.x), float(tracked_object.velocity.y)], dtype=np.float32
            )
            candidates.append(
                {
                    "center": center.astype(np.float32),
                    "velocity": velocity,
                    # CBF consumes circles. Lateral clearance is the useful
                    # radius for a lane-level bypass; final validation below
                    # preserves the longer longitudinal vehicle envelope.
                    "radius": collision_half_width,
                    "collision_half_length": collision_half_length,
                    "collision_half_width": collision_half_width,
                    "heading": float(tracked_object.center.heading),
                    "distance": distance,
                    "track_token": str(tracked_object.track_token),
                    "is_pedestrian": tracked_object.tracked_object_type == TrackedObjectType.PEDESTRIAN,
                }
            )
        candidates.sort(key=lambda item: item["distance"])
        return candidates[: self._max_agents]

    def _sample_overtake_path(
        self, ego_state: EgoState, agents: List[Dict[str, Any]], iteration: int
    ) -> np.ndarray | None:
        if self._overtake_path_world is not None:
            elapsed = max(0, int(iteration) - self._overtake_start_iteration)
            if elapsed < len(self._overtake_path_world) - 2:
                return self._slice_overtake_path(ego_state, elapsed)
            self._overtake_path_world = None
            self._overtake_start_iteration = -1

        start = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        heading = float(ego_state.center.heading)
        forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        left = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)

        lead = None
        for agent in agents:
            relative = np.asarray(agent["center"], dtype=np.float64) - start
            longitudinal = float(relative @ forward)
            lateral = float(relative @ left)
            if longitudinal < self._overtake_min_lead_distance_m:
                continue
            if longitudinal > self._overtake_max_lead_distance_m:
                continue
            if abs(lateral) > self._overtake_same_lane_width_m:
                continue
            if lead is None or longitudinal < lead["longitudinal"]:
                lead = {"longitudinal": longitudinal, "lateral": lateral}

        if lead is None and not self._force_overtake_demo:
            return None

        total_distance = max(
            self._overtake_distance_m,
            (float(lead["longitudinal"]) + self._overtake_follow_distance_m + 18.0)
            if lead is not None
            else self._overtake_distance_m,
            self._overtake_target_speed_mps * self._trajectory_sample_interval * (self._model_trajectory_steps - 1),
        )
        longitudinal = np.linspace(0.0, total_distance, self._model_trajectory_steps, dtype=np.float64)
        lane_change_start = max(0.0, self._overtake_follow_distance_m)
        lane_change_distance = max(8.0, self._overtake_lane_change_distance_m)
        lane_change_end = lane_change_start + lane_change_distance
        return_start = max(lane_change_end + 2.0, total_distance - lane_change_distance)

        lateral = np.zeros_like(longitudinal)
        offset = self._overtake_side_sign * self._overtake_lateral_offset_m
        for index, distance in enumerate(longitudinal):
            if distance < lane_change_start:
                lateral[index] = 0.0
            elif distance < lane_change_end:
                ratio = (distance - lane_change_start) / lane_change_distance
                lateral[index] = offset * self._smoothstep(ratio)
            elif not self._overtake_return_to_original_lane or distance < return_start:
                lateral[index] = offset
            else:
                ratio = (distance - return_start) / max(total_distance - return_start, 1e-6)
                lateral[index] = offset * (1.0 - self._smoothstep(ratio))

        path = start + longitudinal[:, None] * forward + lateral[:, None] * left
        path[0] = start
        if not self._overtake_path_is_safe(path, agents, lateral):
            return None
        self._overtake_path_world = path
        self._overtake_start_iteration = int(iteration)
        return self._slice_overtake_path(ego_state, 0)

    def _overtake_path_is_safe(self, path: np.ndarray, agents: List[Dict[str, Any]], lateral: np.ndarray) -> bool:
        if self._overtake_require_lane and not self._path_stays_in_lane(path):
            return False

        if not agents:
            return True

        offset_threshold = 0.65 * abs(self._overtake_lateral_offset_m)
        query_path = path[np.abs(lateral) >= offset_threshold]
        if len(query_path) == 0:
            return True
        for agent in agents:
            center = np.asarray(agent["center"], dtype=np.float64)
            min_clearance = self._overtake_min_clearance_m
            distances = np.linalg.norm(query_path - center[None, :], axis=1)
            if float(np.min(distances)) < min_clearance:
                return False
        return True

    def _path_stays_in_lane(self, path: np.ndarray) -> bool:
        if self._map_api is None:
            return not self._overtake_require_lane

        for xy in path[:: self._overtake_lane_sample_stride]:
            point = Point2D(float(xy[0]), float(xy[1]))
            if self._map_api.get_all_map_objects(point, SemanticMapLayer.LANE):
                continue
            if self._overtake_allow_lane_connector and self._map_api.get_all_map_objects(
                point, SemanticMapLayer.LANE_CONNECTOR
            ):
                continue
            return False
        return True

    def _slice_overtake_path(self, ego_state: EgoState, elapsed: int) -> np.ndarray | None:
        if self._overtake_path_world is None:
            return None

        start_index = min(max(0, elapsed), len(self._overtake_path_world) - 2)
        end_index = min(len(self._overtake_path_world), start_index + self._model_trajectory_steps)
        path = self._overtake_path_world[start_index:end_index].copy()
        if len(path) < 2:
            return None

        current_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        path += current_xy - path[0]

        if len(path) < self._model_trajectory_steps:
            last = path[-1]
            previous = path[-2]
            direction = last - previous
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                heading = float(ego_state.center.heading)
                direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
            else:
                direction = direction / norm
            step = self._overtake_target_speed_mps * self._trajectory_sample_interval
            extra_count = self._model_trajectory_steps - len(path)
            extra = last + np.arange(1, extra_count + 1, dtype=np.float64)[:, None] * step * direction
            path = np.vstack([path, extra])
        return path

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _sample_world_path(
        self,
        ego_state: EgoState,
        local_goal: np.ndarray,
        agents: List[Dict[str, Any]],
        iteration: int,
    ) -> np.ndarray:
        if self._model is None or self._sample_fm is None or self._build_can_tf_5d is None:
            raise RuntimeError("SafeFlow planner has not been initialized")

        start_xy = np.asarray(ego_state.center.array[:2], dtype=np.float32)
        transform = self._build_can_tf_5d(start_xy, local_goal)
        goal_distance = float(transform["dist"])
        theta_can = _wrap_pi(float(ego_state.center.heading) - float(transform["phi"]))
        speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        yaw_rate = float(ego_state.dynamic_car_state.angular_velocity)

        # Convert physical nuPlan speed into the checkpoint's [0.15, 0.6] scale.
        model_speed = float(np.clip(speed * self._model_speed_scale, 0.15, 0.6))
        condition = np.asarray(
            [goal_distance, model_speed, math.cos(theta_can), math.sin(theta_can), yaw_rate],
            dtype=np.float32,
        )

        ellipses = []
        prediction_times = np.linspace(
            0.0,
            self._obstacle_prediction_horizon_s,
            self._obstacle_prediction_steps,
            dtype=np.float32,
        )
        for agent in agents:
            for prediction_time in prediction_times:
                predicted_center = agent["center"] + agent["velocity"] * float(prediction_time)
                center_can = transform["w2c"](predicted_center.reshape(1, 2))[0].astype(np.float32)
                ellipses.append({"center": center_can, "radius": float(agent["radius"])})

        torch.manual_seed(self._seed + int(iteration))
        if self._device == "cuda":
            torch.cuda.manual_seed_all(self._seed + int(iteration))

        sampled, _, _ = self._sample_fm(
            model_5ch=self._model,
            T_steps=self._model_trajectory_steps,
            device=self._device,
            num_segments=self._fm_num_segments,
            ode_method=self._ode_method,
            noise_scale=self._fm_noise_scale,
            cond=condition,
            start_xy_norm=np.asarray([0.0, 0.0], dtype=np.float32),
            lock_start=True,
            goal_xy_norm=np.asarray([goal_distance, 0.0], dtype=np.float32),
            start_guidance_weight=self._start_guidance_weight,
            goal_guidance_weight=self._goal_guidance_weight,
            use_cbf=self._use_cbf,
            ellipses=ellipses,
            tau0=self._cbf_tau0,
            tau1=self._cbf_tau1,
            cbf_kwargs=self._cbf_kwargs,
        )
        path_can_raw = np.asarray(sampled[:2].T, dtype=np.float64)
        path_can = path_can_raw.copy()
        if path_can.shape != (self._model_trajectory_steps, 2) or not np.isfinite(path_can).all():
            raise ValueError(f"Invalid SafeFlow path: shape={path_can.shape}")

        path_can[0] = 0.0
        endpoint = np.asarray([goal_distance, 0.0], dtype=np.float64)
        blend_start = max(1, int(0.75 * len(path_can)))
        blend = np.linspace(0.0, 1.0, len(path_can) - blend_start, dtype=np.float64)[:, None]
        path_can[blend_start:] = (1.0 - blend) * path_can[blend_start:] + blend * endpoint
        path_can[-1] = endpoint
        world_xy = np.asarray(transform["c2w"](path_can), dtype=np.float64)
        self._dump_debug_path(
            iteration=iteration,
            ego_state=ego_state,
            local_goal=local_goal,
            condition=condition,
            path_can_raw=path_can_raw,
            path_can=path_can,
            world_xy=world_xy,
            ellipses=ellipses,
            goal_distance=goal_distance,
            theta_can=theta_can,
            model_speed=model_speed,
            yaw_rate=yaw_rate,
        )
        if self._route_geometry_guidance_weight <= 0.0 and self._looks_loopy(world_xy, local_goal):
            raise ValueError("SafeFlow produced a loopy path")
        return world_xy

    def _path_is_collision_free(self, world_xy: np.ndarray, agents: List[Dict[str, Any]]) -> bool:
        """Check ego-center samples against heading-aligned vehicle envelopes."""
        if not agents:
            return True
        path = np.asarray(world_xy, dtype=np.float64)
        for index, point in enumerate(path):
            time_s = float(index) * self._trajectory_sample_interval
            for agent in agents:
                predicted_center = np.asarray(agent["center"], dtype=np.float64) + (
                    np.asarray(agent["velocity"], dtype=np.float64) * time_s
                )
                relative = point - predicted_center
                heading = float(agent["heading"])
                longitudinal = float(relative[0] * math.cos(heading) + relative[1] * math.sin(heading))
                lateral = float(-relative[0] * math.sin(heading) + relative[1] * math.cos(heading))
                if (
                    abs(longitudinal) < float(agent["collision_half_length"])
                    and abs(lateral) < float(agent["collision_half_width"])
                ):
                    return False
        return True

    def _build_emergency_stop_trajectory(self, ego_state: EgoState) -> InterpolatedTrajectory:
        """Build a physically simple straight-line maximum-comfort braking trajectory."""
        speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
        heading = float(ego_state.center.heading)
        start = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
        stop_time = speed / self._emergency_decel_mps2
        points = []
        for index in range(self._model_trajectory_steps):
            time_s = float(index) * self._trajectory_sample_interval
            moving_time = min(time_s, stop_time)
            distance = speed * moving_time - 0.5 * self._emergency_decel_mps2 * moving_time * moving_time
            points.append(start + max(0.0, distance) * direction)
        return self._build_trajectory(ego_state, np.asarray(points, dtype=np.float64))

    def _build_stationary_trajectory(self, ego_state: EgoState) -> InterpolatedTrajectory:
        """Hold position while a pedestrian occupies the predicted crosswalk."""
        point = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        points = np.repeat(point[None, :], self._model_trajectory_steps, axis=0)
        return self._build_trajectory(ego_state, points)

    def _build_pedestrian_yield_trajectory(
        self, ego_state: EgoState, reference_xy: np.ndarray, agents: List[Dict[str, Any]]
    ) -> InterpolatedTrajectory:
        """Approach the crosswalk, then hold until the predicted crossing clears."""
        reference = np.asarray(reference_xy, dtype=np.float64)
        safe_points = [np.asarray(ego_state.center.array[:2], dtype=np.float64)]
        for index, point in enumerate(reference[1:], start=1):
            candidate = np.vstack([safe_points, point])
            if not self._path_is_collision_free(candidate, agents):
                break
            safe_points.append(point)
        path = np.asarray(safe_points, dtype=np.float64)
        if len(path) < 2:
            return self._build_stationary_trajectory(ego_state)
        if len(path) < self._model_trajectory_steps:
            path = np.vstack(
                [path, np.repeat(path[-1][None, :], self._model_trajectory_steps - len(path), axis=0)]
            )
        return self._build_trajectory(ego_state, path[: self._model_trajectory_steps])

    def _dump_debug_path(
        self,
        iteration: int,
        ego_state: EgoState,
        local_goal: np.ndarray,
        condition: np.ndarray,
        path_can_raw: np.ndarray,
        path_can: np.ndarray,
        world_xy: np.ndarray,
        ellipses: List[Dict[str, Any]],
        goal_distance: float,
        theta_can: float,
        model_speed: float,
        yaw_rate: float,
    ) -> None:
        if self._debug_output_dir is None or self._debug_dump_count >= self._debug_max_steps:
            return

        self._debug_output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"iter_{int(iteration):04d}"
        np.savez_compressed(
            self._debug_output_dir / f"{stem}.npz",
            path_can_raw=path_can_raw,
            path_can=path_can,
            world_xy=world_xy,
            local_goal=np.asarray(local_goal, dtype=np.float64),
            condition=np.asarray(condition, dtype=np.float64),
            ellipses_center=np.asarray([ellipse["center"] for ellipse in ellipses], dtype=np.float64)
            if ellipses
            else np.zeros((0, 2), dtype=np.float64),
            ellipses_radius=np.asarray([ellipse["radius"] for ellipse in ellipses], dtype=np.float64)
            if ellipses
            else np.zeros((0,), dtype=np.float64),
        )

        progress = path_can[:, 0]
        raw_progress = path_can_raw[:, 0]
        segment_lengths = np.linalg.norm(np.diff(path_can, axis=0), axis=1)
        raw_segment_lengths = np.linalg.norm(np.diff(path_can_raw, axis=0), axis=1)
        summary = {
            "iteration": int(iteration),
            "ego": {
                "x": float(ego_state.center.x),
                "y": float(ego_state.center.y),
                "heading": float(ego_state.center.heading),
            },
            "local_goal": [float(local_goal[0]), float(local_goal[1])],
            "condition": [float(value) for value in condition.tolist()],
            "goal_distance": float(goal_distance),
            "theta_can": float(theta_can),
            "model_speed": float(model_speed),
            "yaw_rate": float(yaw_rate),
            "num_ellipses": len(ellipses),
            "ellipses": [
                {
                    "center": [float(ellipse["center"][0]), float(ellipse["center"][1])],
                    "radius": float(ellipse["radius"]),
                }
                for ellipse in ellipses
            ],
            "raw": {
                "x_min": float(np.min(path_can_raw[:, 0])),
                "x_max": float(np.max(path_can_raw[:, 0])),
                "y_min": float(np.min(path_can_raw[:, 1])),
                "y_max": float(np.max(path_can_raw[:, 1])),
                "path_length": float(np.sum(raw_segment_lengths)),
                "negative_progress_steps": int(np.sum(np.diff(raw_progress) < -1e-3)),
                "min_progress_delta": float(np.min(np.diff(raw_progress))),
            },
            "blended": {
                "x_min": float(np.min(path_can[:, 0])),
                "x_max": float(np.max(path_can[:, 0])),
                "y_min": float(np.min(path_can[:, 1])),
                "y_max": float(np.max(path_can[:, 1])),
                "path_length": float(np.sum(segment_lengths)),
                "direct_distance": float(np.linalg.norm(path_can[-1] - path_can[0])),
                "negative_progress_steps": int(np.sum(np.diff(progress) < -1e-3)),
                "min_progress_delta": float(np.min(np.diff(progress))),
            },
        }
        with open(self._debug_output_dir / f"{stem}.json", "w", encoding="utf-8") as file:
            import json

            json.dump(summary, file, indent=2)

        csv_path = self._debug_output_dir / f"{stem}.csv"
        with open(csv_path, "w", encoding="utf-8") as file:
            file.write("index,raw_x,raw_y,blend_x,blend_y,world_x,world_y\n")
            for index, (raw, blended, world) in enumerate(zip(path_can_raw, path_can, world_xy)):
                file.write(
                    f"{index},{raw[0]:.9f},{raw[1]:.9f},{blended[0]:.9f},{blended[1]:.9f},"
                    f"{world[0]:.9f},{world[1]:.9f}\n"
                )

        logger.info("Dumped SafeFlow debug path: %s", csv_path)
        self._debug_dump_count += 1

    def _looks_loopy(self, world_xy: np.ndarray, local_goal: np.ndarray) -> bool:
        if len(world_xy) < 4:
            return False

        segment_lengths = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
        path_length = float(segment_lengths.sum())
        direct_distance = float(np.linalg.norm(world_xy[-1] - world_xy[0]))
        goal_distance = float(np.linalg.norm(local_goal - world_xy[0]))
        if goal_distance < 1e-6:
            return False

        if direct_distance > 1e-6 and path_length > 3.0 * direct_distance + 2.0:
            return True

        goal_direction = (local_goal - world_xy[0]) / goal_distance
        progress = (world_xy - world_xy[0]) @ goal_direction
        if float(np.min(np.diff(progress))) < -0.5:
            return True

        if float(progress[-1]) < 0.5 * goal_distance and path_length > 1.5 * goal_distance:
            return True

        return False

    def _time_parameterize(self, world_xy: np.ndarray, reference_progress: np.ndarray) -> np.ndarray:
        segment_lengths = np.linalg.norm(np.diff(world_xy, axis=0), axis=1)
        path_progress = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if path_progress[-1] < 1e-6:
            raise ValueError("SafeFlow produced a zero-length path")

        count = self._model_trajectory_steps
        ref = reference_progress[:count]
        if len(ref) < count:
            ref = np.pad(ref, (0, count - len(ref)), mode="edge")
        target_progress = min(self._local_goal_distance_m, float(reference_progress[-1]))
        fraction = np.clip(ref / max(target_progress, 1e-6), 0.0, 1.0)
        query = fraction * path_progress[-1]

        result = np.column_stack(
            [
                np.interp(query, path_progress, world_xy[:, 0]),
                np.interp(query, path_progress, world_xy[:, 1]),
            ]
        )
        result[0] = world_xy[0]

        max_speed = max(self._max_output_speed_mps, 1.0)
        max_step = max_speed * self._trajectory_sample_interval
        for index in range(1, len(result)):
            delta = result[index] - result[index - 1]
            distance = float(np.linalg.norm(delta))
            if distance > max_step:
                result[index] = result[index - 1] + delta * (max_step / distance)
        return result

    def _build_trajectory(self, ego_state: EgoState, world_xy: np.ndarray) -> InterpolatedTrajectory:
        if len(world_xy) < 2:
            raise ValueError("SafeFlow trajectory needs at least two poses")

        dt = self._trajectory_sample_interval
        headings = np.empty(len(world_xy), dtype=np.float64)
        headings[0] = float(ego_state.center.heading)
        lookahead = 4
        max_heading_step = 0.45 * dt
        for index in range(1, len(world_xy)):
            target_index = min(len(world_xy) - 1, index + lookahead)
            delta = world_xy[target_index] - world_xy[index - 1]
            if float(np.linalg.norm(delta)) > 1e-5:
                raw_heading = math.atan2(float(delta[1]), float(delta[0]))
            else:
                raw_heading = headings[index - 1]
            heading_delta = _wrap_pi(float(raw_heading - headings[index - 1]))
            heading_delta = float(np.clip(heading_delta, -max_heading_step, max_heading_step))
            headings[index] = headings[index - 1] + heading_delta

        states: List[EgoState] = [ego_state]
        previous_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        vehicle = ego_state.car_footprint.vehicle_parameters
        for index in range(1, len(world_xy)):
            distance = float(np.linalg.norm(world_xy[index] - world_xy[index - 1]))
            speed = distance / dt
            acceleration = (speed - previous_speed) / dt
            angular_velocity = _wrap_pi(float(headings[index] - headings[index - 1])) / dt
            time_point = TimePoint(ego_state.time_us + int(round(index * dt * 1e6)))
            states.append(
                EgoState.build_from_center(
                    center=StateSE2(float(world_xy[index, 0]), float(world_xy[index, 1]), float(headings[index])),
                    center_velocity_2d=StateVector2D(speed, 0.0),
                    center_acceleration_2d=StateVector2D(acceleration, 0.0),
                    tire_steering_angle=0.0,
                    time_point=time_point,
                    vehicle_parameters=vehicle,
                    angular_vel=angular_velocity,
                )
            )
            previous_speed = speed
        return InterpolatedTrajectory(states)

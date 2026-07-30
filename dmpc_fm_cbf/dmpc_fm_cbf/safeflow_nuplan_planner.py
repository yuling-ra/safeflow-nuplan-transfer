from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

logger = logging.getLogger(__name__)


def _wrap_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


class SafeFlowNuPlanPlanner(AbstractPlanner):
    """Expose the canonical 5-channel SafeFlow model as a nuPlan planner."""

    requires_scenario = False

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
        max_output_speed_mps: float = 15.0,
        seed: int = 7,
        idm_target_velocity: float = 8.0,
        idm_min_gap_to_lead_agent: float = 1.0,
        idm_headway_time: float = 1.5,
        idm_accel_max: float = 1.0,
        idm_decel_max: float = 3.0,
        idm_occupancy_map_radius: float = 40.0,
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
        self._max_output_speed_mps = float(max_output_speed_mps)
        self._seed = int(seed)

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

    def name(self) -> str:
        return "safeflow_fm_cbf" if self._use_cbf else "safeflow_fm"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._reference_planner.initialize(initialization)
        self._load_model()

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

        try:
            checkpoint = torch.load(self._ckpt_path, map_location=self._device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(self._ckpt_path, map_location=self._device)
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
        reference = self._reference_planner.compute_planner_trajectory(current_input)
        reference_states = reference.get_sampled_trajectory()
        ego_state, observation = current_input.history.current_state

        try:
            local_goal, reference_progress = self._select_local_goal(ego_state, reference_states)
            if local_goal is None:
                return reference

            agents = self._extract_agents(ego_state, observation)
            world_xy = self._sample_world_path(ego_state, local_goal, agents, current_input.iteration.index)
            world_xy = self._time_parameterize(world_xy, reference_progress)
            trajectory = self._build_trajectory(ego_state, world_xy)
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
            logger.exception("SafeFlow inference failed; using the route-aware IDM fallback for this step")
            return reference

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
        candidates = []
        for tracked_object in observation.tracked_objects.tracked_objects:
            center = np.asarray(tracked_object.center.array[:2], dtype=np.float64)
            distance = float(np.linalg.norm(center - ego_xy))
            if distance > self._obstacle_query_radius_m:
                continue
            box = tracked_object.box
            radius = 0.5 * math.hypot(float(box.width), float(box.length)) + self._obstacle_margin_m
            candidates.append({"center": center.astype(np.float32), "radius": radius, "distance": distance})
        candidates.sort(key=lambda item: item["distance"])
        return candidates[: self._max_agents]

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

        # Training speeds were sampled from [0.15, 0.6] m/s.
        model_speed = float(np.clip(speed, 0.15, 0.6))
        condition = np.asarray(
            [goal_distance, model_speed, math.cos(theta_can), math.sin(theta_can), yaw_rate],
            dtype=np.float32,
        )

        ellipses = []
        for agent in agents:
            center_can = transform["w2c"](agent["center"].reshape(1, 2))[0].astype(np.float32)
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
        path_can = np.asarray(sampled[:2].T, dtype=np.float64)
        if path_can.shape != (self._model_trajectory_steps, 2) or not np.isfinite(path_can).all():
            raise ValueError(f"Invalid SafeFlow path: shape={path_can.shape}")

        path_can[0] = 0.0
        endpoint = np.asarray([goal_distance, 0.0], dtype=np.float64)
        blend_start = max(1, int(0.75 * len(path_can)))
        blend = np.linspace(0.0, 1.0, len(path_can) - blend_start, dtype=np.float64)[:, None]
        path_can[blend_start:] = (1.0 - blend) * path_can[blend_start:] + blend * endpoint
        path_can[-1] = endpoint
        return np.asarray(transform["c2w"](path_can), dtype=np.float64)

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
        for index in range(1, len(world_xy)):
            delta = world_xy[index] - world_xy[index - 1]
            if float(np.linalg.norm(delta)) > 1e-5:
                headings[index] = math.atan2(float(delta[1]), float(delta[0]))
            else:
                headings[index] = headings[index - 1]

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

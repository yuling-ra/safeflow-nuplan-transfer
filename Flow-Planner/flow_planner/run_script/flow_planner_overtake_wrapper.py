"""Optional collision-checked overtake layer for the Flow-Planner checkpoint.

The released checkpoint is kept intact.  This module only changes the
trajectory after an explicit same-lane lead vehicle is detected, and is used
by the opt-in straight-road demonstration launcher.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from flow_planner.planner import FlowPlanner


class FlowPlannerOvertake(FlowPlanner):
    """Run Flow-Planner with an explicit, map-checked overtake fallback."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._overtake_token: Optional[str] = None
        self._overtake_side: Optional[float] = None
        self._returning = False

    def name(self) -> str:
        return "diffusion_planner_overtake_wrapper"

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _relative(ego: EgoState, obj: Any) -> Tuple[float, float]:
        heading = float(ego.center.heading)
        forward = np.asarray([math.cos(heading), math.sin(heading)])
        left = np.asarray([-math.sin(heading), math.cos(heading)])
        relative = np.asarray(obj.center.array[:2], dtype=np.float64) - np.asarray(
            ego.center.array[:2], dtype=np.float64
        )
        return float(relative @ forward), float(relative @ left)

    def _find_lead(self, ego: EgoState, observation: Any) -> Optional[Any]:
        if not isinstance(observation, DetectionsTracks):
            return None
        vehicles = [
            obj
            for obj in observation.tracked_objects.tracked_objects
            if obj.tracked_object_type == TrackedObjectType.VEHICLE
        ]
        if self._overtake_token is not None:
            tracked = next((obj for obj in vehicles if str(obj.track_token) == self._overtake_token), None)
            if tracked is not None:
                longitudinal, _ = self._relative(ego, tracked)
                if longitudinal > -8.0:
                    return tracked
        candidates = []
        for obj in vehicles:
            longitudinal, lateral = self._relative(ego, obj)
            if 5.0 <= longitudinal <= 45.0 and abs(lateral) <= 2.5:
                candidates.append((longitudinal, obj))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _lane_available(self, points: np.ndarray) -> bool:
        if self._map_api is None:
            return True
        for point in points[::4]:
            map_point = Point2D(float(point[0]), float(point[1]))
            in_lane = self._map_api.get_all_map_objects(map_point, SemanticMapLayer.LANE)
            in_connector = self._map_api.get_all_map_objects(map_point, SemanticMapLayer.LANE_CONNECTOR)
            if not in_lane and not in_connector:
                return False
        return True

    def _path_is_safe(self, ego: EgoState, points: np.ndarray, headings: np.ndarray, observation: Any) -> bool:
        if not isinstance(observation, DetectionsTracks):
            return True
        tracked = observation.tracked_objects.tracked_objects
        dt = self._future_trajectory_sampling.time_horizon / self._future_trajectory_sampling.num_poses
        for index, point in enumerate(points):
            time_s = index * dt
            ego_box = OrientedBox(
                StateSE2(float(point[0]), float(point[1]), float(headings[index])),
                ego.car_footprint.oriented_box.length,
                ego.car_footprint.oriented_box.width,
                ego.car_footprint.oriented_box.height,
            )
            for obj in tracked:
                if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                    continue
                center = np.asarray(obj.center.array[:2], dtype=np.float64)
                center += np.asarray([float(obj.velocity.x), float(obj.velocity.y)]) * time_s
                predicted = OrientedBox(
                    StateSE2(float(center[0]), float(center[1]), float(obj.center.heading)),
                    obj.box.length,
                    obj.box.width,
                    obj.box.height,
                )
                if ego_box.geometry.intersects(predicted.geometry):
                    return False
        return True

    def _build_states(self, ego: EgoState, points: np.ndarray, headings: np.ndarray) -> InterpolatedTrajectory:
        dt = self._future_trajectory_sampling.time_horizon / self._future_trajectory_sampling.num_poses
        states: List[EgoState] = [ego]
        previous_speed = float(ego.dynamic_car_state.speed)
        for index in range(1, len(points)):
            distance = float(np.linalg.norm(points[index] - points[index - 1]))
            speed = distance / max(dt, 1e-3)
            acceleration = (speed - previous_speed) / max(dt, 1e-3)
            yaw_rate = (float(headings[index]) - float(headings[index - 1]) + math.pi) % (2.0 * math.pi)
            yaw_rate = (yaw_rate - math.pi) / max(dt, 1e-3)
            states.append(
                EgoState.build_from_center(
                    center=StateSE2(float(points[index, 0]), float(points[index, 1]), float(headings[index])),
                    center_velocity_2d=StateVector2D(speed, 0.0),
                    center_acceleration_2d=StateVector2D(acceleration, 0.0),
                    tire_steering_angle=0.0,
                    time_point=TimePoint(ego.time_us + int(round(index * dt * 1e6))),
                    vehicle_parameters=ego.car_footprint.vehicle_parameters,
                    angular_vel=yaw_rate,
                )
            )
            previous_speed = speed
        return InterpolatedTrajectory(states)

    def _make_overtake_path(self, ego: EgoState, lead: Any, observation: Any, side: float) -> Optional[InterpolatedTrajectory]:
        dt = self._future_trajectory_sampling.time_horizon / self._future_trajectory_sampling.num_poses
        count = self._future_trajectory_sampling.num_poses
        current_speed = max(float(ego.dynamic_car_state.speed), 5.5)
        target_speed = max(current_speed + 1.0, 8.0)
        distances = np.arange(count, dtype=np.float64) * target_speed * dt
        heading = float(ego.center.heading)
        forward = np.asarray([math.cos(heading), math.sin(heading)])
        left = np.asarray([-math.sin(heading), math.cos(heading)])
        start = np.asarray(ego.center.array[:2], dtype=np.float64)

        lead_longitudinal, _ = self._relative(ego, lead)
        if self._returning:
            # The ego is already in the passing lane.  Shift one lane-width
            # back toward the original lane over 12 m.
            lateral = -side * 3.7 * np.asarray([self._smoothstep(distance / 12.0) for distance in distances])
        else:
            change_start = max(0.0, lead_longitudinal - 14.0)
            change_end = change_start + 10.0
            return_start = max(change_end + 4.0, lead_longitudinal + 10.0)
            return_end = return_start + 10.0
            lateral = np.zeros(count, dtype=np.float64)
            for index, distance in enumerate(distances):
                if distance < change_start:
                    continue
                if distance < change_end:
                    lateral[index] = side * 3.7 * self._smoothstep((distance - change_start) / 10.0)
                elif distance < return_start:
                    lateral[index] = side * 3.7
                else:
                    lateral[index] = side * 3.7 * (1.0 - self._smoothstep((distance - return_start) / 10.0))

        points = start + distances[:, None] * forward + lateral[:, None] * left
        points[0] = start
        deltas = np.diff(points, axis=0, prepend=points[[0]])
        headings = np.full(count, heading, dtype=np.float64)
        for index in range(1, count):
            if np.linalg.norm(deltas[index]) > 1e-5:
                headings[index] = math.atan2(float(deltas[index, 1]), float(deltas[index, 0]))
        if not self._lane_available(points) or not self._path_is_safe(ego, points, headings, observation):
            return None
        return self._build_states(ego, points, headings)

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego, observation = current_input.history.current_state
        lead = self._find_lead(ego, observation)
        if lead is not None:
            token = str(lead.track_token)
            longitudinal, _ = self._relative(ego, lead)
            if self._overtake_token is None:
                self._overtake_token = token
                for side in (1.0, -1.0):
                    trajectory = self._make_overtake_path(ego, lead, observation, side)
                    if trajectory is not None:
                        self._overtake_side = side
                        self._returning = False
                        return trajectory
                self._overtake_token = None
            elif self._overtake_side is not None:
                # Start rejoining as soon as the ego front has cleared the
                # lead.  Waiting until the lead is far behind leaves the
                # demonstration in the adjacent lane at the end of the log.
                if longitudinal < 5.0:
                    self._returning = True
                trajectory = self._make_overtake_path(ego, lead, observation, self._overtake_side)
                if trajectory is not None:
                    return trajectory
        else:
            self._overtake_token = None
            self._overtake_side = None
            self._returning = False
        return super().compute_planner_trajectory(current_input)

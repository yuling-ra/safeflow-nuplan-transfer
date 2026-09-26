from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, DefaultDict, Generator, List, Optional, Set, Tuple

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatuses, Transform
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class SingleAgentScenario(AbstractScenario):
    """Scenario proxy that slows one selected vehicle in recorded traffic."""

    def __init__(self, scenario: AbstractScenario) -> None:
        self._scenario = scenario
        self._vehicle_speed_scale = self._env_float("SINGLE_AGENT_VEHICLE_SPEED_SCALE", 0.35)
        self._vehicle_position_scale = self._env_float(
            "SINGLE_AGENT_VEHICLE_POSITION_SCALE", self._vehicle_speed_scale
        )
        self._vehicle_straight = self._env_bool("SINGLE_AGENT_VEHICLE_STRAIGHT", True)
        # Preserve surrounding traffic for learned-planner inputs while
        # slowing only the selected lead vehicle.
        self._keep_other_vehicles = self._env_bool("SINGLE_AGENT_KEEP_OTHER_VEHICLES", False)
        self._track_token = self._select_track_token()
        self._initial_track_pose = self._find_initial_track_pose()
        self._excluded_track_tokens: Set[str] = set()
        if self._keep_other_vehicles:
            self._exclude_replay_conflicts()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return max(0.0, float(value))
        except ValueError:
            return default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    def _select_track_token(self) -> Optional[str]:
        total_iterations = self._scenario.get_number_of_iterations()
        if total_iterations <= 0:
            return None

        sample_count = min(5, total_iterations)
        vehicle_stats: DefaultDict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        token_counts: DefaultDict[str, int] = defaultdict(int)

        for iteration in range(sample_count):
            ego = self._scenario.get_ego_state_at_iteration(iteration)
            ego_xy = ego.center.array[:2]
            ego_heading = float(ego.center.heading)
            forward = (math.cos(ego_heading), math.sin(ego_heading))
            left = (-math.sin(ego_heading), math.cos(ego_heading))
            for obj in self._scenario.get_tracked_objects_at_iteration(iteration).tracked_objects:
                if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                    continue
                xy = obj.center.array[:2]
                dx = float(xy[0] - ego_xy[0])
                dy = float(xy[1] - ego_xy[1])
                distance_squared = dx * dx + dy * dy
                longitudinal = dx * forward[0] + dy * forward[1]
                lateral = dx * left[0] + dy * left[1]
                token = obj.track_token
                token_counts[token] += 1
                vehicle_stats[token].append((distance_squared, longitudinal, lateral))

        if not vehicle_stats:
            return None

        def ranking(item: Tuple[str, List[Tuple[float, float, float]]]) -> Tuple[int, int, float, float]:
            token, stats = item
            front_same_lane = [
                (distance_squared, longitudinal)
                for distance_squared, longitudinal, lateral in stats
                if 4.0 <= longitudinal <= 35.0 and abs(lateral) <= 3.0
            ]
            if front_same_lane:
                nearest_front = min(front_same_lane, key=lambda value: value[1])
                return (0, -token_counts[token], nearest_front[1], nearest_front[0])
            return (1, -token_counts[token], min(distance_squared for distance_squared, _, _ in stats), 0.0)

        return min(vehicle_stats.items(), key=ranking)[0]

    def _find_initial_track_pose(self) -> Optional[Tuple[float, float, float]]:
        if self._track_token is None:
            return None

        for iteration in range(self._scenario.get_number_of_iterations()):
            for obj in self._scenario.get_tracked_objects_at_iteration(iteration).tracked_objects:
                if obj.tracked_object_type == TrackedObjectType.VEHICLE and obj.track_token == self._track_token:
                    center = obj.center
                    return (float(center.x), float(center.y), float(center.heading))
        return None

    def _slow_selected_object(self, obj: Any) -> Any:
        if not isinstance(obj, Agent):
            return obj

        center = obj.center
        if self._initial_track_pose is not None:
            initial_x, initial_y, initial_heading = self._initial_track_pose
            dx = float(center.x) - initial_x
            dy = float(center.y) - initial_y
            if self._vehicle_straight:
                forward_x = math.cos(initial_heading)
                forward_y = math.sin(initial_heading)
                longitudinal = dx * forward_x + dy * forward_y
                center = StateSE2(
                    initial_x + longitudinal * self._vehicle_position_scale * forward_x,
                    initial_y + longitudinal * self._vehicle_position_scale * forward_y,
                    initial_heading,
                )
            elif self._vehicle_position_scale != 1.0:
                center = StateSE2(
                    initial_x + dx * self._vehicle_position_scale,
                    initial_y + dy * self._vehicle_position_scale,
                    float(center.heading),
                )

        box = obj.box
        slowed_box = OrientedBox(center, box.length, box.width, box.height)
        velocity_x = float(obj.velocity.x) * self._vehicle_speed_scale
        velocity_y = float(obj.velocity.y) * self._vehicle_speed_scale
        slowed_angular_velocity = 0.0 if self._vehicle_straight else (
            float(obj.angular_velocity) * self._vehicle_speed_scale if obj.angular_velocity is not None else None
        )
        if self._vehicle_straight and self._initial_track_pose is not None:
            _, _, initial_heading = self._initial_track_pose
            forward_x = math.cos(initial_heading)
            forward_y = math.sin(initial_heading)
            longitudinal_speed = max(0.0, velocity_x * forward_x + velocity_y * forward_y)
            velocity_x = longitudinal_speed * forward_x
            velocity_y = longitudinal_speed * forward_y
        slowed_velocity = StateVector2D(velocity_x, velocity_y)

        return Agent(
            tracked_object_type=obj.tracked_object_type,
            oriented_box=slowed_box,
            velocity=slowed_velocity,
            metadata=obj.metadata,
            angular_velocity=slowed_angular_velocity,
            predictions=None,
            past_trajectory=None,
        )

    def _exclude_replay_conflicts(self) -> None:
        """Remove open-loop rear traffic that cannot react to a slowed lead."""
        if self._track_token is None or self._scenario.get_number_of_iterations() <= 0:
            return

        for iteration in range(self._scenario.get_number_of_iterations()):
            ego = self._scenario.get_ego_state_at_iteration(iteration)
            heading = float(ego.center.heading)
            forward = (math.cos(heading), math.sin(heading))
            left = (-math.sin(heading), math.cos(heading))
            vehicles = [
                obj
                for obj in self._scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
                if obj.tracked_object_type == TrackedObjectType.VEHICLE
            ]
            selected = next((obj for obj in vehicles if obj.track_token == self._track_token), None)
            slowed_selected = self._slow_selected_object(selected) if selected is not None else None

            for obj in vehicles:
                if obj.track_token == self._track_token:
                    continue
                dx = float(obj.center.x - ego.center.x)
                dy = float(obj.center.y - ego.center.y)
                longitudinal = dx * forward[0] + dy * forward[1]
                lateral = dx * left[0] + dy * left[1]
                # A non-reactive vehicle starting behind ego in the same lane
                # will catch ego after the lead vehicle is slowed.  Keeping it
                # would create an artificial rear-end collision in the demo.
                if longitudinal < -4.0 and abs(lateral) < 3.5:
                    self._excluded_track_tokens.add(str(obj.track_token))
                if slowed_selected is not None and slowed_selected.box.geometry.distance(obj.box.geometry) <= 0.25:
                    self._excluded_track_tokens.add(str(obj.track_token))

            # Also remove replay pairs that already collide in the source
            # observations; they are unrelated to the planner's behavior.
            for first_index, first in enumerate(vehicles):
                for second in vehicles[first_index + 1 :]:
                    if first.box.geometry.intersects(second.box.geometry):
                        self._excluded_track_tokens.update((str(first.track_token), str(second.track_token)))

    def _filter(self, detections: DetectionsTracks) -> DetectionsTracks:
        if self._track_token is None:
            return DetectionsTracks(TrackedObjects([]))
        objects = []
        for obj in detections.tracked_objects:
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            if str(obj.track_token) in self._excluded_track_tokens:
                continue
            if obj.track_token == self._track_token:
                objects.append(self._slow_selected_object(obj))
            elif self._keep_other_vehicles:
                objects.append(obj)
        return DetectionsTracks(TrackedObjects(objects))

    @property
    def token(self) -> str:
        return self._scenario.token

    @property
    def log_name(self) -> str:
        return self._scenario.log_name

    @property
    def scenario_name(self) -> str:
        suffix = "multicar_overtake" if self._keep_other_vehicles else "single_agent"
        return f"{self._scenario.scenario_name}_{suffix}"

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        return self._scenario.ego_vehicle_parameters

    @property
    def scenario_type(self) -> str:
        suffix = "multicar_overtake" if self._keep_other_vehicles else "single_agent"
        return f"{self._scenario.scenario_type}_{suffix}"

    @property
    def map_api(self) -> AbstractMap:
        return self._scenario.map_api

    @property
    def database_interval(self) -> float:
        return self._scenario.database_interval

    def get_number_of_iterations(self) -> int:
        return self._scenario.get_number_of_iterations()

    def get_time_point(self, iteration: int) -> TimePoint:
        return self._scenario.get_time_point(iteration)

    def get_lidar_to_ego_transform(self) -> Transform:
        return self._scenario.get_lidar_to_ego_transform()

    def get_mission_goal(self) -> Optional[StateSE2]:
        return self._scenario.get_mission_goal()

    def get_route_roadblock_ids(self) -> List[str]:
        return self._scenario.get_route_roadblock_ids()

    def get_expert_goal_state(self) -> StateSE2:
        return self._scenario.get_expert_goal_state()

    def get_tracked_objects_at_iteration(
        self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling] = None
    ) -> DetectionsTracks:
        return self._filter(self._scenario.get_tracked_objects_at_iteration(iteration, future_trajectory_sampling))

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        track_tokens = None if self._keep_other_vehicles else (
            {self._track_token} if self._track_token is not None else set()
        )
        if filter_track_tokens is not None:
            track_tokens = filter_track_tokens if track_tokens is None else track_tokens & filter_track_tokens
        return self._filter(
            self._scenario.get_tracked_objects_within_time_window_at_iteration(
                iteration, past_time_horizon, future_time_horizon, track_tokens, future_trajectory_sampling
            )
        )

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        return self._scenario.get_sensors_at_iteration(iteration, channels)

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        return self._scenario.get_ego_state_at_iteration(iteration)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        return self._scenario.get_traffic_light_status_at_iteration(iteration)

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        return self._scenario.get_past_traffic_light_status_history(iteration, time_horizon, num_samples)

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        return self._scenario.get_future_traffic_light_status_history(iteration, time_horizon, num_samples)

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        return self._scenario.get_future_timestamps(iteration, time_horizon, num_samples)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        return self._scenario.get_past_timestamps(iteration, time_horizon, num_samples)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        return self._scenario.get_ego_future_trajectory(iteration, time_horizon, num_samples)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        return self._scenario.get_ego_past_trajectory(iteration, time_horizon, num_samples)

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        return self._scenario.get_past_sensors(iteration, time_horizon, num_samples, channels)

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        for detections in self._scenario.get_past_tracked_objects(
            iteration, time_horizon, num_samples, future_trajectory_sampling
        ):
            yield self._filter(detections)

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        for detections in self._scenario.get_future_tracked_objects(
            iteration, time_horizon, num_samples, future_trajectory_sampling
        ):
            yield self._filter(detections)

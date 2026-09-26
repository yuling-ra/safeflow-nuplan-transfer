"""Strict three-vehicle scenario with one oncoming and one left-turning vehicle."""

from __future__ import annotations

import math
import os
from bisect import bisect_left
from collections import defaultdict
from typing import DefaultDict, Dict, Generator, List, Optional, Set, Tuple

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


class ThreeVehicleScenario(AbstractScenario):
    """Keep ego plus exactly two recorded vehicle tracks.

    The two roles are an oncoming vehicle and a left-turning vehicle.  Their
    recorded nuPlan poses are preserved so they remain on map-valid traffic
    paths.  Before/after a recorded track is visible, its endpoint is extended
    using the recorded velocity so the observation always contains two agents.
    """

    def __init__(self, scenario: AbstractScenario) -> None:
        self._scenario = scenario
        self._tracks = self._collect_vehicle_tracks()
        self._role_tokens = self._select_role_tokens()
        self._track_tokens = set(self._role_tokens.values())
        if len(self._track_tokens) != 2:
            raise RuntimeError(f"ThreeVehicleScenario requires two distinct tracks, got {self._role_tokens}")
        self._track_lookup = {
            token: {iteration: agent for iteration, agent in samples}
            for token, samples in self._tracks.items()
            if token in self._track_tokens
        }

    @property
    def role_tokens(self) -> Dict[str, str]:
        """Return the selected token for each traffic role."""
        return dict(self._role_tokens)

    def _collect_vehicle_tracks(self) -> DefaultDict[str, List[Tuple[int, Agent]]]:
        """Collect recorded vehicle tracks over the complete scenario."""
        tracks: DefaultDict[str, List[Tuple[int, Agent]]] = defaultdict(list)
        total_iterations = self._scenario.get_number_of_iterations()
        if total_iterations <= 0:
            raise RuntimeError("Cannot build a three-vehicle scenario with no iterations")
        for iteration in range(total_iterations):
            for obj in self._scenario.get_tracked_objects_at_iteration(iteration).tracked_objects:
                if obj.tracked_object_type == TrackedObjectType.VEHICLE and isinstance(obj, Agent):
                    tracks[obj.track_token].append((iteration, obj))
        return tracks

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _track_stats(self, token: str) -> Tuple[float, float, int, float]:
        """Return min distance, heading change, oncoming evidence, and path length."""
        samples = self._tracks[token]
        min_distance = float("inf")
        oncoming_evidence = 0
        headings: List[float] = []
        for iteration, agent in samples:
            ego = self._scenario.get_ego_state_at_iteration(iteration)
            dx = float(agent.center.x - ego.center.x)
            dy = float(agent.center.y - ego.center.y)
            distance = math.hypot(dx, dy)
            min_distance = min(min_distance, distance)
            forward_x = math.cos(float(ego.center.heading))
            forward_y = math.sin(float(ego.center.heading))
            longitudinal = dx * forward_x + dy * forward_y
            velocity_along_ego = float(agent.velocity.x) * forward_x + float(agent.velocity.y) * forward_y
            if distance <= 45.0 and longitudinal >= -10.0 and velocity_along_ego <= -2.0:
                oncoming_evidence += 1
            headings.append(float(agent.center.heading))

        heading_change = 0.0
        for previous, current in zip(headings, headings[1:]):
            heading_change += self._wrap_pi(current - previous)
        first = samples[0][1].center
        last = samples[-1][1].center
        path_length = math.hypot(float(last.x - first.x), float(last.y - first.y))
        return min_distance, abs(heading_change), oncoming_evidence, path_length

    def _select_role_tokens(self) -> Dict[str, str]:
        """Select and validate the oncoming and left-turning tracks."""
        explicit = {
            "oncoming": os.getenv("THREE_VEHICLE_ONCOMING_TOKEN", "").strip(),
            "left_turn": os.getenv("THREE_VEHICLE_LEFT_TURN_TOKEN", "").strip(),
        }
        if all(explicit.values()):
            missing = [token for token in explicit.values() if token not in self._tracks]
            if missing:
                raise RuntimeError(f"Configured three-vehicle tracks are absent from the scenario: {missing}")
            selected = explicit
        else:
            stats = {token: self._track_stats(token) for token in self._tracks}
            oncoming = [
                (value[0], -value[2], token)
                for token, value in stats.items()
                if value[2] >= 5 and len(self._tracks[token]) >= 20
            ]
            if not oncoming:
                raise RuntimeError("No persistent oncoming vehicle track was found")
            oncoming.sort()
            oncoming_token = oncoming[0][2]
            left_turn = [
                (value[0], -value[1], token)
                for token, value in stats.items()
                if token != oncoming_token
                and value[1] >= math.radians(35.0)
                and value[3] >= 8.0
                and len(self._tracks[token]) >= 20
            ]
            if not left_turn:
                raise RuntimeError("No persistent left-turning vehicle track was found")
            left_turn.sort()
            selected = {"oncoming": oncoming_token, "left_turn": left_turn[0][2]}

        oncoming_stats = self._track_stats(selected["oncoming"])
        left_turn_stats = self._track_stats(selected["left_turn"])
        if oncoming_stats[2] < 5:
            raise RuntimeError(f"Configured oncoming track is not moving against ego traffic: {selected['oncoming']}")
        if left_turn_stats[1] < math.radians(35.0) or left_turn_stats[3] < 8.0:
            raise RuntimeError(f"Configured left-turn track does not execute a real turn: {selected['left_turn']}")
        return selected

    @staticmethod
    def _clone_agent(agent: Agent, center: StateSE2) -> Agent:
        box = agent.box
        return Agent(
            tracked_object_type=agent.tracked_object_type,
            oriented_box=OrientedBox(center, box.length, box.width, box.height),
            velocity=StateVector2D(float(agent.velocity.x), float(agent.velocity.y)),
            metadata=agent.metadata,
            angular_velocity=agent.angular_velocity,
            predictions=None,
            past_trajectory=None,
        )

    def _agent_at_iteration(self, token: str, iteration: int) -> Agent:
        """Return a map-path-preserving agent pose for every scenario iteration."""
        direct = self._track_lookup[token].get(iteration)
        if direct is not None:
            return direct

        samples = self._tracks[token]
        sample_iterations = [sample_iteration for sample_iteration, _ in samples]
        insertion = bisect_left(sample_iterations, iteration)
        if insertion == 0 or insertion == len(samples):
            source_iteration, source = samples[0] if insertion == 0 else samples[-1]
            dt = float(iteration - source_iteration) * float(self.database_interval)
            center = StateSE2(
                float(source.center.x) + float(source.velocity.x) * dt,
                float(source.center.y) + float(source.velocity.y) * dt,
                float(source.center.heading),
            )
            return self._clone_agent(source, center)

        before_iteration, before = samples[insertion - 1]
        after_iteration, after = samples[insertion]
        ratio = float(iteration - before_iteration) / float(after_iteration - before_iteration)
        heading_delta = self._wrap_pi(float(after.center.heading - before.center.heading))
        center = StateSE2(
            float(before.center.x) + ratio * float(after.center.x - before.center.x),
            float(before.center.y) + ratio * float(after.center.y - before.center.y),
            float(before.center.heading) + ratio * heading_delta,
        )
        return self._clone_agent(before, center)

    def _detections_at_iteration(
        self, iteration: int, filter_track_tokens: Optional[Set[str]] = None
    ) -> DetectionsTracks:
        tokens = self._track_tokens if filter_track_tokens is None else self._track_tokens & filter_track_tokens
        ordered_tokens = [self._role_tokens["oncoming"], self._role_tokens["left_turn"]]
        objects = [self._agent_at_iteration(token, iteration) for token in ordered_tokens if token in tokens]
        return DetectionsTracks(TrackedObjects(objects))

    def _filter(self, detections: DetectionsTracks) -> DetectionsTracks:
        """Filter historical detections without changing their recorded poses."""
        objects = [
            obj
            for obj in detections.tracked_objects
            if obj.tracked_object_type == TrackedObjectType.VEHICLE and obj.track_token in self._track_tokens
        ]
        return DetectionsTracks(TrackedObjects(objects))

    # Proxy methods
    @property
    def token(self) -> str:
        return self._scenario.token

    @property
    def log_name(self) -> str:
        return self._scenario.log_name

    @property
    def scenario_name(self) -> str:
        return f"{self._scenario.scenario_name}_three_vehicle"

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        return self._scenario.ego_vehicle_parameters

    @property
    def scenario_type(self) -> str:
        return f"{self._scenario.scenario_type}_three_vehicle"

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
        del future_trajectory_sampling
        return self._detections_at_iteration(iteration)

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        del past_time_horizon, future_time_horizon, future_trajectory_sampling
        return self._detections_at_iteration(iteration, filter_track_tokens)

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

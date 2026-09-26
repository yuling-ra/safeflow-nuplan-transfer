"""Route-aligned synthetic three-vehicle scenarios for obstacle avoidance."""

from __future__ import annotations

import math
import os
from typing import Generator, List, Optional, Set, Tuple

import numpy as np

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatuses,
    Transform,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, SensorChannel, Sensors
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


class ThreeVehicleSyntheticScenario(AbstractScenario):
    """Replace recorded traffic with two controlled vehicles on the ego route.

    The base scenario still supplies the map, route, traffic lights, and ego
    motion.  Placing obstacles by route progress makes both test cases
    reproducible and guarantees that the nominal route intersects them.
    """

    def __init__(self, base_scenario: AbstractScenario) -> None:
        self._base_scenario = base_scenario
        self._mode = os.getenv("SYNTHETIC_AVOIDANCE_MODE", "combined").strip().lower()
        if self._mode not in {
            "straight_lane_change",
            "intersection_turn",
            "combined",
            "right_turn_pedestrian",
            "roundabout_turn",
        }:
            raise ValueError(f"Unsupported SYNTHETIC_AVOIDANCE_MODE: {self._mode}")
        configured_iterations = int(os.getenv("SYNTHETIC_SCENARIO_MAX_ITERATIONS", "0"))
        if self._mode == "roundabout_turn" and configured_iterations > 0:
            # The completed roundabout route is longer than the source log.
            # Allow this synthetic experiment to continue beyond the logged
            # horizon while keeping all other modes tied to the source scene.
            self._number_of_iterations = configured_iterations
        else:
            self._number_of_iterations = (
                min(base_scenario.get_number_of_iterations(), configured_iterations)
                if configured_iterations > 0
                else base_scenario.get_number_of_iterations()
            )

        route = [
            np.asarray(base_scenario.get_ego_state_at_iteration(index).center.array[:2], dtype=np.float64)
            for index in range(base_scenario.get_number_of_iterations())
        ]
        if len(route) < 2:
            raise RuntimeError("Synthetic avoidance scenario requires at least two ego route samples")
        self._route = np.asarray(route, dtype=np.float64)
        if self._mode == "roundabout_turn":
            self._route = self._extend_roundabout_route(self._route)
        segment_lengths = np.linalg.norm(np.diff(self._route, axis=0), axis=1)
        self._route_progress = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        if self._mode == "straight_lane_change":
            default_progress = (24.0, 26.0)
        elif self._mode == "intersection_turn":
            default_progress = (30.0, 40.0)
        elif self._mode == "right_turn_pedestrian":
            default_progress = (18.0, 28.0)
        elif self._mode == "roundabout_turn":
            default_progress = (7.0, 13.0)
        else:
            default_progress = (24.0, 42.0)
        self._obstacle_progress = (
            self._env_float("SYNTHETIC_OBSTACLE_1_PROGRESS_M", default_progress[0]),
            self._env_float("SYNTHETIC_OBSTACLE_2_PROGRESS_M", default_progress[1]),
        )
        if self._obstacle_progress[0] >= self._obstacle_progress[1]:
            raise ValueError("Synthetic obstacle 1 must be before obstacle 2 on the route")
        self._keep_recorded_vehicles = self._env_bool("SYNTHETIC_KEEP_RECORDED_VEHICLES", False)
        self._excluded_recorded_vehicle_tokens: Set[str] = set()
        if self._keep_recorded_vehicles:
            # Non-reactive replay vehicles that start behind the ego in the
            # same lane can catch a modified ego trajectory and create an
            # artificial rear-end overlap. Keep dense surrounding traffic,
            # but remove those unavoidable replay conflicts.
            initial_ego = base_scenario.get_ego_state_at_iteration(0)
            ego_xy = initial_ego.center.array[:2]
            heading = float(initial_ego.center.heading)
            forward = np.asarray([math.cos(heading), math.sin(heading)])
            left = np.asarray([-math.sin(heading), math.cos(heading)])
            for obj in base_scenario.get_tracked_objects_at_iteration(0).tracked_objects:
                if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                    continue
                relative = np.asarray(obj.center.array[:2]) - ego_xy
                if float(relative @ forward) < -4.0 and abs(float(relative @ left)) < 3.5:
                    self._excluded_recorded_vehicle_tokens.add(str(obj.track_token))
        self._vehicle_speeds = (
            self._env_float(
                "SYNTHETIC_VEHICLE_1_SPEED_MPS",
                1.2 if self._mode == "right_turn_pedestrian"
                else (1.2 if self._mode == "combined" else 0.0),
            ),
            self._env_float(
                "SYNTHETIC_VEHICLE_2_SPEED_MPS",
                1.6 if self._mode == "right_turn_pedestrian"
                else (0.8 if self._mode == "combined" else 0.0),
            ),
        )
        if self._mode == "roundabout_turn":
            self._vehicle_speeds = (
                self._env_float("SYNTHETIC_VEHICLE_1_SPEED_MPS", 0.35),
                self._env_float("SYNTHETIC_VEHICLE_2_SPEED_MPS", 0.25),
            )
        self._roundabout_lateral_offsets = (
            self._env_float("SYNTHETIC_VEHICLE_1_LATERAL_OFFSET_M", 2.8),
            self._env_float("SYNTHETIC_VEHICLE_2_LATERAL_OFFSET_M", -2.8),
        )
        if self._keep_recorded_vehicles:
            # Recorded traffic is replayed open-loop.  A recorded vehicle can
            # therefore occupy the same space as a synthetic obstacle even
            # though the base log itself was collision-free.  Remove any such
            # token for the whole scenario, including near misses, so the
            # dense-traffic scene does not contain an artificial agent-agent
            # collision in NuBoard.
            synthetic_boxes_by_iteration = [
                [vehicle.box.geometry for vehicle in self._vehicles(iteration)]
                for iteration in range(self._number_of_iterations)
            ]
            for iteration, synthetic_boxes in enumerate(synthetic_boxes_by_iteration):
                recorded_vehicles = [
                    recorded
                    for recorded in self._base_scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
                    if recorded.tracked_object_type == TrackedObjectType.VEHICLE
                ]
                for recorded in recorded_vehicles:
                    if recorded.tracked_object_type != TrackedObjectType.VEHICLE:
                        continue
                    if any(recorded.box.geometry.distance(box) <= 0.75 for box in synthetic_boxes):
                        self._excluded_recorded_vehicle_tokens.add(str(recorded.track_token))
                # The recorded actors are also open-loop.  Remove both sides
                # of any recorded vehicle collision so the dense scene does
                # not show unrelated traffic crashing into itself.
                for first_index, first in enumerate(recorded_vehicles):
                    for second in recorded_vehicles[first_index + 1 :]:
                        if first.box.geometry.intersects(second.box.geometry):
                            self._excluded_recorded_vehicle_tokens.update(
                                (str(first.track_token), str(second.track_token))
                            )
        self._pedestrian_speed = self._env_float("SYNTHETIC_PEDESTRIAN_SPEED_MPS", 1.4)
        pedestrian_rng = np.random.default_rng(int(os.getenv("SYNTHETIC_PEDESTRIAN_SEED", "17")))
        phase_min = self._env_float("SYNTHETIC_PEDESTRIAN_PHASE_MIN_S", 0.0)
        phase_max = self._env_float("SYNTHETIC_PEDESTRIAN_PHASE_MAX_S", 2.0)
        if phase_max < phase_min:
            raise ValueError("SYNTHETIC_PEDESTRIAN_PHASE_MAX_S must be >= phase min")
        self._pedestrian_phases = pedestrian_rng.uniform(phase_min, phase_max, size=2).tolist()
        self._pedestrian_speed_scales = pedestrian_rng.uniform(0.85, 1.15, size=2).tolist()
        self._pedestrian_direction_signs = pedestrian_rng.choice([-1.0, 1.0], size=2).tolist()
        # The pedestrian crossing is intentionally before the junction, on
        # the straight approach road rather than inside the right-turn arc.
        self._pedestrian_progress = self._env_float(
            "SYNTHETIC_PEDESTRIAN_PROGRESS_M",
            min(7.0, float(self._route_progress[-1]) * 0.3)
            if self._mode == "right_turn_pedestrian"
            else -1.0,
        )

    def _find_turn_progress(self) -> float:
        """Place the synthetic crosswalk at the strongest route heading change."""
        if len(self._route) < 4:
            return min(20.0, float(self._route_progress[-1]) * 0.5)
        headings = np.arctan2(np.diff(self._route[:, 1]), np.diff(self._route[:, 0]))
        heading_delta = np.asarray(
            [self._wrap_pi(float(headings[i + 1] - headings[i])) for i in range(len(headings) - 1)]
        )
        # Right turns have negative wrapped heading change.  Fall back to the
        # largest turn if the recorded route uses the opposite convention.
        candidates = np.where(heading_delta < -0.15)[0]
        index = int(candidates[np.argmin(heading_delta[candidates])]) if len(candidates) else int(np.argmax(np.abs(heading_delta)))
        return float(self._route_progress[min(index + 1, len(self._route_progress) - 1)])

    def _extend_roundabout_route(self, route: np.ndarray) -> np.ndarray:
        """Continue the recorded route through one complete roundabout loop."""
        # The recorded right-turn route ends at the roundabout entrance. The
        # following lane graph follows the second circle and returns to the
        # same entrance after one full loop.
        lane_sequence = (
            ("67359", SemanticMapLayer.LANE),
            ("70262", SemanticMapLayer.LANE_CONNECTOR),
            ("68554", SemanticMapLayer.LANE),
            ("70465", SemanticMapLayer.LANE_CONNECTOR),
            ("68571", SemanticMapLayer.LANE),
            ("70304", SemanticMapLayer.LANE_CONNECTOR),
            ("67346", SemanticMapLayer.LANE),
            ("64410", SemanticMapLayer.LANE_CONNECTOR),
            ("67573", SemanticMapLayer.LANE),
            ("69378", SemanticMapLayer.LANE_CONNECTOR),
            ("67302", SemanticMapLayer.LANE),
            ("69962", SemanticMapLayer.LANE_CONNECTOR),
        )
        extension = []
        for object_id, layer in lane_sequence:
            map_object = self._base_scenario.map_api.get_map_object(object_id, layer)
            points = np.asarray(
                [[pose.x, pose.y] for pose in map_object.baseline_path.discrete_path],
                dtype=np.float64,
            )
            if len(points) < 2:
                continue
            if extension:
                points = points[1:]
            extension.append(points)
        if not extension:
            return route

        appended = np.vstack(extension)
        # Remove the short overlap between the recorded endpoint and the first
        # map centerline point while preserving the route's first pose.
        if np.linalg.norm(appended[0] - route[-1]) < 2.0:
            appended[0] = route[-1]
        return np.vstack([route, appended])

    def _build_extended_route_states(self) -> List[EgoState]:
        """Create pose-only reference states for the map-completed route."""
        if self._mode != "roundabout_turn":
            return []
        initial = self._base_scenario.get_ego_state_at_iteration(0)
        speed = max(1.0, float(initial.dynamic_car_state.center_velocity_2d.magnitude()))
        states: List[EgoState] = []
        for index, point in enumerate(self._route):
            if index == 0:
                delta = self._route[1] - self._route[0]
            elif index == len(self._route) - 1:
                delta = self._route[-1] - self._route[-2]
            else:
                delta = self._route[index + 1] - self._route[index - 1]
            heading = math.atan2(float(delta[1]), float(delta[0]))
            center = StateSE2(float(point[0]), float(point[1]), heading)
            velocity = StateVector2D(speed * math.cos(heading), speed * math.sin(heading))
            states.append(
                EgoState.build_from_center(
                    center=center,
                    center_velocity_2d=velocity,
                    center_acceleration_2d=StateVector2D(0.0, 0.0),
                    tire_steering_angle=0.0,
                    time_point=TimePoint(initial.time_us + int(index * self.database_interval * 1e6)),
                    vehicle_parameters=initial.car_footprint.vehicle_parameters,
                )
            )
        return states

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name, "").strip()
        return float(value) if value else default

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name, "").strip().lower()
        return default if not value else value not in {"0", "false", "no", "off"}

    @staticmethod
    def _wrap_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _route_pose(self, progress_m: float) -> StateSE2:
        progress = float(np.clip(progress_m, 0.0, self._route_progress[-1]))
        after = int(np.searchsorted(self._route_progress, progress, side="right"))
        after = min(max(after, 1), len(self._route) - 1)
        before = after - 1
        span = float(self._route_progress[after] - self._route_progress[before])
        ratio = 0.0 if span <= 1e-6 else (progress - self._route_progress[before]) / span
        point = self._route[before] + ratio * (self._route[after] - self._route[before])
        delta = self._route[after] - self._route[before]
        heading = math.atan2(float(delta[1]), float(delta[0]))
        return StateSE2(float(point[0]), float(point[1]), heading)

    def _extended_route_pose(self, progress_m: float) -> StateSE2:
        """Follow the recorded turn, then continue along its outgoing road heading."""
        if progress_m <= float(self._route_progress[-1]):
            return self._route_pose(progress_m)
        endpoint = self._route_pose(float(self._route_progress[-1]))
        extension = progress_m - float(self._route_progress[-1])
        return StateSE2(
            float(endpoint.x) + extension * math.cos(float(endpoint.heading)),
            float(endpoint.y) + extension * math.sin(float(endpoint.heading)),
            float(endpoint.heading),
        )

    def _vehicles(self, iteration: int) -> List[Agent]:
        timestamp_us = self.get_time_point(min(max(iteration, 0), self.get_number_of_iterations() - 1)).time_us
        elapsed_s = float(iteration) * float(self.database_interval)
        vehicles: List[Agent] = []
        for index, (progress, speed) in enumerate(zip(self._obstacle_progress, self._vehicle_speeds), start=1):
            if self._mode == "combined" and index == 1:
                start = self._route_pose(progress)
                distance = speed * elapsed_s
                center = StateSE2(
                    float(start.x) + distance * math.cos(float(start.heading)),
                    float(start.y) + distance * math.sin(float(start.heading)),
                    float(start.heading),
                )
            elif self._mode == "combined" and index == 2:
                center = self._route_pose(progress + speed * elapsed_s)
            elif self._mode == "right_turn_pedestrian":
                center = self._extended_route_pose(progress + speed * elapsed_s)
            else:
                center = self._route_pose(progress + speed * elapsed_s)
            if self._mode == "roundabout_turn":
                lateral = self._roundabout_lateral_offsets[index - 1]
                normal = np.asarray(
                    [-math.sin(float(center.heading)), math.cos(float(center.heading))],
                    dtype=np.float64,
                )
                center = StateSE2(
                    float(center.x) + lateral * float(normal[0]),
                    float(center.y) + lateral * float(normal[1]),
                    float(center.heading),
                )
            velocity = StateVector2D(
                speed * math.cos(float(center.heading)),
                speed * math.sin(float(center.heading)),
            )
            token = f"synthetic_{self._mode}_obstacle_{index}"
            vehicles.append(
                Agent(
                    tracked_object_type=TrackedObjectType.VEHICLE,
                    oriented_box=OrientedBox(center, length=4.5, width=2.0, height=1.5),
                    velocity=velocity,
                    metadata=SceneObjectMetadata(
                        timestamp_us=timestamp_us,
                        token=f"{token}_{iteration}",
                        track_id=index,
                        track_token=token,
                        category_name="vehicle",
                    ),
                )
            )
        return vehicles

    def _pedestrians(self, iteration: int) -> List[Agent]:
        if self._mode != "right_turn_pedestrian" or self._pedestrian_progress < 0.0:
            return []
        timestamp_us = self.get_time_point(min(max(iteration, 0), self.get_number_of_iterations() - 1)).time_us
        crosswalk = self._route_pose(self._pedestrian_progress)
        tangent = np.asarray([math.cos(float(crosswalk.heading)), math.sin(float(crosswalk.heading))])
        normal = np.asarray([-tangent[1], tangent[0]])
        center = np.asarray([float(crosswalk.x), float(crosswalk.y)])
        elapsed_s = float(iteration) * float(self.database_interval)
        pedestrians: List[Agent] = []
        # Two people make a single straight crossing and remain on the far
        # sidewalk.  Their direction and start time are randomized but seeded.
        crossing_half_width = self._env_float("SYNTHETIC_PEDESTRIAN_CROSSING_HALF_WIDTH_M", 3.5)
        for index, (phase, speed_scale, direction_sign) in enumerate(
            zip(self._pedestrian_phases, self._pedestrian_speed_scales, self._pedestrian_direction_signs), start=1
        ):
            speed = self._pedestrian_speed * speed_scale
            crossing_duration = 2.0 * crossing_half_width / max(speed, 1e-3)
            elapsed_crossing = max(0.0, elapsed_s - phase)
            finished = elapsed_crossing >= crossing_duration
            distance_in_crossing = min(elapsed_crossing * speed, 2.0 * crossing_half_width)
            distance = (-crossing_half_width + distance_in_crossing) * direction_sign
            direction = direction_sign
            lateral = distance
            # Stagger the two walkers along the crossing so the randomized
            # directions do not place both boxes on top of each other.
            longitudinal_offset = -0.9 if index == 1 else 0.9
            position = center + longitudinal_offset * tangent + lateral * normal
            velocity = np.zeros(2) if finished else direction * speed * normal
            heading = math.atan2(float(velocity[1]), float(velocity[0])) if np.linalg.norm(velocity) > 1e-6 else float(crosswalk.heading)
            pedestrians.append(
                Agent(
                    tracked_object_type=TrackedObjectType.PEDESTRIAN,
                    oriented_box=OrientedBox(
                        StateSE2(float(position[0]), float(position[1]), heading),
                        length=0.6,
                        width=0.6,
                        height=1.7,
                    ),
                    velocity=StateVector2D(float(velocity[0]), float(velocity[1])),
                    metadata=SceneObjectMetadata(
                        timestamp_us=timestamp_us,
                        token=f"synthetic_right_turn_pedestrian_{index}_{iteration}",
                        track_id=100 + index,
                        track_token=f"synthetic_right_turn_pedestrian_{index}",
                        category_name="pedestrian",
                    ),
                )
            )
        return pedestrians

    def _detections(self, iteration: int, filter_track_tokens: Optional[Set[str]] = None) -> DetectionsTracks:
        vehicles = self._vehicles(iteration) + self._pedestrians(iteration)
        if self._keep_recorded_vehicles:
            recorded = self._base_scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
            vehicles = [
                vehicle
                for vehicle in recorded
                if vehicle.tracked_object_type == TrackedObjectType.VEHICLE
                if str(vehicle.track_token) not in self._excluded_recorded_vehicle_tokens
            ] + vehicles
        if filter_track_tokens is not None:
            vehicles = [vehicle for vehicle in vehicles if vehicle.track_token in filter_track_tokens]
        return DetectionsTracks(TrackedObjects(vehicles))

    @property
    def token(self) -> str:
        return self._base_scenario.token

    @property
    def log_name(self) -> str:
        return self._base_scenario.log_name

    @property
    def scenario_name(self) -> str:
        return f"{self._base_scenario.scenario_name}_{self._mode}_3vehicle"

    @property
    def ego_vehicle_parameters(self) -> VehicleParameters:
        return self._base_scenario.ego_vehicle_parameters

    @property
    def scenario_type(self) -> str:
        return f"{self._base_scenario.scenario_type}_{self._mode}_3vehicle"

    @property
    def map_api(self) -> AbstractMap:
        return self._base_scenario.map_api

    @property
    def database_interval(self) -> float:
        return self._base_scenario.database_interval

    def get_number_of_iterations(self) -> int:
        return self._number_of_iterations

    def get_route_reference_states(self, start_iteration: int = 0) -> List[EgoState]:
        """Return the untrimmed recorded ego route used for planner geometry."""
        if self._mode == "roundabout_turn":
            return self._build_extended_route_states()[start_iteration:]
        return [
            self._base_scenario.get_ego_state_at_iteration(index)
            for index in range(start_iteration, self._base_scenario.get_number_of_iterations())
        ]

    def get_time_point(self, iteration: int) -> TimePoint:
        if self._mode == "roundabout_turn" and iteration >= self._base_scenario.get_number_of_iterations():
            initial_time = self._base_scenario.get_time_point(0).time_us
            return TimePoint(initial_time + int(iteration * self.database_interval * 1e6))
        return self._base_scenario.get_time_point(iteration)

    def get_lidar_to_ego_transform(self) -> Transform:
        return self._base_scenario.get_lidar_to_ego_transform()

    def get_mission_goal(self) -> Optional[StateSE2]:
        return self._base_scenario.get_mission_goal()

    def get_route_roadblock_ids(self) -> List[str]:
        return self._base_scenario.get_route_roadblock_ids()

    def get_expert_goal_state(self) -> StateSE2:
        return self._base_scenario.get_expert_goal_state()

    def get_tracked_objects_at_iteration(
        self, iteration: int, future_trajectory_sampling: Optional[TrajectorySampling] = None
    ) -> DetectionsTracks:
        del future_trajectory_sampling
        return self._detections(iteration)

    def get_tracked_objects_within_time_window_at_iteration(
        self,
        iteration: int,
        past_time_horizon: float,
        future_time_horizon: float,
        filter_track_tokens: Optional[Set[str]] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> DetectionsTracks:
        del past_time_horizon, future_time_horizon, future_trajectory_sampling
        return self._detections(iteration, filter_track_tokens)

    def get_sensors_at_iteration(self, iteration: int, channels: Optional[List[SensorChannel]] = None) -> Sensors:
        return self._base_scenario.get_sensors_at_iteration(iteration, channels)

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        if self._mode == "roundabout_turn" and iteration >= self._base_scenario.get_number_of_iterations():
            return self._base_scenario.get_ego_state_at_iteration(self._base_scenario.get_number_of_iterations() - 1)
        return self._base_scenario.get_ego_state_at_iteration(iteration)

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Generator[TrafficLightStatusData, None, None]:
        if self._mode == "roundabout_turn" and iteration >= self._base_scenario.get_number_of_iterations():
            iteration = self._base_scenario.get_number_of_iterations() - 1
        return self._base_scenario.get_traffic_light_status_at_iteration(iteration)

    def get_past_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        return self._base_scenario.get_past_traffic_light_status_history(iteration, time_horizon, num_samples)

    def get_future_traffic_light_status_history(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TrafficLightStatuses, None, None]:
        return self._base_scenario.get_future_traffic_light_status_history(iteration, time_horizon, num_samples)

    def get_future_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        return self._base_scenario.get_future_timestamps(iteration, time_horizon, num_samples)

    def get_past_timestamps(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[TimePoint, None, None]:
        return self._base_scenario.get_past_timestamps(iteration, time_horizon, num_samples)

    def get_ego_future_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        return self._base_scenario.get_ego_future_trajectory(iteration, time_horizon, num_samples)

    def get_ego_past_trajectory(
        self, iteration: int, time_horizon: float, num_samples: Optional[int] = None
    ) -> Generator[EgoState, None, None]:
        return self._base_scenario.get_ego_past_trajectory(iteration, time_horizon, num_samples)

    def get_past_sensors(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        channels: Optional[List[SensorChannel]] = None,
    ) -> Generator[Sensors, None, None]:
        return self._base_scenario.get_past_sensors(iteration, time_horizon, num_samples, channels)

    def get_past_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        base_history = self._base_scenario.get_past_tracked_objects(
            iteration, time_horizon, num_samples, future_trajectory_sampling
        )
        # Preserve the temporal motion of synthetic actors.  Returning the
        # current frame for every history sample makes a moving obstacle look
        # frozen to learned planners and removes an important avoidance cue.
        history = list(base_history)
        if not history:
            return

        # At scenario iteration 0, nuPlan clamps the database query to the
        # first available sample.  Do not turn those repeated initial samples
        # into synthetic future motion: doing so creates a discontinuity when
        # Simulation appends the current observation to the history buffer.
        if iteration <= 0:
            for _ in history:
                yield self._detections(0)
            return

        # The past query includes the current sample.  Align its last item
        # with ``iteration`` so the synthetic actors have the same temporal
        # ordering as the ego history.
        first_iteration = max(0, iteration - len(history) + 1)
        for offset, _ in enumerate(history):
            yield self._detections(min(first_iteration + offset, self._number_of_iterations - 1))

    def get_future_tracked_objects(
        self,
        iteration: int,
        time_horizon: float,
        num_samples: Optional[int] = None,
        future_trajectory_sampling: Optional[TrajectorySampling] = None,
    ) -> Generator[DetectionsTracks, None, None]:
        base_future = self._base_scenario.get_future_tracked_objects(
            iteration, time_horizon, num_samples, future_trajectory_sampling
        )
        for offset, _ in enumerate(base_future, start=1):
            yield self._detections(min(iteration + offset, self.get_number_of_iterations() - 1))

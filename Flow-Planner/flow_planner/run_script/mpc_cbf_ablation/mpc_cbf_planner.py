"""A small, self-contained nonlinear MPC-CBF planner for nuPlan ablations.

The planner uses the nuPlan IDM planner only as a nominal route/speed
reference. It then solves a finite-horizon nonlinear MPC problem with a
kinematic bicycle model and hard circular barrier constraints around predicted
tracked objects. This is intentionally separate from SafeFlow and PDM.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Type

import numpy as np
from scipy.optimize import minimize

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from shapely.geometry import Point

logger = logging.getLogger(__name__)


def _wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


class MPC_CBF_Planner(AbstractPlanner):
    """Nonlinear MPC with hard predicted-obstacle barrier constraints."""

    requires_scenario = False

    def __init__(
        self,
        trajectory_steps: int = 60,
        trajectory_sample_interval: float = 0.1,
        mpc_horizon_steps: int = 12,
        wheel_base_m: float = 2.8,
        max_accel_mps2: float = 3.0,
        max_decel_mps2: float = 6.0,
        max_steering_rad: float = 0.55,
        max_steering_rate_radps: float = 0.6,
        max_speed_mps: float = 15.0,
        max_agents: int = 8,
        obstacle_query_radius_m: float = 45.0,
        obstacle_margin_m: float = 0.5,
        obstacle_prediction_horizon_s: float = 2.0,
        tracking_error_margin_m: float = 0.75,
        tracking_weight: float = 2.0,
        heading_weight: float = 0.5,
        speed_weight: float = 0.5,
        terminal_weight: float = 2.0,
        accel_weight: float = 0.05,
        steering_weight: float = 0.05,
        steering_rate_weight: float = 0.2,
        solver_maxiter: int = 35,
        solver_ftol: float = 1e-3,
        cbf_min_distance_m: float = 2.5,
        cbf_gamma: float = 0.2,
        scenario: Any = None,
    ) -> None:
        if trajectory_steps < mpc_horizon_steps + 1 or mpc_horizon_steps < 2:
            raise ValueError("trajectory_steps must be at least mpc_horizon_steps + 1")
        if trajectory_sample_interval <= 0.0:
            raise ValueError("trajectory_sample_interval must be positive")

        self._steps = int(trajectory_steps)
        self._dt = float(trajectory_sample_interval)
        self._horizon = int(mpc_horizon_steps)
        self._wheel_base = float(wheel_base_m)
        self._rear_axle_to_center_dist = 1.4
        self._max_accel = float(max_accel_mps2)
        self._max_decel = float(max_decel_mps2)
        self._max_steering = float(max_steering_rad)
        self._max_steering_rate = float(max_steering_rate_radps)
        self._max_speed = float(max_speed_mps)
        self._max_agents = int(max_agents)
        self._query_radius = float(obstacle_query_radius_m)
        self._obstacle_margin = float(obstacle_margin_m)
        self._tracking_error_margin = max(0.0, float(tracking_error_margin_m))
        self._prediction_horizon = max(0.0, float(obstacle_prediction_horizon_s))
        self._weights = np.asarray(
            [tracking_weight, heading_weight, speed_weight, terminal_weight,
             accel_weight, steering_weight, steering_rate_weight],
            dtype=np.float64,
        )
        self._solver_maxiter = int(solver_maxiter)
        self._solver_ftol = float(solver_ftol)
        self._cbf_min_distance = float(cbf_min_distance_m)
        if not 0.0 < cbf_gamma <= 1.0:
            raise ValueError("cbf_gamma must be in (0, 1]")
        self._cbf_gamma = float(cbf_gamma)
        self._reference_planner = IDMPlanner(
            target_velocity=8.0,
            min_gap_to_lead_agent=1.0,
            headway_time=1.5,
            accel_max=1.5,
            decel_max=3.0,
            planned_trajectory_samples=self._steps,
            planned_trajectory_sample_interval=self._dt,
            occupancy_map_radius=self._query_radius,
        )
        self._iteration = 0
        self._last_solution: np.ndarray | None = None
        self._solver_failures = 0
        self._solver_successes = 0
        self._last_solver_warning_iteration = -1
        self._map_api = None

    def name(self) -> str:
        return "mpc_cbf"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._reference_planner.initialize(initialization)
        self._map_api = initialization.map_api
        self._iteration = 0
        self._last_solution = None
        self._solver_failures = 0
        self._solver_successes = 0
        self._last_solver_warning_iteration = -1

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_state, observation = current_input.history.current_state
        reference = self._reference_planner.compute_planner_trajectory(current_input)
        reference_states = reference.get_sampled_trajectory()
        reference_xy, reference_heading, reference_speed = self._reference_arrays(
            ego_state, reference_states
        )
        obstacles = self._extract_obstacles(ego_state, observation)
        self._rear_axle_to_center_dist = float(ego_state.car_footprint.rear_axle_to_center_dist)
        self._wheel_base = float(ego_state.car_footprint.vehicle_parameters.wheel_base)
        if self._iteration == 0:
            ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
            logger.info(
                "MPC-CBF detected %d obstacles at initial state: %s",
                len(obstacles),
                [
                    {
                        "distance_m": round(float(np.linalg.norm(item["center"] - ego_xy)), 2),
                        "radius_m": round(float(item["radius"]), 2),
                        "type": item["type"].name,
                    }
                    for item in obstacles[:8]
                ],
            )
        x0 = np.asarray(
            [
                float(ego_state.rear_axle.x),
                float(ego_state.rear_axle.y),
                float(ego_state.rear_axle.heading),
                float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
            ],
            dtype=np.float64,
        )

        initial_guess = self._initial_guess(x0, reference_xy, reference_heading, reference_speed)
        solution = self._solve(x0, reference_xy, reference_heading, reference_speed, obstacles, initial_guess)
        if solution is None:
            self._solver_failures += 1
            if self._last_solver_warning_iteration != self._iteration:
                logger.warning(
                    "MPC-CBF had no feasible SLSQP solution at replanning step %d; "
                    "using the safest checked fallback.",
                    self._iteration,
                )
                self._last_solver_warning_iteration = self._iteration
            solution = self._fallback_solution(x0, reference_xy, reference_heading, reference_speed, obstacles)
        else:
            self._solver_successes += 1
            self._last_solution = solution[1].copy()

        states = self._join_reference_tail(solution[0], reference_xy, reference_heading, reference_speed)
        if self._iteration == 0:
            logger.info(
                "MPC-CBF initial trajectory: ego=(%.2f, %.2f, %.2f, %.2f), "
                "idm_start=(%.2f, %.2f, %.2f), idm_end=(%.2f, %.2f, %.2f), "
                "mpc_end=(%.2f, %.2f, %.2f, %.2f)",
                x0[0], x0[1], x0[2], x0[3],
                reference_xy[0, 0], reference_xy[0, 1], reference_heading[0],
                reference_xy[min(self._horizon, len(reference_xy) - 1), 0],
                reference_xy[min(self._horizon, len(reference_xy) - 1), 1],
                reference_speed[min(self._horizon, len(reference_speed) - 1)],
                states[-1, 0], states[-1, 1], states[-1, 2], states[-1, 3],
            )
        self._iteration += 1
        return self._build_trajectory(ego_state, states)

    def _reference_arrays(self, ego_state: EgoState, reference_states: List[EgoState]):
        xy = np.asarray([[s.rear_axle.x, s.rear_axle.y] for s in reference_states], dtype=np.float64)
        heading = np.unwrap(np.asarray([s.rear_axle.heading for s in reference_states], dtype=np.float64))
        speed = np.asarray(
            [s.dynamic_car_state.rear_axle_velocity_2d.magnitude() for s in reference_states],
            dtype=np.float64,
        )
        reference_is_misaligned = len(xy) == 0
        reference_distance = float("nan")
        reference_heading_error = float("nan")
        map_reference_used = False
        if len(xy) > 0:
            reference_distance = float(np.linalg.norm(xy[0] - np.asarray(ego_state.rear_axle.array[:2])))
            reference_heading_error = abs(_wrap_pi(float(heading[0]) - float(ego_state.rear_axle.heading)))
            reference_is_misaligned = (
                reference_distance > 5.0
                or reference_heading_error > 0.7
            )
            if reference_is_misaligned:
                map_reference = self._build_map_reference_arrays(ego_state)
                if map_reference is not None:
                    xy, heading, speed = map_reference
                    reference_is_misaligned = False
                    map_reference_used = True
                    if self._iteration == 0:
                        logger.warning(
                            "IDM route reference was misaligned by %.2fm; using the nearest "
                            "heading-consistent map lane geometry.",
                            reference_distance,
                        )
            if reference_distance > 5.0 and reference_heading_error <= 0.7 and not map_reference_used:
                # The IDM route can begin on a nearby but displaced segment
                # after a route-graph fallback. Preserve its lane curvature by
                # translating the route to the current rear axle.
                xy = xy + (
                    np.asarray(ego_state.rear_axle.array[:2], dtype=np.float64) - xy[0]
                )[None, :]
                reference_is_misaligned = False
                if self._iteration == 0:
                    logger.warning(
                        "IDM route reference was displaced by %.2fm; translating it to ego while "
                        "preserving route curvature.",
                        reference_distance,
                    )
        if reference_is_misaligned:
            # IDM can fall back to the longest route when a scenario's route
            # roadblocks are inconsistent. Do not optimize toward a reference
            # that starts tens of metres away from the current vehicle.
            start = np.asarray(ego_state.rear_axle.array[:2], dtype=np.float64)
            start_heading = float(ego_state.rear_axle.heading)
            start_speed = max(0.0, float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
            times = np.arange(self._steps, dtype=np.float64) * self._dt
            xy = start[None, :] + times[:, None] * start_speed * np.asarray(
                [math.cos(start_heading), math.sin(start_heading)], dtype=np.float64
            )[None, :]
            heading = np.full(self._steps, start_heading, dtype=np.float64)
            speed = np.full(self._steps, start_speed, dtype=np.float64)
            if self._iteration == 0:
                logger.warning(
                    "IDM route reference was misaligned with ego (first-point distance %.2fm); "
                    "using a local constant-heading reference.",
                    reference_distance,
                )
        return self._pad(xy, self._steps), self._pad(heading, self._steps), self._pad(speed, self._steps)

    def _build_map_reference_arrays(
        self, ego_state: EgoState
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Build a short road-aligned reference when IDM returns a bad route."""
        if self._map_api is None:
            return None
        ego_xy = np.asarray(ego_state.rear_axle.array[:2], dtype=np.float64)
        ego_heading = float(ego_state.rear_axle.heading)
        ego_speed = max(0.5, float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x))
        point = Point(float(ego_xy[0]), float(ego_xy[1]))
        objects = []
        for layer in (SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.LANE):
            try:
                objects.extend(self._map_api.get_proximal_map_objects(
                    Point2D(float(ego_xy[0]), float(ego_xy[1])), 15.0, [layer]
                )[layer])
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        if not objects:
            return None

        def object_score(obj: Any) -> tuple[float, float]:
            path = getattr(obj, "baseline_path", None)
            samples = list(getattr(path, "discrete_path", [])) if path is not None else []
            if len(samples) < 2:
                return float("inf"), float("inf")
            nearest = min(samples, key=lambda state: (state.x - ego_xy[0]) ** 2 + (state.y - ego_xy[1]) ** 2)
            heading_error = abs(_wrap_pi(float(nearest.heading) - ego_heading))
            return float(obj.polygon.distance(point) + 4.0 * heading_error), heading_error

        candidates = [(object_score(obj), obj) for obj in objects]
        candidates = [item for item in candidates if item[0][1] <= 0.7]
        if not candidates:
            return None
        current = min(candidates, key=lambda item: item[0])[1]

        path_states: List[Any] = []
        for _ in range(6):
            path = getattr(current, "baseline_path", None)
            samples = list(getattr(path, "discrete_path", [])) if path is not None else []
            if len(samples) < 2:
                break
            for state in samples:
                if not path_states or math.hypot(state.x - path_states[-1].x, state.y - path_states[-1].y) > 0.05:
                    path_states.append(state)
            outgoing = list(getattr(current, "outgoing_edges", []))
            if not outgoing:
                break
            last_heading = float(path_states[-1].heading)
            aligned = [
                edge for edge in outgoing
                if getattr(edge, "baseline_path", None) is not None
                and len(getattr(edge.baseline_path, "discrete_path", [])) >= 2
            ]
            if not aligned:
                break
            current = min(
                aligned,
                key=lambda edge: abs(
                    _wrap_pi(float(edge.baseline_path.discrete_path[0].heading) - last_heading)
                ),
            )

        if len(path_states) < 3:
            return None
        distances = np.zeros(len(path_states), dtype=np.float64)
        for index in range(1, len(path_states)):
            distances[index] = distances[index - 1] + math.hypot(
                path_states[index].x - path_states[index - 1].x,
                path_states[index].y - path_states[index - 1].y,
            )
        nearest_index = int(np.argmin((np.asarray([state.x for state in path_states]) - ego_xy[0]) ** 2
                                      + (np.asarray([state.y for state in path_states]) - ego_xy[1]) ** 2))
        start_distance = float(distances[nearest_index])
        target_distances = start_distance + ego_speed * np.arange(self._steps, dtype=np.float64) * self._dt
        x = np.interp(target_distances, distances, np.asarray([state.x for state in path_states]))
        y = np.interp(target_distances, distances, np.asarray([state.y for state in path_states]))
        raw_heading = np.unwrap(np.asarray([state.heading for state in path_states], dtype=np.float64))
        heading = np.interp(target_distances, distances, raw_heading)
        return np.column_stack([x, y]), heading, np.full(self._steps, ego_speed, dtype=np.float64)

    @staticmethod
    def _pad(values: np.ndarray, size: int) -> np.ndarray:
        if len(values) >= size:
            return values[:size].copy()
        return np.concatenate([values, np.repeat(values[-1:], size - len(values), axis=0)])

    def _extract_obstacles(self, ego_state: EgoState, observation: Observation) -> List[Dict[str, Any]]:
        if not isinstance(observation, DetectionsTracks):
            return []
        ego_xy = np.asarray(ego_state.center.array[:2], dtype=np.float64)
        # Use the larger half-dimension for the circular footprint.  Using the
        # box diagonal is a valid outer bound, but is excessively conservative
        # for lane-following scenes and often makes the nonlinear program
        # infeasible before the vehicle has any chance to react.
        ego_radius = 0.5 * max(
            float(ego_state.car_footprint.oriented_box.length),
            float(ego_state.car_footprint.oriented_box.width),
        )
        result: List[Dict[str, Any]] = []
        for tracked in observation.tracked_objects.tracked_objects:
            if tracked.tracked_object_type not in {
                TrackedObjectType.VEHICLE,
                TrackedObjectType.PEDESTRIAN,
                TrackedObjectType.BICYCLE,
            }:
                continue
            center = np.asarray(tracked.center.array[:2], dtype=np.float64)
            distance = float(np.linalg.norm(center - ego_xy))
            if distance > self._query_radius:
                continue
            half_length = 0.5 * float(tracked.box.length)
            half_width = 0.5 * float(tracked.box.width)
            longitudinal_radius = (
                0.5 * float(ego_state.car_footprint.oriented_box.length)
                + half_length
                + self._obstacle_margin
                + self._tracking_error_margin
            )
            lateral_radius = (
                0.5 * float(ego_state.car_footprint.oriented_box.width)
                + half_width
                + self._obstacle_margin
                + self._tracking_error_margin
            )
            radius = (
                0.5 * max(float(tracked.box.length), float(tracked.box.width))
                + ego_radius
                + self._obstacle_margin
                + self._tracking_error_margin
            )
            velocity = np.asarray([float(tracked.velocity.x), float(tracked.velocity.y)], dtype=np.float64)
            result.append({
                "center": center,
                "velocity": velocity,
                "radius": radius,
                "longitudinal_radius": longitudinal_radius,
                "lateral_radius": lateral_radius,
                "heading": float(tracked.center.heading),
                "type": tracked.tracked_object_type,
            })
        result.sort(key=lambda item: float(np.linalg.norm(item["center"] - ego_xy)))
        return result[: self._max_agents]

    def _rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((self._horizon + 1, 4), dtype=np.float64)
        states[0] = x0
        for k in range(self._horizon):
            x, y, heading, speed = states[k]
            accel, steering = controls[k]
            speed_next = np.clip(speed + self._dt * accel, 0.0, self._max_speed)
            states[k + 1, 0] = x + self._dt * speed * math.cos(heading)
            states[k + 1, 1] = y + self._dt * speed * math.sin(heading)
            states[k + 1, 2] = heading + self._dt * speed * math.tan(steering) / self._wheel_base
            states[k + 1, 3] = speed_next
        return states

    def _initial_guess(self, x0, ref_xy, ref_heading, ref_speed) -> np.ndarray:
        count = self._horizon
        accel = np.diff(np.r_[x0[3], ref_speed[:count]]) / self._dt
        yaw_rate = np.diff(np.r_[x0[2], ref_heading[:count]]) / self._dt
        steering = np.arctan2(self._wheel_base * yaw_rate, np.maximum(np.r_[x0[3], ref_speed[:count - 1]], 0.5))
        return np.column_stack([
            np.clip(accel[:count], -self._max_decel, self._max_accel),
            np.clip(steering[:count], -self._max_steering, self._max_steering),
        ])

    def _solve(self, x0, ref_xy, ref_heading, ref_speed, obstacles, initial_guess):
        def unpack(vector: np.ndarray) -> np.ndarray:
            return vector.reshape(self._horizon, 2)

        def objective(vector: np.ndarray) -> float:
            controls = unpack(vector)
            states = self._rollout(x0, controls)
            ref_xy_h = ref_xy[: self._horizon + 1]
            ref_heading_h = ref_heading[: self._horizon + 1]
            ref_speed_h = ref_speed[: self._horizon + 1]
            position_error = np.sum((states[:, :2] - ref_xy_h) ** 2, axis=1)
            heading_error = np.asarray([_wrap_pi(v) for v in states[:, 2] - ref_heading_h]) ** 2
            speed_error = (states[:, 3] - ref_speed_h) ** 2
            control_cost = np.sum(controls[:, 0] ** 2) * self._weights[4]
            steering_cost = np.sum(controls[:, 1] ** 2) * self._weights[5]
            rate_cost = np.sum(np.diff(controls[:, 1], prepend=controls[0, 1]) ** 2) * self._weights[6]
            return float(
                self._weights[0] * np.mean(position_error)
                + self._weights[1] * np.mean(heading_error)
                + self._weights[2] * np.mean(speed_error)
                + self._weights[3] * (position_error[-1] + speed_error[-1])
                + control_cost + steering_cost + rate_cost
            )

        def safety_constraints(vector: np.ndarray) -> np.ndarray:
            states = self._rollout(x0, unpack(vector))
            return self._discrete_cbf_values(states, obstacles)

        def actuator_constraints(vector: np.ndarray) -> np.ndarray:
            controls = unpack(vector)
            steering_rate = np.diff(controls[:, 1], prepend=controls[0, 1]) / self._dt
            return self._max_steering_rate - np.abs(steering_rate)

        bounds = [(-self._max_decel, self._max_accel), (-self._max_steering, self._max_steering)] * self._horizon
        constraints = [
            {"type": "ineq", "fun": safety_constraints},
            {"type": "ineq", "fun": actuator_constraints},
        ]
        guesses = [initial_guess]
        if self._last_solution is not None and self._last_solution.shape == initial_guess.shape:
            # Receding-horizon warm start: preserve the previously safe action
            # sequence and repeat its last control at the new horizon.
            guesses.insert(0, np.vstack([self._last_solution[1:], self._last_solution[-1:]]))

        # A braking and two mild lateral guesses help SLSQP leave an infeasible
        # nominal IDM trajectory when a lead vehicle occupies the reference lane.
        braking = np.zeros_like(initial_guess)
        braking[:, 0] = -self._max_decel
        guesses.extend(
            [
                braking,
                np.column_stack([braking[:, 0], np.full(self._horizon, 0.20)]),
                np.column_stack([braking[:, 0], np.full(self._horizon, -0.20)]),
            ]
        )

        best = None
        for guess in guesses:
            try:
                result = minimize(
                    objective,
                    np.asarray(guess, dtype=np.float64).reshape(-1),
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self._solver_maxiter, "ftol": self._solver_ftol, "disp": False},
                )
            except (FloatingPointError, ValueError, RuntimeError) as error:
                logger.debug("MPC-CBF solver exception: %s", error)
                continue
            if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
                continue
            margin = float(np.min(safety_constraints(result.x)))
            if margin < -1e-3:
                continue
            candidate = (float(result.fun), result.x.copy())
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            return None
        controls = unpack(best[1])
        return self._rollout(x0, controls), controls

    def _discrete_cbf_values(self, states: np.ndarray, obstacles: List[Dict[str, Any]]) -> np.ndarray:
        """Evaluate h[k+1] - (1-gamma)h[k] for moving obstacles."""
        values = []
        for obstacle in obstacles:
            radius = max(float(obstacle["radius"]), self._cbf_min_distance)
            barriers = []
            for k in range(self._horizon + 1):
                center = obstacle["center"] + obstacle["velocity"] * min(
                    k * self._dt, self._prediction_horizon
                )
                ego_center = states[k, :2] + self._rear_axle_to_center_dist * np.asarray(
                    [math.cos(states[k, 2]), math.sin(states[k, 2])], dtype=np.float64
                )
                delta = ego_center - center
                if obstacle["type"] == TrackedObjectType.VEHICLE:
                    heading = float(obstacle["heading"])
                    longitudinal = delta[0] * math.cos(heading) + delta[1] * math.sin(heading)
                    lateral = -delta[0] * math.sin(heading) + delta[1] * math.cos(heading)
                    long_radius = max(float(obstacle["longitudinal_radius"]), self._cbf_min_distance)
                    lat_radius = max(float(obstacle["lateral_radius"]), self._cbf_min_distance)
                    barriers.append(
                        float((longitudinal / long_radius) ** 2 + (lateral / lat_radius) ** 2 - 1.0)
                    )
                else:
                    barriers.append(float(np.sum(delta ** 2) - radius * radius))
            # Enforce both forward invariance and membership in the safe set.
            # The latter prevents a solver from accepting a trajectory that is
            # merely moving toward an already violated barrier.
            values.extend(barriers[1:])
            values.extend(
                barriers[k + 1] - (1.0 - self._cbf_gamma) * barriers[k]
                for k in range(self._horizon)
            )
        return np.asarray(values, dtype=np.float64) if values else np.ones(1, dtype=np.float64)

    def _fallback_solution(self, x0, ref_xy, ref_heading, ref_speed, obstacles):
        controls = np.zeros((self._horizon, 2), dtype=np.float64)
        controls[:, 0] = -self._max_decel
        candidates = [controls]
        if self._last_solution is not None and self._last_solution.shape == controls.shape:
            candidates.insert(0, np.vstack([self._last_solution[1:], self._last_solution[-1:]]))
        for steering in (0.20, -0.20, self._max_steering, -self._max_steering):
            candidate = controls.copy()
            candidate[:, 1] = steering
            candidates.append(candidate)

        def margin(candidate: np.ndarray) -> float:
            states = self._rollout(x0, candidate)
            return float(np.min(self._discrete_cbf_values(states, obstacles)))

        checked = [(margin(candidate), candidate) for candidate in candidates]
        best_margin, best_controls = max(checked, key=lambda item: item[0])
        if best_margin < -1e-3:
            logger.error(
                "MPC-CBF fallback is infeasible (minimum barrier margin %.3f); "
                "the scenario is outside this planner's actuation/sensing limits.",
                best_margin,
            )
        return self._rollout(x0, best_controls), best_controls

    def _join_reference_tail(self, solved_states, ref_xy, ref_heading, ref_speed) -> np.ndarray:
        # Do not append an unconstrained nominal tail. The returned trajectory
        # is deliberately limited to the MPC prediction horizon, so every
        # emitted point has been checked against the hard barrier constraints.
        return np.asarray(solved_states[: self._horizon + 1], dtype=np.float64)

    def _build_trajectory(self, ego_state: EgoState, states: np.ndarray) -> InterpolatedTrajectory:
        result: List[EgoState] = [ego_state]
        vehicle = ego_state.car_footprint.vehicle_parameters
        previous_speed = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
        previous_heading = float(ego_state.rear_axle.heading)
        for index in range(1, len(states)):
            speed = max(0.0, float(states[index, 3]))
            acceleration = (speed - previous_speed) / self._dt
            heading = _wrap_pi(float(states[index, 2]))
            heading_delta = _wrap_pi(heading - previous_heading)
            yaw_rate = heading_delta / self._dt
            steering = math.atan2(self._wheel_base * yaw_rate, max(speed, 0.5))
            result.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(float(states[index, 0]), float(states[index, 1]), heading),
                    rear_axle_velocity_2d=StateVector2D(speed, 0.0),
                    rear_axle_acceleration_2d=StateVector2D(acceleration, 0.0),
                    tire_steering_angle=float(np.clip(steering, -self._max_steering, self._max_steering)),
                    time_point=TimePoint(ego_state.time_us + int(round(index * self._dt * 1e6))),
                    vehicle_parameters=vehicle,
                    angular_vel=yaw_rate,
                )
            )
            previous_speed = speed
            previous_heading = heading
        return InterpolatedTrajectory(result)

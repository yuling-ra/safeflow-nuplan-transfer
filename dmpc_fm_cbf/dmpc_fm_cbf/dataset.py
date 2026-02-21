"""
Dataset Generation for Flow Matching Training.

Provides:
- Bicycle dynamics simulation
- Optimal control trajectory generation
- Dataset creation utilities
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class BicycleParams:
    """Bicycle model parameters."""
    L: float = 0.8           # Wheelbase
    dt: float = 0.1          # Timestep
    delta_max: float = 0.524  # Max steering angle (30 deg)
    a_max: float = 0.8       # Max acceleration
    delta_rate_max: float = 2.09  # Max steering rate (120 deg/s)
    v_min: float = 0.1       # Minimum velocity
    v_max: float = 2.0       # Maximum velocity


class BicycleDynamics:
    """
    Bicycle kinematic model.
    
    State: [x, y, theta, v, delta]
        - (x, y): Position
        - theta: Heading angle
        - v: Forward velocity
        - delta: Steering angle
    
    Control: [a, delta_rate]
        - a: Acceleration
        - delta_rate: Steering rate
    """
    
    def __init__(self, params: Optional[BicycleParams] = None):
        self.params = params or BicycleParams()
    
    def step(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Forward dynamics step.
        
        Args:
            state: Current state [x, y, theta, v, delta]
            control: Control input [a, delta_rate]
            dt: Timestep (uses default if None)
        
        Returns:
            Next state [x, y, theta, v, delta]
        """
        p = self.params
        dt = dt or p.dt
        
        x, y, theta, v, delta = state
        a, delta_rate = control
        
        # Clip controls
        a = np.clip(a, -p.a_max, p.a_max)
        delta_rate = np.clip(delta_rate, -p.delta_rate_max, p.delta_rate_max)
        
        # Update steering
        delta_new = delta + delta_rate * dt
        delta_new = np.clip(delta_new, -p.delta_max, p.delta_max)
        
        # Update velocity
        v_new = v + a * dt
        v_new = np.clip(v_new, p.v_min, p.v_max)
        
        # Update position and heading
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / p.L) * np.tan(delta) * dt
        
        return np.array([x_new, y_new, theta_new, v_new, delta_new])
    
    def rollout(
        self,
        state0: np.ndarray,
        controls: np.ndarray,
        dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Rollout trajectory from controls.
        
        Args:
            state0: Initial state [5]
            controls: Control sequence [T, 2]
            dt: Timestep
        
        Returns:
            State trajectory [T+1, 5]
        """
        T = len(controls)
        states = [state0.copy()]
        state = state0.copy()
        
        for t in range(T):
            state = self.step(state, controls[t], dt)
            states.append(state)
        
        return np.array(states)


def generate_reference_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    T: int,
    radius: float = 3.0,
    mode: str = "arc"
) -> np.ndarray:
    """
    Generate reference trajectory from start to goal.
    
    Args:
        start: Start position [2]
        goal: Goal position [2]
        T: Number of timesteps
        radius: Trajectory radius parameter
        mode: 'arc' or 'straight'
    
    Returns:
        Reference positions [T, 2]
    """
    start = np.asarray(start).reshape(2)
    goal = np.asarray(goal).reshape(2)
    
    if mode == "straight":
        t = np.linspace(0, 1, T)
        ref = start[None, :] + t[:, None] * (goal - start)[None, :]
        return ref
    
    # Arc trajectory
    d = np.linalg.norm(goal - start)
    mid = (start + goal) / 2
    
    # Direction perpendicular to start-goal line
    direction = goal - start
    perp = np.array([-direction[1], direction[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    
    # Arc center offset
    arc_height = min(radius, d / 2)
    center = mid + arc_height * perp
    
    # Generate arc
    angles = np.linspace(0, np.pi, T)
    ref = np.zeros((T, 2))
    
    r1 = np.linalg.norm(start - center)
    r2 = np.linalg.norm(goal - center)
    
    for i, a in enumerate(angles):
        blend = i / (T - 1)
        r = (1 - blend) * r1 + blend * r2
        
        # Interpolate angle
        a1 = np.arctan2(start[1] - center[1], start[0] - center[0])
        a2 = np.arctan2(goal[1] - center[1], goal[0] - center[0])
        
        angle = a1 + blend * (a2 - a1)
        ref[i] = center + r * np.array([np.cos(angle), np.sin(angle)])
    
    return ref


def tracking_cost(
    state: np.ndarray,
    ref: np.ndarray,
    control: np.ndarray,
    goal: np.ndarray,
    weights: Dict[str, float]
) -> float:
    """
    Compute tracking cost for optimal control.
    
    Args:
        state: Current state [5]
        ref: Reference position [2]
        control: Control input [2]
        goal: Goal position [2]
        weights: Cost weights dict
    
    Returns:
        Cost value
    """
    pos = state[:2]
    
    # Position tracking
    pos_err = np.sum((pos - ref) ** 2)
    
    # Goal reaching
    goal_err = np.sum((pos - goal) ** 2)
    
    # Control effort
    ctrl_err = np.sum(control ** 2)
    
    cost = (
        weights.get("pos", 1.0) * pos_err +
        weights.get("goal", 0.1) * goal_err +
        weights.get("ctrl", 0.01) * ctrl_err
    )
    
    return cost


def optimize_trajectory_ilqr(
    dynamics: BicycleDynamics,
    state0: np.ndarray,
    goal: np.ndarray,
    ref_traj: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Optimize trajectory using iterative LQR.
    
    Simplified implementation - uses gradient descent on controls.
    
    Args:
        dynamics: Bicycle dynamics
        state0: Initial state [5]
        goal: Goal position [2]
        ref_traj: Reference trajectory [T, 2]
        max_iters: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        states: Optimized states [T+1, 5]
        controls: Optimized controls [T, 2]
        success: Whether optimization converged
    """
    T = len(ref_traj)
    p = dynamics.params
    
    # Initialize controls (simple heuristic)
    controls = np.zeros((T, 2))
    
    # Direction to goal
    dir_to_goal = goal - state0[:2]
    target_heading = np.arctan2(dir_to_goal[1], dir_to_goal[0])
    
    for i in range(T):
        # Simple acceleration to desired speed
        blend = i / T
        desired_v = 0.3 + 0.5 * (1 - abs(2 * blend - 1))
        controls[i, 0] = np.clip(desired_v - 0.3, -p.a_max, p.a_max) * 0.1
    
    # Gradient descent optimization
    lr = 0.01
    best_cost = float('inf')
    best_controls = controls.copy()
    
    weights = {"pos": 10.0, "goal": 50.0, "ctrl": 0.1}
    
    for iteration in range(max_iters):
        # Rollout
        states = dynamics.rollout(state0, controls)
        
        # Compute cost
        total_cost = 0
        for t in range(T):
            total_cost += tracking_cost(
                states[t], ref_traj[t], controls[t], goal, weights
            )
        total_cost += 100.0 * np.sum((states[-1, :2] - goal) ** 2)
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_controls = controls.copy()
        
        # Finite difference gradient
        eps = 1e-3
        grad = np.zeros_like(controls)
        
        for t in range(T):
            for d in range(2):
                controls[t, d] += eps
                states_plus = dynamics.rollout(state0, controls)
                cost_plus = sum(
                    tracking_cost(states_plus[i], ref_traj[i], controls[i], goal, weights)
                    for i in range(T)
                ) + 100.0 * np.sum((states_plus[-1, :2] - goal) ** 2)
                
                controls[t, d] -= 2 * eps
                states_minus = dynamics.rollout(state0, controls)
                cost_minus = sum(
                    tracking_cost(states_minus[i], ref_traj[i], controls[i], goal, weights)
                    for i in range(T)
                ) + 100.0 * np.sum((states_minus[-1, :2] - goal) ** 2)
                
                controls[t, d] += eps
                grad[t, d] = (cost_plus - cost_minus) / (2 * eps)
        
        # Update
        controls = controls - lr * grad
        
        # Clip controls
        controls[:, 0] = np.clip(controls[:, 0], -p.a_max, p.a_max)
        controls[:, 1] = np.clip(controls[:, 1], -p.delta_rate_max, p.delta_rate_max)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    # Final rollout with best controls
    states = dynamics.rollout(state0, best_controls)
    
    # Check if reached goal
    final_dist = np.linalg.norm(states[-1, :2] - goal)
    success = final_dist < 0.5
    
    return states, best_controls, success


def make_ring_dataset(
    N: int = 100,
    T: int = 50,
    radius: float = 3.0,
    v0_range: Tuple[float, float] = (0.15, 0.6),
    save_path: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Generate dataset of trajectories from center to ring boundary.
    
    Args:
        N: Number of trajectories
        T: Trajectory length
        radius: Ring radius
        v0_range: Initial velocity range
        save_path: Path to save dataset (optional)
        seed: Random seed
    
    Returns:
        Dataset dict with keys:
            - xy: Position trajectories [N, T, 2]
            - controls: Control sequences [N, T, 2]
            - goals: Goal positions [N, 2]
            - params: Generation parameters
    """
    np.random.seed(seed)
    
    dynamics = BicycleDynamics()
    
    xy_all = []
    controls_all = []
    goals_all = []
    ref_all = []
    
    for i in range(N):
        # Random goal on ring boundary
        angle = np.random.uniform(0, 2 * np.pi)
        goal = radius * np.array([np.cos(angle), np.sin(angle)])
        
        # Random initial velocity
        v0 = np.random.uniform(*v0_range)
        
        # Initial state at center
        theta0 = angle + np.random.uniform(-0.3, 0.3)
        state0 = np.array([0.0, 0.0, theta0, v0, 0.0])
        
        # Generate reference trajectory
        ref_traj = generate_reference_trajectory(
            state0[:2], goal, T, radius=radius * 0.5, mode="arc"
        )
        
        # Optimize trajectory
        states, controls, success = optimize_trajectory_ilqr(
            dynamics, state0, goal, ref_traj, max_iters=30
        )
        
        if success or i < N // 2:  # Keep some even if not perfect
            xy_all.append(states[:T, :2])
            controls_all.append(controls[:T])
            goals_all.append(goal)
            ref_all.append(ref_traj)
        
        if (i + 1) % (N // 10) == 0:
            print(f"{100 * (i + 1) // N}% ({i + 1}/{N}) done")
    
    dataset = {
        "xy": np.array(xy_all),
        "controls": np.array(controls_all),
        "goals": np.array(goals_all),
        "refs": np.array(ref_all),
        "params": {
            "N": len(xy_all),
            "T": T,
            "radius": radius,
            "dt": dynamics.params.dt,
            "L": dynamics.params.L,
            "delta_max": dynamics.params.delta_max,
            "a_max": dynamics.params.a_max,
        }
    }
    
    if save_path:
        np.save(save_path, dataset)
        print(f"Saved: {save_path}")
    
    return dataset


def make_velocity_dataset(
    xy_traj: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Convert position trajectories to velocity sequences.
    
    Args:
        xy_traj: Position trajectories [N, T, 2]
        dt: Timestep
    
    Returns:
        Velocity sequences [N, T-1, 2]
    """
    # Finite difference
    v = np.diff(xy_traj, axis=1) / dt
    return v


def canonicalize_dataset(
    xy_traj: np.ndarray,
    goals: np.ndarray,
    obstacles: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Transform dataset to canonical frames.
    
    Args:
        xy_traj: Position trajectories [N, T, 2]
        goals: Goal positions [N, 2]
        obstacles: Obstacle positions [N, 2] (optional)
    
    Returns:
        xy_can: Canonicalized trajectories [N, T, 2]
        cond: Conditioning vectors [N, cond_dim]
        metas: List of transformation metadata
    """
    from .canonical import canonicalize_trajectory
    
    N = len(xy_traj)
    xy_can_list = []
    cond_list = []
    metas = []
    
    for i in range(N):
        obs = obstacles[i] if obstacles is not None else None
        xy_can, meta = canonicalize_trajectory(
            xy_traj[i], xy_traj[i, 0], goals[i], obs
        )
        xy_can_list.append(xy_can)
        
        # Build condition vector
        cond = [meta.dist_to_goal]
        if obs is not None:
            obs_rel = obs - xy_traj[i, 0]
            cond.extend(obs_rel.tolist())
        cond_list.append(cond)
        metas.append(meta)
    
    return np.array(xy_can_list), np.array(cond_list, dtype=np.float32), metas

"""
Canonical Frame Transformations for SE(2) Equivariance.

Current project-wide convention (aligned with 5D FM training/inference):
- Origin at ego/start position
- Goal lies on +X axis in canonical frame
- No reflection step
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


def rot2(angle: float) -> np.ndarray:
    """
    Create 2D rotation matrix.
    
    Args:
        angle: Rotation angle in radians
    
    Returns:
        2x2 rotation matrix
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass
class CanonicalMeta:
    """Metadata for canonical frame transformation."""
    phi: float          # Rotation angle (goal direction)
    reflect: bool       # Whether Y was reflected
    origin: np.ndarray  # Original ego position
    dist_to_goal: float # Distance to goal


@dataclass
class Canonical5DMeta:
    """
    Metadata for 5D canonicalization used by the current FM training setup:
    state=[x, y, theta, v, r], control=[a, r_dot].
    """
    phi: float            # world->canonical rotation is rot2(-phi)
    origin_xy: np.ndarray # world start position [2]
    goal_dist: float      # ||goal - start||


def build_can_tf_5d(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> Dict[str, object]:
    """
    Build the canonical frame transform used by 5D FM planning.

    Returns a dict compatible with notebook usage:
      {
        'w2c': callable world->canonical XY,
        'c2w': callable canonical->world XY,
        'w2c_angle': callable world angle -> canonical angle,
        'start_can': [0, 0],
        'goal_can': [dist, 0],
        'dist': float,
        'phi': float,            # world heading of goal ray
        'angle_offset': float,   # alias of phi for notebook compatibility
      }
    """
    start = np.asarray(start_xy, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    gvec = goal - start
    dist = float(np.linalg.norm(gvec) + 1e-9)
    phi = float(np.arctan2(gvec[1], gvec[0]))

    R_wc = rot2(-phi)  # world -> canonical
    R_cw = rot2(phi)   # canonical -> world

    def world_to_can(xy):
        xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return (R_wc @ (xy_arr - start).T).T

    def can_to_world(xy_can):
        xy_arr = np.asarray(xy_can, dtype=np.float64).reshape(-1, 2)
        return (R_cw @ xy_arr.T).T + start

    def world_to_can_angle(theta_world):
        th = np.asarray(theta_world, dtype=np.float64)
        return _wrap_pi(th - phi)

    return {
        "w2c": world_to_can,
        "c2w": can_to_world,
        "w2c_angle": world_to_can_angle,
        "start_can": np.array([0.0, 0.0], dtype=np.float32),
        "goal_can": np.array([dist, 0.0], dtype=np.float32),
        "dist": dist,
        "phi": phi,
        "angle_offset": phi,
    }


def canonicalize(
    x_ego: np.ndarray,
    goal: np.ndarray,
    obs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, CanonicalMeta]:
    """
    Transform world coordinates to canonical frame.
    
    Canonical frame (project unified):
    - Origin at ego position
    - Positive X-axis towards goal
    - No reflection
    
    Args:
        x_ego: Ego agent position [2]
        goal: Goal position [2]
        obs: Obstacle/other agent position [2], optional
    
    Returns:
        cond: Conditioning vector [dist_to_goal, obs_x, obs_y] or [dist_to_goal]
        meta: Transformation metadata for inverse transform
    """
    x_ego = np.asarray(x_ego, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)

    goal_rel = goal - x_ego
    dist_to_goal = np.linalg.norm(goal_rel) + 1e-9

    # Goal direction becomes +X after world->canonical rotation.
    phi = np.arctan2(goal_rel[1], goal_rel[0])
    R = rot2(-phi)

    reflect = False  # kept for backward-compat metadata schema

    if obs is not None:
        obs = np.asarray(obs, dtype=np.float64).reshape(2)
        obs_rel = obs - x_ego
        obs_can = R @ obs_rel

        cond = np.array([dist_to_goal, obs_can[0], obs_can[1]], dtype=np.float32)
    else:
        cond = np.array([dist_to_goal], dtype=np.float32)
    
    meta = CanonicalMeta(
        phi=phi,
        reflect=reflect,
        origin=x_ego.copy(),
        dist_to_goal=dist_to_goal
    )
    
    return cond, meta


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def world_to_canonical_xy_5d(
    xy_world: np.ndarray,
    *,
    origin_xy: np.ndarray,
    phi: float
) -> np.ndarray:
    """
    Transform XY from world frame to 5D canonical frame (goal on +X).
    """
    xy = np.asarray(xy_world, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    R = rot2(-float(phi))
    return (R @ (xy - origin).T).T


def canonical_to_world_xy_5d(
    xy_can: np.ndarray,
    *,
    origin_xy: np.ndarray,
    phi: float
) -> np.ndarray:
    """
    Transform XY from 5D canonical frame (goal on +X) back to world frame.
    """
    xy = np.asarray(xy_can, dtype=np.float64)
    origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    R = rot2(float(phi))
    return (R @ xy.T).T + origin


def canonicalize_episode_5d_noobs(
    states_T5: np.ndarray,
    controls_T2: np.ndarray,
    goal_xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Canonical5DMeta]:
    """
    Canonicalize one episode for 5D FM training (no static obstacle fields).

    Inputs:
      states_T5:   (T,5) [x, y, theta, v, r] in world
      controls_T2: (T,2) [a, r_dot]
      goal_xy:     (2,) goal in world

    Returns:
      x1_5T: (5,T)  [x_can, y_can, theta_can, a, r_dot]
      cond5: (5,)   [goal_dist, v0, cos(theta0_can), sin(theta0_can), r0]
      meta:  Canonical5DMeta
    """
    st = np.asarray(states_T5, dtype=np.float64)
    u = np.asarray(controls_T2, dtype=np.float64)
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)

    if st.ndim != 2 or st.shape[1] < 5:
        raise ValueError(f"states_T5 must be (T,5+). Got {st.shape}")
    if u.ndim != 2 or u.shape[1] != 2 or u.shape[0] != st.shape[0]:
        raise ValueError(f"controls_T2 must be (T,2) with same T. Got {u.shape}, states {st.shape}")

    start_xy = st[0, 0:2].copy()
    g0 = g - start_xy
    goal_dist = float(np.linalg.norm(g0) + 1e-9)
    phi = float(np.arctan2(g0[1], g0[0]))

    xy_can = world_to_canonical_xy_5d(st[:, 0:2], origin_xy=start_xy, phi=phi)
    theta_can = _wrap_pi(st[:, 2] - phi)

    x1 = np.zeros((5, st.shape[0]), dtype=np.float64)
    x1[0, :] = xy_can[:, 0]
    x1[1, :] = xy_can[:, 1]
    x1[2, :] = theta_can
    x1[3, :] = u[:, 0]
    x1[4, :] = u[:, 1]

    v0 = float(st[0, 3])
    r0 = float(st[0, 4])
    theta0_can = float(theta_can[0])
    cond = np.array(
        [goal_dist, v0, float(np.cos(theta0_can)), float(np.sin(theta0_can)), r0],
        dtype=np.float32
    )

    meta = Canonical5DMeta(
        phi=phi,
        origin_xy=start_xy.astype(np.float64),
        goal_dist=goal_dist,
    )
    return x1.astype(np.float32), cond, meta


def build_planning_cond_5d(
    states_T5: np.ndarray,
    controls_T2: np.ndarray,
    goal_xy: np.ndarray,
    *,
    obs_center_world: Optional[np.ndarray] = None,
    obs_radius: Optional[float] = None,
    cond_dim: Optional[int] = None,
) -> Tuple[np.ndarray, Canonical5DMeta]:
    """
    Build planning-style conditioning tightly coupled to 5D canonicalization (+X goal).

    Base cond5 layout (legacy):
      [goal_dist, v0, cos(theta0_can), sin(theta0_can), r0]

    Recommended cond9 layout:
      [goal_dist, obs_x_can, obs_y_can, obs_r, v0, r0, cos(theta0_can), sin(theta0_can), has_obs]

    Args:
      states_T5, controls_T2, goal_xy: same as canonicalize_episode_5d_noobs
      obs_center_world: optional obstacle center in world [2]
      obs_radius: optional obstacle radius
      cond_dim:
        - None or 9: return cond9
        - 5: return legacy cond5

    Returns:
      cond: np.ndarray of shape (cond_dim,)
      meta: Canonical5DMeta
    """
    _, cond5, meta = canonicalize_episode_5d_noobs(states_T5, controls_T2, goal_xy)

    has_obs = 0.0
    obs_x_can = 0.0
    obs_y_can = 0.0
    obs_r = 0.0

    if obs_center_world is not None and obs_radius is not None:
        obs_world = np.asarray(obs_center_world, dtype=np.float64).reshape(1, 2)
        obs_can = world_to_canonical_xy_5d(
            obs_world,
            origin_xy=meta.origin_xy,
            phi=meta.phi,
        )[0]
        obs_x_can = float(obs_can[0])
        obs_y_can = float(obs_can[1])
        obs_r = float(obs_radius)
        has_obs = 1.0

    # cond5 indices: [goal_dist, v0, cos, sin, r0]
    cond9 = np.array(
        [
            float(cond5[0]),  # goal_dist
            obs_x_can,
            obs_y_can,
            obs_r,
            float(cond5[1]),  # v0
            float(cond5[4]),  # r0
            float(cond5[2]),  # cos(theta0_can)
            float(cond5[3]),  # sin(theta0_can)
            has_obs,
        ],
        dtype=np.float32,
    )

    if cond_dim is None or int(cond_dim) == 9:
        return cond9, meta
    if int(cond_dim) == 5:
        return cond5.astype(np.float32), meta
    raise ValueError(f"Unsupported cond_dim={cond_dim}. Use 5 or 9.")


def build_cond5_canonical_planning(
    start_xy_world: np.ndarray,
    goal_xy_world: np.ndarray,
    *,
    obs_center_world: Optional[np.ndarray] = None,
    obs_radius: float = 0.0,
) -> Tuple[np.ndarray, Canonical5DMeta]:
    """
    Build fixed 5D planning cond tightly tied to the +X canonical frame:
      cond5 = [g'_x, g'_y, o'_x, o'_y, R]

    Notes:
      - g' is goal in canonical frame. Under +X convention this is roughly [dist, 0].
      - o' is obstacle center in canonical frame (or zeros if no obstacle).
      - R is obstacle radius (or 0 if no obstacle).
    """
    start = np.asarray(start_xy_world, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xy_world, dtype=np.float64).reshape(2)

    g0 = goal - start
    goal_dist = float(np.linalg.norm(g0) + 1e-9)
    phi = float(np.arctan2(g0[1], g0[0]))
    meta = Canonical5DMeta(phi=phi, origin_xy=start.copy(), goal_dist=goal_dist)

    g_can = world_to_canonical_xy_5d(goal.reshape(1, 2), origin_xy=meta.origin_xy, phi=meta.phi)[0]

    if obs_center_world is None:
        o_can = np.zeros(2, dtype=np.float64)
        r = 0.0
    else:
        obs = np.asarray(obs_center_world, dtype=np.float64).reshape(1, 2)
        o_can = world_to_canonical_xy_5d(obs, origin_xy=meta.origin_xy, phi=meta.phi)[0]
        r = float(obs_radius)

    cond5 = np.array(
        [float(g_can[0]), float(g_can[1]), float(o_can[0]), float(o_can[1]), float(r)],
        dtype=np.float32,
    )
    return cond5, meta


def uncanonicalize(v_can: np.ndarray, meta: CanonicalMeta) -> np.ndarray:
    """
    Transform velocity from canonical frame back to world frame.
    
    Args:
        v_can: Velocity in canonical frame [2]
        meta: Transformation metadata
    
    Returns:
        Velocity in world frame [2]
    """
    v_can = np.asarray(v_can, dtype=np.float64).reshape(2)
    v = v_can.copy()
    
    # Undo reflection
    if meta.reflect:
        v[1] *= -1
    
    # Undo rotation
    R = rot2(meta.phi)
    return R @ v


def canonicalize_trajectory(
    traj: np.ndarray,
    x_ego: np.ndarray,
    goal: np.ndarray,
    obs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, CanonicalMeta]:
    """
    Transform an entire trajectory to canonical frame.
    
    Args:
        traj: Trajectory [T, 2]
        x_ego: Ego position at start [2]
        goal: Goal position [2]
        obs: Obstacle position [2], optional
    
    Returns:
        traj_can: Transformed trajectory [T, 2]
        meta: Transformation metadata
    """
    traj = np.asarray(traj, dtype=np.float64)
    x_ego = np.asarray(x_ego, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    
    # Get transformation
    cond, meta = canonicalize(x_ego, goal, obs)
    
    # Apply to all trajectory points
    R = rot2(-meta.phi)
    traj_rel = traj - meta.origin
    traj_can = (R @ traj_rel.T).T
    
    if meta.reflect:
        traj_can[:, 1] *= -1
    
    return traj_can, meta


def uncanonicalize_trajectory(
    traj_can: np.ndarray,
    meta: CanonicalMeta
) -> np.ndarray:
    """
    Transform trajectory from canonical frame back to world frame.
    
    Args:
        traj_can: Trajectory in canonical frame [T, 2]
        meta: Transformation metadata
    
    Returns:
        Trajectory in world frame [T, 2]
    """
    traj_can = np.asarray(traj_can, dtype=np.float64)
    traj = traj_can.copy()
    
    # Undo reflection
    if meta.reflect:
        traj[:, 1] *= -1
    
    # Undo rotation and translation
    R = rot2(meta.phi)
    traj = (R @ traj.T).T + meta.origin
    
    return traj


def canonicalize_velocity_sequence(
    v_seq: np.ndarray,
    meta: CanonicalMeta
) -> np.ndarray:
    """
    Transform velocity sequence to canonical frame.
    
    Args:
        v_seq: Velocity sequence [T, 2]
        meta: Transformation metadata
    
    Returns:
        Transformed velocity sequence [T, 2]
    """
    v_seq = np.asarray(v_seq, dtype=np.float64)
    R = rot2(-meta.phi)
    v_can = (R @ v_seq.T).T
    
    if meta.reflect:
        v_can[:, 1] *= -1
    
    return v_can


def uncanonicalize_velocity_sequence(
    v_can: np.ndarray,
    meta: CanonicalMeta
) -> np.ndarray:
    """
    Transform velocity sequence from canonical frame back to world frame.
    
    Args:
        v_can: Velocity sequence in canonical frame [T, 2]
        meta: Transformation metadata
    
    Returns:
        Velocity sequence in world frame [T, 2]
    """
    v_can = np.asarray(v_can, dtype=np.float64)
    v = v_can.copy()
    
    if meta.reflect:
        v[:, 1] *= -1
    
    R = rot2(meta.phi)
    return (R @ v.T).T


def canonicalize_runtime_obs_xy(
    x_ego: np.ndarray,
    goal: np.ndarray,
    obs: np.ndarray
) -> Tuple[np.ndarray, Dict]:
    """
    Runtime canonicalization for obstacle avoidance.
    Returns cond vector and simple metadata dict.
    
    Args:
        x_ego: Current ego position [2]
        goal: Goal position [2]
        obs: Obstacle position [2]
    
    Returns:
        cond3: Conditioning vector [dist, obs_x, obs_y]
        meta: Dict with 'phi' and 'reflect' keys
    """
    x_ego = np.asarray(x_ego, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    obs = np.asarray(obs, dtype=np.float64).reshape(2)
    
    goal_rel = goal - x_ego
    obs_rel = obs - x_ego
    
    dist = np.linalg.norm(goal_rel) + 1e-9
    phi = np.arctan2(goal_rel[1], goal_rel[0])
    
    R = rot2(-phi)
    obs_can = R @ obs_rel
    
    reflect = False
    if obs_can[1] < 0:
        obs_can[1] *= -1
        reflect = True
    
    cond3 = np.array([dist, obs_can[0], obs_can[1]], dtype=np.float32)
    meta = {"phi": phi, "reflect": reflect}
    
    return cond3, meta


def uncanonicalize_v(v_can: np.ndarray, meta: Dict) -> np.ndarray:
    """
    Simple velocity uncanonicalization using dict metadata.
    
    Args:
        v_can: Velocity in canonical frame [2]
        meta: Dict with 'phi' and 'reflect' keys
    
    Returns:
        Velocity in world frame [2]
    """
    v = np.asarray(v_can, dtype=np.float64).reshape(2).copy()
    
    if meta["reflect"]:
        v[1] *= -1
    
    R = rot2(meta["phi"])
    return R @ v

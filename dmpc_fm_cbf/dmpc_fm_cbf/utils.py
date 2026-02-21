"""
Utility functions for DMPC + Flow Matching + CBF framework.
"""

import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numpy(x) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x, device: Optional[torch.device] = None, dtype=torch.float32) -> torch.Tensor:
    """Convert array to tensor."""
    if torch.is_tensor(x):
        t = x.to(dtype=dtype)
    else:
        t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return float(angle)


def normalize_vector(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < eps:
        return v
    return v / norm


def clip_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    """Clip velocity to maximum speed."""
    v = np.asarray(v, dtype=np.float64)
    speed = np.linalg.norm(v)
    if speed > v_max:
        return v_max * v / speed
    return v


def ema_smooth(v_new: np.ndarray, v_prev: Optional[np.ndarray], beta: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    if v_prev is None:
        return v_new
    return beta * v_prev + (1 - beta) * v_new


def goal_pull(v: np.ndarray, x: np.ndarray, goal: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """
    Softly pull velocity towards goal direction if pointing away.
    
    Args:
        v: Current velocity
        x: Current position
        goal: Goal position
        alpha: Blending factor
    
    Returns:
        Adjusted velocity
    """
    d = goal - x
    d_norm = normalize_vector(d)
    v_norm = normalize_vector(v)
    
    if np.dot(v_norm, d_norm) < 0.0:
        speed = np.linalg.norm(v)
        v = (1 - alpha) * v + alpha * speed * d_norm
    return v


def accel_limit(v_new: np.ndarray, v_prev: Optional[np.ndarray], 
                a_max: float, dt: float) -> np.ndarray:
    """
    Limit acceleration magnitude per timestep.
    
    Args:
        v_new: Desired new velocity
        v_prev: Previous velocity
        a_max: Maximum acceleration
        dt: Timestep
    
    Returns:
        Acceleration-limited velocity
    """
    if v_prev is None:
        return v_new
    
    dv = v_new - v_prev
    dv_norm = np.linalg.norm(dv)
    max_dv = a_max * dt
    
    if dv_norm > max_dv:
        dv = max_dv * dv / dv_norm
    
    return v_prev + dv


class SoftHinge:
    """Soft hinge penalty function: max(0, z)^p"""
    
    def __init__(self, power: float = 2.0):
        self.power = power
    
    def __call__(self, z: float) -> float:
        return max(0.0, float(z)) ** self.power


def distance_cost(x: np.ndarray, other: np.ndarray, d_safe: float, 
                  weight: float = 1.0, power: float = 2.0) -> float:
    """
    Compute soft collision avoidance cost.
    
    Args:
        x: Current position
        other: Other agent position  
        d_safe: Safe distance threshold
        weight: Cost weight
        power: Penalty power
    
    Returns:
        Cost value
    """
    dist = np.linalg.norm(x - other)
    violation = max(0.0, d_safe - dist)
    return weight * (violation ** power)


def goal_cost(x: np.ndarray, goal: np.ndarray, weight: float = 1.0) -> float:
    """Compute quadratic goal reaching cost."""
    return weight * np.sum((x - goal) ** 2)


def smoothness_cost(heading: float, prev_heading: Optional[float], 
                    weight: float = 1.0) -> float:
    """Compute heading change smoothness cost."""
    if prev_heading is None:
        return 0.0
    dh = wrap_angle(heading - prev_heading)
    return weight * (dh ** 2)

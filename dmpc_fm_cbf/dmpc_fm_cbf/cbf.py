"""
Control Barrier Function (CBF) Implementation for Collision Avoidance.

This module provides CBF-based velocity projection to ensure safety constraints
are satisfied. Supports:
- Elliptical/circular static obstacles
- Dynamic inter-agent collision avoidance with pressure-based adaptive gains
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union


def ellipse_h_grad_batch(
    x_world_2N: torch.Tensor,
    ellipse: Dict
) -> tuple:
    """
    Compute ellipse barrier function value and gradient.
    
    h(x) = (a*(x-cx))^2 + (b*(y-cy))^2 - 1
    
    Args:
        x_world_2N: Positions in world coords [2, N]
        ellipse: Dict with 'center' and either ('a', 'b') or 'radius'
    
    Returns:
        h_e: Barrier values [N]
        grad: Gradients [N, 2]
    """
    if not torch.is_tensor(x_world_2N):
        x_world_2N = torch.as_tensor(x_world_2N, dtype=torch.float32)
    
    device = x_world_2N.device
    dtype = x_world_2N.dtype
    
    center = ellipse["center"]
    if not torch.is_tensor(center):
        center = torch.as_tensor(center, dtype=dtype, device=device)
    else:
        center = center.to(device=device, dtype=dtype)
    
    # Get ellipse parameters
    if "a" in ellipse or "b" in ellipse:
        a = float(ellipse.get("a", 1.0))
        b = float(ellipse.get("b", 1.0))
    elif "radius" in ellipse:
        r = float(ellipse["radius"])
        a = 1.0 / max(r, 1e-9)
        b = a
    else:
        raise KeyError("ellipse must have keys {'center','a','b'} or {'center','radius'}")
    
    # Compute barrier function
    x_N2 = x_world_2N.transpose(0, 1)  # [N, 2]
    delta = x_N2 - center.view(1, 2)    # [N, 2]
    
    ax = a * delta[:, 0]
    by = b * delta[:, 1]
    h_e = ax * ax + by * by - 1.0
    
    # Compute gradient
    grad = torch.stack([
        2.0 * (a ** 2) * delta[:, 0],
        2.0 * (b ** 2) * delta[:, 1],
    ], dim=1)  # [N, 2]
    
    return h_e, grad


@torch.no_grad()
def cbf_project_velocity_world(
    x_1x2N: torch.Tensor,
    v_nom_1x2N: torch.Tensor,
    ellipses: Optional[List[Dict]] = None,
    *,
    extra_margin: float = 0.0,
    max_corr_norm: float = 2.0,
    passes: int = 5,
    other_agents_traj: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    agent_safe_radius: float = 0.5,
    gamma_min: float = 1.0,
    w_p: float = 5.0,
    press_k: float = 1.0,
    press_n: float = 2.0,
) -> torch.Tensor:
    """
    Project nominal velocity to satisfy CBF constraints.
    
    Uses adaptive class-K function with pressure-based gain:
        gamma(x) = gamma_min + w_p * sum_j(press_k / ||x - x_j||^press_n)
    
    CBF constraint: dh/dt + gamma(x) * h(x) >= 0
                    grad_h · v + gamma(x) * h(x) >= 0
    
    Args:
        x_1x2N: Positions [1, 2, N]
        v_nom_1x2N: Nominal velocities [1, 2, N]
        ellipses: List of ellipse obstacles
        extra_margin: Additional safety margin
        max_corr_norm: Maximum correction norm per pass
        passes: Number of projection passes
        other_agents_traj: Other agent trajectories for inter-agent CBF
        agent_safe_radius: Safe radius for inter-agent collision
        gamma_min: Minimum CBF gain
        w_p: Pressure weight
        press_k: Pressure numerator constant
        press_n: Pressure distance power
    
    Returns:
        Safe velocity [1, 2, N]
    """
    ellipses = ellipses or []
    has_ellipses = len(ellipses) > 0
    
    # Normalize other_agents_traj to list of [1, 2, N] tensors
    others: List[torch.Tensor] = []
    if other_agents_traj is not None:
        if isinstance(other_agents_traj, list):
            others = other_agents_traj
        elif torch.is_tensor(other_agents_traj):
            t = other_agents_traj
            if t.dim() == 3 and t.shape[0] == 1 and t.shape[1] == 2:
                others = [t]
            elif t.dim() == 3 and t.shape[1] == 2:
                # [M, 2, N] -> list of [1, 2, N]
                others = [t[i].unsqueeze(0) for i in range(t.shape[0])]
            elif t.dim() == 4 and t.shape[1] == 1 and t.shape[2] == 2:
                # [M, 1, 2, N] -> list of [1, 2, N]
                others = [t[i] for i in range(t.shape[0])]
            else:
                raise ValueError(f"Unsupported other_agents_traj shape: {tuple(t.shape)}")
        else:
            raise TypeError("other_agents_traj must be None, list[Tensor], or Tensor")
    
    # Unify device and dtype
    device = x_1x2N.device
    dtype = x_1x2N.dtype
    others = [o.to(device=device, dtype=dtype) for o in others]
    
    has_others = len(others) > 0
    if not has_ellipses and not has_others:
        return v_nom_1x2N
    
    B, D, N = x_1x2N.shape
    assert (B, D) == (1, 2), f"expected (1, 2, N), got {tuple(x_1x2N.shape)}"
    assert v_nom_1x2N.shape == x_1x2N.shape
    
    x_world_2N = x_1x2N.squeeze(0)  # [2, N]
    v_now = v_nom_1x2N.squeeze(0)   # [2, N]
    
    # Compute pressure-based adaptive gain
    if has_others:
        P_total = torch.zeros(N, device=device, dtype=dtype)
        x_self_N2 = x_world_2N.transpose(0, 1)  # [N, 2]
        
        for other in others:
            assert other.shape == (1, 2, N), f"each other must be (1,2,N), got {tuple(other.shape)}"
            x_other_N2 = other.squeeze(0).transpose(0, 1)  # [N, 2]
            delta = x_self_N2 - x_other_N2
            dist = torch.norm(delta, dim=1).clamp_min(1e-3)
            P_total = P_total + (press_k / dist.pow(press_n))
        
        gamma_x = gamma_min + w_p * P_total
    else:
        gamma_x = torch.full((N,), float(gamma_min), device=device, dtype=dtype)
    
    # Iterative projection
    for _ in range(max(1, int(passes))):
        all_s = []
        all_a = []
        
        v_T = v_now.transpose(0, 1)  # [N, 2]
        
        # Static ellipse constraints
        if has_ellipses:
            for e in ellipses:
                h_e, grad_w = ellipse_h_grad_batch(x_world_2N, e)
                h_eff = h_e - float(extra_margin)
                s_e = (grad_w * v_T).sum(dim=1) + gamma_x * h_eff
                all_s.append(s_e)
                all_a.append(grad_w)
        
        # Inter-agent constraints
        if has_others:
            x_self_N2 = x_world_2N.transpose(0, 1)  # [N, 2]
            for other in others:
                x_other_N2 = other.squeeze(0).transpose(0, 1)  # [N, 2]
                delta = x_self_N2 - x_other_N2  # [N, 2]
                dist = torch.norm(delta, dim=1).clamp_min(1e-9)  # [N]
                
                # h = ||x - x_other||^2 - r^2
                r = float(agent_safe_radius)
                h_agent = dist * dist - r * r
                h_eff = h_agent - float(extra_margin)
                
                # grad_h = 2 * (x - x_other)
                grad_h = 2.0 * delta  # [N, 2]
                
                s_agent = (grad_h * v_T).sum(dim=1) + gamma_x * h_eff
                all_s.append(s_agent)
                all_a.append(grad_h)
        
        if len(all_s) == 0:
            break
        
        # Stack constraints
        s_all = torch.stack(all_s, dim=1)  # [N, num_constraints]
        a_all = torch.stack(all_a, dim=1)  # [N, num_constraints, 2]
        
        # Find most violated constraint
        idx = s_all.argmin(dim=1)  # [N]
        
        s_min = s_all[torch.arange(N, device=device), idx]  # [N]
        a_min = a_all[torch.arange(N, device=device), idx]  # [N, 2]
        
        # Project velocity for violated constraints
        mask = (s_min < 0).float()  # [N]
        
        a_norm_sq = (a_min ** 2).sum(dim=1).clamp_min(1e-9)  # [N]
        corr_scale = (-s_min / a_norm_sq).clamp(min=0)  # [N]
        
        # Limit correction magnitude
        corr = corr_scale.unsqueeze(1) * a_min  # [N, 2]
        corr_norm = torch.norm(corr, dim=1, keepdim=True).clamp_min(1e-9)
        corr = corr * (torch.clamp(corr_norm, max=max_corr_norm) / corr_norm)
        
        # Apply correction only where violated
        v_now = v_now + (mask.unsqueeze(0) * corr.transpose(0, 1))
    
    return v_now.unsqueeze(0)


def cbf_project_velocity_step(
    x_now: np.ndarray,
    v_nom: np.ndarray,
    other_now: np.ndarray,
    *,
    ellipses: Optional[List[Dict]] = None,
    agent_safe_radius: float = 0.6,
    extra_margin: float = 0.0,
    max_corr_norm: float = 10.0,
    gamma_min: float = 6.0,
    w_p: float = 6.0,
    press_k: float = 1.0,
    press_n: float = 2.0,
) -> np.ndarray:
    """
    Single-step CBF velocity projection (numpy interface).
    
    Args:
        x_now: Current position [2]
        v_nom: Nominal velocity [2]
        other_now: Other agent position [2]
        ellipses: Static obstacles (optional)
        agent_safe_radius: Safe distance
        extra_margin: Additional margin
        max_corr_norm: Max correction magnitude
        gamma_min: Minimum CBF gain
        w_p: Pressure weight
        press_k: Pressure constant
        press_n: Pressure power
    
    Returns:
        Safe velocity [2]
    """
    x_now = np.asarray(x_now, dtype=np.float64).reshape(2)
    v_nom = np.asarray(v_nom, dtype=np.float64).reshape(2)
    other_now = np.asarray(other_now, dtype=np.float64).reshape(2)
    
    # Convert to tensors
    x_traj = torch.tensor(x_now, dtype=torch.float32).view(1, 2, 1)
    v_traj = torch.tensor(v_nom, dtype=torch.float32).view(1, 2, 1)
    other_traj = torch.tensor(other_now, dtype=torch.float32).view(1, 2, 1)
    
    v_safe = cbf_project_velocity_world(
        x_traj,
        v_traj,
        ellipses=ellipses or [],
        extra_margin=float(extra_margin),
        max_corr_norm=float(max_corr_norm),
        passes=5,
        other_agents_traj=other_traj,
        agent_safe_radius=float(agent_safe_radius),
        gamma_min=float(gamma_min),
        w_p=float(w_p),
        press_k=float(press_k),
        press_n=float(press_n),
    )
    
    return v_safe.squeeze(0).squeeze(-1).numpy().astype(np.float64)

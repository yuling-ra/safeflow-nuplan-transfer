"""
DMPC Controllers for Multi-Agent Trajectory Planning.

Provides:
- TwoCarDMPCController: Full-featured two-car DMPC with FM and CBF
- two_car_fm_dmpc: FM-based DMPC with CBF safety
- two_car_pure_dmpc: Pure DMPC without FM (baseline)
- two_car_dmpc_with_cbf: DMPC with CBF execution shield
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass

from .utils import (
    wrap_angle, normalize_vector, clip_speed, 
    ema_smooth, goal_pull, accel_limit
)
from .canonical import canonicalize_runtime_obs_xy, uncanonicalize_v
from .cbf import cbf_project_velocity_step, cbf_project_velocity_world
from .ode_solver import sample_v_seq_fm_odeint

# Try importing torchdiffeq
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


@dataclass
class DMPCConfig:
    """Configuration for DMPC controller."""
    # Timing
    dt: float = 0.1
    max_steps: int = 500
    goal_tol: float = 0.2
    
    # Horizon
    H: int = 15
    K: int = 3  # Number of candidate samples
    
    # Speed limits
    v_max_A: float = 1.4
    v_max_B: float = 1.4
    speed_scale_A: float = 1.0
    speed_scale_B: float = 0.4  # B yields more
    
    # Smoothing
    ema_beta: float = 0.9
    heading_smooth: float = 0.35
    a_max: float = 2.0
    
    # CBF parameters
    agent_safe_radius: float = 0.6
    gamma_min: float = 6.0
    w_p: float = 6.0
    press_k: float = 1.0
    press_n: float = 2.0
    max_corr_norm: float = 10.0
    
    # Cost weights
    w_goal: float = 45.0
    w_smooth: float = 2.0
    w_dist: float = 80.0
    d_safe: float = 1.0
    
    # FM ODE settings
    ode_steps: int = 20
    noise_scale: float = 1.0
    solver_method: str = "euler"
    solver_tol: float = 1e-5
    
    # Replanning
    replan_every: int = 5
    r_stop: float = 1.0
    k_goal_speed: float = 2.0


def _get_model_cond_dim(model: nn.Module) -> Optional[int]:
    """Infer conditioning dimension from model."""
    if hasattr(model, "cond_dim"):
        return model.cond_dim
    if hasattr(model, "cond_mlp") and model.cond_mlp is not None:
        for layer in model.cond_mlp:
            if hasattr(layer, "in_features"):
                return layer.in_features
    return None


def _eval_trajectory_cost(
    x0: np.ndarray,
    goal: np.ndarray,
    v_seq: np.ndarray,
    other_pred: np.ndarray,
    dt: float,
    w_goal: float = 45.0,
    w_smooth: float = 2.0,
    w_dist: float = 80.0,
    d_safe: float = 1.0,
) -> float:
    """
    Evaluate trajectory cost.
    
    Args:
        x0: Starting position [2]
        goal: Goal position [2]
        v_seq: Velocity sequence [H, 2]
        other_pred: Other agent predicted positions [H+1, 2]
        dt: Timestep
        w_goal: Goal weight
        w_smooth: Smoothness weight
        w_dist: Distance violation weight
        d_safe: Safe distance threshold
    
    Returns:
        Total cost
    """
    H = len(v_seq)
    x = x0.copy()
    J = 0.0
    
    # Rollout and accumulate costs
    for k in range(H):
        # Update position
        x = x + dt * v_seq[k]
        
        # Distance violation cost
        dist = np.linalg.norm(x - other_pred[min(k + 1, len(other_pred) - 1)])
        violation = max(0.0, d_safe - dist)
        J += w_dist * (violation ** 2)
        
        # Smoothness cost
        if k > 0:
            dv = v_seq[k] - v_seq[k - 1]
            J += w_smooth * np.sum(dv ** 2)
    
    # Terminal goal cost
    J += w_goal * np.sum((x - goal) ** 2)
    
    return J


def _predict_straight_line(
    x0: np.ndarray,
    goal: np.ndarray,
    v_max: float,
    H: int,
    dt: float
) -> np.ndarray:
    """Predict straight-line trajectory towards goal."""
    x = x0.copy()
    traj = [x.copy()]
    
    d = goal - x
    d_norm = np.linalg.norm(d)
    if d_norm > 1e-6:
        v = v_max * d / d_norm
    else:
        v = np.zeros(2)
    
    for _ in range(H):
        x = x + dt * v
        traj.append(x.copy())
    
    return np.array(traj)


def two_car_fm_dmpc(
    xA0: np.ndarray,
    gA: np.ndarray,
    xB0: np.ndarray,
    gB: np.ndarray,
    model: nn.Module,
    *,
    dt: float = 0.1,
    H: int = 15,
    K: int = 3,
    v_max_A: float = 1.4,
    v_max_B: float = 1.4,
    speed_scale_A: float = 1.0,
    speed_scale_B: float = 0.4,
    max_steps: int = 500,
    goal_tol: float = 0.2,
    ema_beta: float = 0.9,
    a_max: float = 2.0,
    r_stop: float = 1.0,
    k_goal_speed: float = 2.0,
    w_goal: float = 45.0,
    w_smooth: float = 2.0,
    w_dist: float = 80.0,
    d_safe: float = 1.0,
    agent_safe_radius: float = 0.6,
    gamma_min: float = 6.0,
    w_p: float = 6.0,
    press_k: float = 1.0,
    press_n: float = 2.0,
    max_corr_norm: float = 10.0,
    ode_steps: int = 20,
    noise_scale: float = 1.0,
    solver_method: str = "euler",
    solver_tol: float = 1e-5,
    device: str = "cpu",
    debug_every: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-car DMPC with Flow Matching trajectory generation.
    
    Args:
        xA0, gA: Agent A start and goal
        xB0, gB: Agent B start and goal
        model: Flow Matching velocity model
        dt: Timestep
        H: Planning horizon
        K: Number of candidate trajectories
        v_max_A, v_max_B: Maximum velocities
        speed_scale_A, speed_scale_B: Speed scaling (B < A for yielding)
        max_steps: Maximum simulation steps
        goal_tol: Goal tolerance
        ema_beta: EMA smoothing factor
        a_max: Max acceleration for smoothing
        r_stop: Slowdown radius
        k_goal_speed: Goal speed coefficient
        w_goal, w_smooth, w_dist, d_safe: Cost weights
        agent_safe_radius: CBF safe radius
        gamma_min, w_p, press_k, press_n, max_corr_norm: CBF parameters
        ode_steps: ODE integration steps
        noise_scale: Initial noise scale
        solver_method: ODE solver
        solver_tol: Solver tolerance
        device: Compute device
        debug_every: Print debug info every N steps
    
    Returns:
        xsA: Agent A trajectory [steps, 2]
        xsB: Agent B trajectory [steps, 2]
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    
    # Initialize
    xA = np.asarray(xA0, dtype=np.float64).reshape(2).copy()
    xB = np.asarray(xB0, dtype=np.float64).reshape(2).copy()
    gA = np.asarray(gA, dtype=np.float64).reshape(2)
    gB = np.asarray(gB, dtype=np.float64).reshape(2)
    
    xsA = [xA.copy()]
    xsB = [xB.copy()]
    
    prev_vA = None
    prev_vB = None
    
    # Time span for ODE
    T_SPAN = torch.linspace(0, 1, ode_steps + 1, device=dev)
    
    # Get conditioning dimension
    cond_dim = _get_model_cond_dim(model)
    
    def sample_v_seq(x_self, goal, other_pos, speed_scale, v_max):
        """Sample velocity sequence using FM-ODE."""
        # Canonicalize
        cond3, meta = canonicalize_runtime_obs_xy(x_self, goal, other_pos)
        
        # Pad condition
        cond = cond3.copy()
        if cond_dim is not None and len(cond) < cond_dim:
            cond = np.concatenate([cond, np.zeros(cond_dim - len(cond), dtype=np.float32)])
        
        # Sample
        v_seq_can = sample_v_seq_fm_odeint(
            model=model,
            cond=cond,
            seq_len=H,
            T_SPAN=T_SPAN,
            solver_method=solver_method,
            solver_tol=solver_tol,
            noise_scale=noise_scale,
            device=str(dev),
        )
        
        # Transform to world frame and apply speed scaling
        v_seq_world = np.zeros((H, 2), dtype=np.float64)
        xk = x_self.copy()
        
        for k in range(H):
            # Distance-based speed scaling
            dist_goal = np.linalg.norm(xk - goal)
            speed_cap = min(v_max * speed_scale, k_goal_speed * dist_goal)
            
            # Transform velocity
            v_w = uncanonicalize_v(v_seq_can[k], meta)
            v_norm = np.linalg.norm(v_w) + 1e-9
            v_seq_world[k] = speed_cap * v_w / v_norm
            
            xk = xk + dt * v_seq_world[k]
        
        return v_seq_world
    
    def pick_best_v_seq(x_self, goal, other_pos, other_pred, speed_scale, v_max):
        """Sample K trajectories and pick best."""
        best_cost = float('inf')
        best_seq = None
        
        for _ in range(K):
            v_seq = sample_v_seq(x_self, goal, other_pos, speed_scale, v_max)
            cost = _eval_trajectory_cost(
                x_self, goal, v_seq, other_pred, dt,
                w_goal=w_goal, w_smooth=w_smooth, w_dist=w_dist, d_safe=d_safe
            )
            if cost < best_cost:
                best_cost = cost
                best_seq = v_seq.copy()
        
        return best_seq if best_seq is not None else sample_v_seq(x_self, goal, other_pos, speed_scale, v_max)
    
    def rollout_pred(x0, v_seq):
        """Rollout position trajectory from velocities."""
        x = x0.copy()
        traj = [x.copy()]
        for v in v_seq:
            x = x + dt * v
            traj.append(x.copy())
        return np.array(traj)
    
    # Main loop
    for step in range(max_steps):
        # Check goal reached
        dA = np.linalg.norm(xA - gA)
        dB = np.linalg.norm(xB - gB)
        
        if dA <= goal_tol and dB <= goal_tol:
            break
        
        # Initial predictions (straight line)
        predA = _predict_straight_line(xA, gA, v_max_A, H, dt)
        predB = _predict_straight_line(xB, gB, v_max_B, H, dt)
        
        # Agent A plans against B's prediction
        doneA = dA <= goal_tol
        if not doneA:
            seqA = pick_best_v_seq(xA, gA, xB, predB, speed_scale_A, v_max_A)
            predA = rollout_pred(xA, seqA)
            vA0 = seqA[0].copy()
        else:
            vA0 = np.zeros(2)
        
        # Agent B plans against A's updated prediction
        doneB = dB <= goal_tol
        if not doneB:
            seqB = pick_best_v_seq(xB, gB, xA, predA, speed_scale_B, v_max_B)
            vB0 = seqB[0].copy()
        else:
            vB0 = np.zeros(2)
        
        # Smooth velocities
        vA_exec = ema_smooth(vA0, prev_vA, ema_beta)
        vB_exec = ema_smooth(vB0, prev_vB, ema_beta)
        
        # Acceleration limiting
        vA_exec = accel_limit(vA_exec, prev_vA, a_max, dt)
        vB_exec = accel_limit(vB_exec, prev_vB, a_max, dt)
        
        # Goal pull
        vA_exec = goal_pull(vA_exec, xA, gA, alpha=0.15)
        vB_exec = goal_pull(vB_exec, xB, gB, alpha=0.20)
        
        # CBF safety projection
        vA_safe = cbf_project_velocity_step(
            xA, vA_exec, xB,
            agent_safe_radius=agent_safe_radius,
            gamma_min=gamma_min, w_p=w_p,
            press_k=press_k, press_n=press_n,
            max_corr_norm=max_corr_norm,
        )
        vB_safe = cbf_project_velocity_step(
            xB, vB_exec, xA,
            agent_safe_radius=agent_safe_radius,
            gamma_min=gamma_min, w_p=w_p,
            press_k=press_k, press_n=press_n,
            max_corr_norm=max_corr_norm,
        )
        
        # Update positions
        xA = xA + dt * vA_safe
        xB = xB + dt * vB_safe
        
        xsA.append(xA.copy())
        xsB.append(xB.copy())
        
        prev_vA = vA_safe.copy()
        prev_vB = vB_safe.copy()
        
        # Debug output
        if debug_every and step % debug_every == 0:
            dist_AB = np.linalg.norm(xA - xB)
            print(f"[step {step:03d}] dist_AB={dist_AB:.3f} |dA|={dA:.2f} |dB|={dB:.2f}")
    
    return np.array(xsA), np.array(xsB)


def two_car_pure_dmpc(
    xA0: np.ndarray,
    gA: np.ndarray,
    xB0: np.ndarray,
    gB: np.ndarray,
    *,
    dt: float = 0.1,
    H: int = 15,
    v_max_A: float = 1.4,
    v_max_B: float = 1.4,
    max_steps: int = 500,
    goal_tol: float = 0.2,
    heading_span_deg: float = 140.0,
    heading_samples: int = 25,
    speed_scales: Tuple[float, ...] = (0.0, 0.4, 0.7, 1.0),
    d_safe: float = 1.0,
    w_goal: float = 18.0,
    w_turn: float = 0.10,
    w_dist_A: float = 70.0,
    w_dist_B: float = 110.0,
    w_speed: float = 0.6,
    heading_smooth: float = 0.35,
    n_best_response_iters: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure DMPC baseline without Flow Matching.
    
    Uses heading + speed grid search for trajectory optimization.
    """
    xA = np.asarray(xA0, dtype=np.float64).reshape(2).copy()
    xB = np.asarray(xB0, dtype=np.float64).reshape(2).copy()
    gA = np.asarray(gA, dtype=np.float64).reshape(2)
    gB = np.asarray(gB, dtype=np.float64).reshape(2)
    
    xsA = [xA.copy()]
    xsB = [xB.copy()]
    
    prev_hA = None
    prev_hB = None
    
    span = np.deg2rad(heading_span_deg)
    
    def predict_straight(x0, h, s, v_max):
        """Predict trajectory with given heading and speed."""
        x = x0.copy()
        traj = [x.copy()]
        v = v_max * s * np.array([np.cos(h), np.sin(h)])
        for _ in range(H):
            x = x + dt * v
            traj.append(x.copy())
        return np.array(traj)
    
    def score_agent(x0, goal, other_pred, h, s, v_max, w_dist, prev_h):
        """Score a candidate heading and speed."""
        x = x0.copy()
        J = 0.0
        v = v_max * s * np.array([np.cos(h), np.sin(h)])
        
        for k in range(H):
            x = x + dt * v
            dist = np.linalg.norm(x - other_pred[min(k + 1, len(other_pred) - 1)])
            violation = max(0.0, d_safe - dist)
            J += w_dist * (violation ** 2)
        
        # Goal cost
        J += w_goal * np.sum((x - goal) ** 2)
        
        # Turn cost
        if prev_h is not None:
            dh = wrap_angle(h - prev_h)
            J += w_turn * (dh ** 2)
        
        # Speed penalty (discourage yielding too much)
        J += w_speed * ((1.0 - s) ** 2)
        
        return J
    
    for step in range(max_steps):
        if np.linalg.norm(xA - gA) <= goal_tol and np.linalg.norm(xB - gB) <= goal_tol:
            break
        
        # Initialize headings
        hA = np.arctan2((gA - xA)[1], (gA - xA)[0])
        hB = np.arctan2((gB - xB)[1], (gB - xB)[0])
        sA, sB = 1.0, 1.0
        
        # Best response iterations
        for _ in range(n_best_response_iters):
            predA = predict_straight(xA, hA, sA, v_max_A)
            predB = predict_straight(xB, hB, sB, v_max_B)
            
            # A best response
            baseA = np.arctan2((gA - xA)[1], (gA - xA)[0])
            offsets = np.linspace(-span / 2, span / 2, heading_samples)
            bestJA, bestHA, bestSA = float('inf'), baseA, 1.0
            
            for off in offsets:
                cand_h = baseA + off
                for cand_s in speed_scales:
                    J = score_agent(xA, gA, predB, cand_h, cand_s, v_max_A, w_dist_A, prev_hA)
                    if J < bestJA:
                        bestJA, bestHA, bestSA = J, cand_h, cand_s
            hA, sA = bestHA, bestSA
            
            # B best response
            baseB = np.arctan2((gB - xB)[1], (gB - xB)[0])
            bestJB, bestHB, bestSB = float('inf'), baseB, 1.0
            
            for off in offsets:
                cand_h = baseB + off
                for cand_s in speed_scales:
                    J = score_agent(xB, gB, predA, cand_h, cand_s, v_max_B, w_dist_B, prev_hB)
                    if J < bestJB:
                        bestJB, bestHB, bestSB = J, cand_h, cand_s
            hB, sB = bestHB, bestSB
        
        # Smooth heading
        if prev_hA is not None:
            alpha = heading_smooth
            hA = prev_hA + alpha * wrap_angle(hA - prev_hA)
        if prev_hB is not None:
            alpha = heading_smooth
            hB = prev_hB + alpha * wrap_angle(hB - prev_hB)
        
        # Execute
        vA = v_max_A * sA * np.array([np.cos(hA), np.sin(hA)])
        vB = v_max_B * sB * np.array([np.cos(hB), np.sin(hB)])
        
        xA = xA + dt * vA
        xB = xB + dt * vB
        
        xsA.append(xA.copy())
        xsB.append(xB.copy())
        
        prev_hA = hA
        prev_hB = hB
    
    return np.array(xsA), np.array(xsB)


def two_car_dmpc_with_cbf(
    xA0: np.ndarray,
    gA: np.ndarray,
    xB0: np.ndarray,
    gB: np.ndarray,
    *,
    dt: float = 0.1,
    H: int = 15,
    v_max_A: float = 1.4,
    v_max_B: float = 1.4,
    max_steps: int = 500,
    goal_tol: float = 0.2,
    heading_span_deg: float = 140.0,
    heading_samples: int = 25,
    speed_scales: Tuple[float, ...] = (0.0, 0.4, 0.7, 1.0),
    d_safe: float = 1.0,
    w_goal: float = 18.0,
    w_turn: float = 0.10,
    w_dist_A: float = 70.0,
    w_dist_B: float = 110.0,
    w_speed: float = 0.6,
    heading_smooth: float = 0.35,
    n_best_response_iters: int = 1,
    agent_safe_radius: float = 0.6,
    gamma_min: float = 6.0,
    w_p: float = 6.0,
    press_k: float = 1.0,
    press_n: float = 2.0,
    max_corr_norm: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DMPC with CBF execution safety shield.
    
    Same planning as pure DMPC, but applies CBF projection
    before executing velocities.
    """
    xA = np.asarray(xA0, dtype=np.float64).reshape(2).copy()
    xB = np.asarray(xB0, dtype=np.float64).reshape(2).copy()
    gA = np.asarray(gA, dtype=np.float64).reshape(2)
    gB = np.asarray(gB, dtype=np.float64).reshape(2)
    
    xsA = [xA.copy()]
    xsB = [xB.copy()]
    
    prev_hA = None
    prev_hB = None
    
    span = np.deg2rad(heading_span_deg)
    
    def predict_straight(x0, h, s, v_max):
        x = x0.copy()
        traj = [x.copy()]
        v = v_max * s * np.array([np.cos(h), np.sin(h)])
        for _ in range(H):
            x = x + dt * v
            traj.append(x.copy())
        return np.array(traj)
    
    def score_agent(x0, goal, other_pred, h, s, v_max, w_dist, prev_h):
        x = x0.copy()
        J = 0.0
        v = v_max * s * np.array([np.cos(h), np.sin(h)])
        
        for k in range(H):
            x = x + dt * v
            dist = np.linalg.norm(x - other_pred[min(k + 1, len(other_pred) - 1)])
            violation = max(0.0, d_safe - dist)
            J += w_dist * (violation ** 2)
        
        J += w_goal * np.sum((x - goal) ** 2)
        
        if prev_h is not None:
            dh = wrap_angle(h - prev_h)
            J += w_turn * (dh ** 2)
        
        J += w_speed * ((1.0 - s) ** 2)
        
        return J
    
    for step in range(max_steps):
        if np.linalg.norm(xA - gA) <= goal_tol and np.linalg.norm(xB - gB) <= goal_tol:
            break
        
        hA = np.arctan2((gA - xA)[1], (gA - xA)[0])
        hB = np.arctan2((gB - xB)[1], (gB - xB)[0])
        sA, sB = 1.0, 1.0
        
        for _ in range(n_best_response_iters):
            predA = predict_straight(xA, hA, sA, v_max_A)
            predB = predict_straight(xB, hB, sB, v_max_B)
            
            baseA = np.arctan2((gA - xA)[1], (gA - xA)[0])
            offsets = np.linspace(-span / 2, span / 2, heading_samples)
            bestJA, bestHA, bestSA = float('inf'), baseA, 1.0
            
            for off in offsets:
                cand_h = baseA + off
                for cand_s in speed_scales:
                    J = score_agent(xA, gA, predB, cand_h, cand_s, v_max_A, w_dist_A, prev_hA)
                    if J < bestJA:
                        bestJA, bestHA, bestSA = J, cand_h, cand_s
            hA, sA = bestHA, bestSA
            
            baseB = np.arctan2((gB - xB)[1], (gB - xB)[0])
            bestJB, bestHB, bestSB = float('inf'), baseB, 1.0
            
            for off in offsets:
                cand_h = baseB + off
                for cand_s in speed_scales:
                    J = score_agent(xB, gB, predA, cand_h, cand_s, v_max_B, w_dist_B, prev_hB)
                    if J < bestJB:
                        bestJB, bestHB, bestSB = J, cand_h, cand_s
            hB, sB = bestHB, bestSB
        
        if prev_hA is not None:
            alpha = heading_smooth
            hA = prev_hA + alpha * wrap_angle(hA - prev_hA)
        if prev_hB is not None:
            alpha = heading_smooth
            hB = prev_hB + alpha * wrap_angle(hB - prev_hB)
        
        # Compute nominal velocities
        vA_nom = v_max_A * sA * np.array([np.cos(hA), np.sin(hA)])
        vB_nom = v_max_B * sB * np.array([np.cos(hB), np.sin(hB)])
        
        # Apply CBF safety projection
        vA_safe = cbf_project_velocity_step(
            xA, vA_nom, xB,
            agent_safe_radius=agent_safe_radius,
            gamma_min=gamma_min, w_p=w_p,
            press_k=press_k, press_n=press_n,
            max_corr_norm=max_corr_norm,
        )
        vB_safe = cbf_project_velocity_step(
            xB, vB_nom, xA,
            agent_safe_radius=agent_safe_radius,
            gamma_min=gamma_min, w_p=w_p,
            press_k=press_k, press_n=press_n,
            max_corr_norm=max_corr_norm,
        )
        
        xA = xA + dt * vA_safe
        xB = xB + dt * vB_safe
        
        xsA.append(xA.copy())
        xsB.append(xB.copy())
        
        prev_hA = hA
        prev_hB = hB
    
    return np.array(xsA), np.array(xsB)


# Alias for backward compatibility
TwoCarDMPCController = None  # TODO: Implement full class version

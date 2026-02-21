"""
ODE Solvers for Flow Matching Trajectory Generation.

Provides integration of Flow Matching velocity fields with optional
CBF safety projection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union, Dict

# Try to import torchdiffeq
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False

from .cbf import cbf_project_velocity_world


class FMODEFunc:
    """
    Flow Matching ODE function with optional CBF projection.
    
    Computes: dx/dt = v_theta(x, t, cond) with optional CBF safety filter.
    """
    
    def __init__(
        self,
        model: nn.Module,
        traj_len: int,
        cond_vec: Optional[torch.Tensor] = None,
        ellipses: Optional[List[Dict]] = None,
        other_agents_traj: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        agent_safe_radius: float = 0.5,
        gamma_min: float = 1.0,
        w_p: float = 5.0,
        press_k: float = 1.0,
        press_n: float = 2.0,
        cbf_margin: float = 0.0,
        cbf_max_corr_norm: float = 2.0,
        cbf_passes: int = 5,
        use_cbf: bool = True,
    ):
        """
        Args:
            model: Flow Matching velocity model
            traj_len: Trajectory length
            cond_vec: Conditioning vector [cond_dim] or [1, cond_dim]
            ellipses: Static obstacle ellipses
            other_agents_traj: Other agent trajectories for CBF
            agent_safe_radius: Safe radius for inter-agent CBF
            gamma_min: Minimum CBF gain
            w_p: Pressure weight
            press_k: Pressure constant
            press_n: Pressure power
            cbf_margin: Additional CBF margin
            cbf_max_corr_norm: Maximum CBF correction norm
            cbf_passes: Number of CBF projection passes
            use_cbf: Whether to apply CBF projection
        """
        self.model = model
        self.traj_len = traj_len
        self.device = next(model.parameters()).device
        
        # Conditioning
        if cond_vec is not None:
            if cond_vec.dim() == 1:
                cond_vec = cond_vec.unsqueeze(0)
            self.cond_vec = cond_vec.to(self.device).float()
        else:
            self.cond_vec = None
        
        # CBF settings
        self.use_cbf = use_cbf
        self.ellipses = ellipses or []
        self.other_agents_traj = other_agents_traj
        self.agent_safe_radius = agent_safe_radius
        self.gamma_min = gamma_min
        self.w_p = w_p
        self.press_k = press_k
        self.press_n = press_n
        self.cbf_margin = cbf_margin
        self.cbf_max_corr_norm = cbf_max_corr_norm
        self.cbf_passes = cbf_passes
        
        # Pre-allocated time buffer
        self._t_buf = torch.zeros((1,), device=self.device, dtype=torch.float32)
    
    def __call__(self, t: float, x_flat: torch.Tensor) -> torch.Tensor:
        """
        ODE right-hand side function.
        
        Args:
            t: Current time (scalar)
            x_flat: Flattened state [2*T]
        
        Returns:
            Flattened velocity [2*T]
        """
        B, D, T = 1, 2, self.traj_len
        
        # Reshape to [1, 2, T]
        x = x_flat.reshape(B, D, T).to(self.device).float()
        
        # Fill time buffer
        if torch.is_tensor(t):
            self._t_buf[0] = t.to(self.device).float()
        else:
            self._t_buf[0] = float(t)
        
        # Get velocity from model
        with torch.no_grad():
            v_nom = self.model(x, self._t_buf, cond=self.cond_vec)
        
        # Apply CBF projection
        if self.use_cbf and (len(self.ellipses) > 0 or self.other_agents_traj is not None):
            v_safe = cbf_project_velocity_world(
                x_1x2N=x,
                v_nom_1x2N=v_nom,
                ellipses=self.ellipses,
                other_agents_traj=self.other_agents_traj,
                agent_safe_radius=self.agent_safe_radius,
                gamma_min=self.gamma_min,
                w_p=self.w_p,
                press_k=self.press_k,
                press_n=self.press_n,
                extra_margin=self.cbf_margin,
                max_corr_norm=self.cbf_max_corr_norm,
                passes=self.cbf_passes,
            )
            return v_safe.reshape(-1)
        
        return v_nom.reshape(-1)


class ODESolver:
    """
    Flow Matching ODE-based trajectory sampler.
    
    Integrates the learned velocity field from t=0 (noise) to t=1 (data)
    with optional CBF safety projection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        *,
        traj_len: int = 50,
        T_final: float = 1.0,
        solver_method: str = "euler",
        solver_tolerance: float = 1e-5,
        ode_steps: int = 50,
        # Conditioning
        cond_vec: Optional[torch.Tensor] = None,
        # CBF settings
        ellipses: Optional[List[Dict]] = None,
        cbf_passes: int = 5,
        cbf_margin: float = 0.0,
        cbf_max_corr_norm: float = 2.0,
        # Inter-agent
        other_agents_traj: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        agent_safe_radius: float = 0.5,
        # Pressure-based adaptive CBF
        gamma_min: float = 1.0,
        w_p: float = 5.0,
        press_k: float = 1.0,
        press_n: float = 2.0,
        use_cbf: bool = True,
    ):
        """
        Args:
            model: Flow Matching velocity model
            traj_len: Trajectory length
            T_final: Final integration time (usually 1.0)
            solver_method: ODE solver ('euler', 'dopri5', etc.)
            solver_tolerance: Solver tolerance for adaptive methods
            ode_steps: Number of integration steps
            cond_vec: Conditioning vector
            ellipses: Static obstacles
            cbf_passes: CBF projection passes
            cbf_margin: CBF safety margin
            cbf_max_corr_norm: Max CBF correction
            other_agents_traj: Other agent trajectories
            agent_safe_radius: Inter-agent safe radius
            gamma_min: Min CBF gain
            w_p: Pressure weight
            press_k: Pressure constant
            press_n: Pressure power
            use_cbf: Enable CBF projection
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.traj_len = traj_len
        self.T_final = T_final
        self.solver_method = solver_method
        self.solver_tolerance = solver_tolerance
        self.ode_steps = ode_steps
        
        # Create ODE function
        self.ode_func = FMODEFunc(
            model=model,
            traj_len=traj_len,
            cond_vec=cond_vec,
            ellipses=ellipses,
            other_agents_traj=other_agents_traj,
            agent_safe_radius=agent_safe_radius,
            gamma_min=gamma_min,
            w_p=w_p,
            press_k=press_k,
            press_n=press_n,
            cbf_margin=cbf_margin,
            cbf_max_corr_norm=cbf_max_corr_norm,
            cbf_passes=cbf_passes,
            use_cbf=use_cbf,
        )
    
    def update_other_agents(
        self,
        other_agents_traj: Optional[Union[List[torch.Tensor], torch.Tensor]]
    ):
        """Update other agent trajectories for CBF."""
        self.ode_func.other_agents_traj = other_agents_traj
    
    def update_condition(self, cond_vec: Optional[torch.Tensor]):
        """Update conditioning vector."""
        if cond_vec is not None:
            if cond_vec.dim() == 1:
                cond_vec = cond_vec.unsqueeze(0)
            self.ode_func.cond_vec = cond_vec.to(self.device).float()
        else:
            self.ode_func.cond_vec = None
    
    @torch.no_grad()
    def sample(
        self,
        x_init: Optional[torch.Tensor] = None,
        T_span: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample trajectory by integrating ODE.
        
        Args:
            x_init: Initial state [1, 2, T] (if None, uses Gaussian noise)
            T_span: Integration time points (if None, uses linspace)
            noise_scale: Scale for initial noise
        
        Returns:
            Final trajectory [1, 2, T]
        """
        T = self.traj_len
        
        # Initialize from noise if not provided
        if x_init is None:
            x_init = torch.randn(1, 2, T, device=self.device) * noise_scale
        else:
            x_init = x_init.to(self.device).float()
            assert x_init.shape == (1, 2, T), f"Expected (1, 2, {T}), got {tuple(x_init.shape)}"
        
        x0_flat = x_init.reshape(-1)
        
        # Create time span
        if T_span is None:
            T_span = torch.linspace(0.0, self.T_final, self.ode_steps + 1, device=self.device)
        else:
            T_span = T_span.to(self.device)
        
        # Integrate ODE
        if HAS_TORCHDIFFEQ and self.solver_method != "euler_manual":
            sol = torchdiffeq.odeint(
                self.ode_func,
                x0_flat,
                T_span,
                atol=self.solver_tolerance,
                rtol=self.solver_tolerance,
                method=self.solver_method,
            )
            x_T = sol[-1].reshape(1, 2, T)
        else:
            # Manual Euler integration
            x = x0_flat
            for i in range(len(T_span) - 1):
                t_i = T_span[i].item()
                dt = (T_span[i + 1] - T_span[i]).item()
                v = self.ode_func(t_i, x)
                x = x + dt * v
            x_T = x.reshape(1, 2, T)
        
        return x_T
    
    @torch.no_grad()
    def sample_velocity_sequence(
        self,
        cond: torch.Tensor,
        seq_len: int,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """
        Sample velocity sequence in canonical frame.
        
        Args:
            cond: Conditioning vector
            seq_len: Sequence length (usually same as traj_len)
            noise_scale: Initial noise scale
        
        Returns:
            Velocity sequence [seq_len, 2] as numpy array
        """
        # Update condition
        self.update_condition(cond)
        
        # Sample trajectory
        x_T = self.sample(noise_scale=noise_scale)
        
        # Return as velocity sequence
        v_seq = x_T.squeeze(0).transpose(0, 1).cpu().numpy()  # [T, 2]
        return v_seq[:seq_len]


def sample_v_seq_fm_odeint(
    model: nn.Module,
    cond: Union[np.ndarray, torch.Tensor],
    seq_len: int,
    T_SPAN: Optional[torch.Tensor] = None,
    solver_method: str = "euler",
    solver_tol: float = 1e-5,
    noise_scale: float = 1.0,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convenience function to sample velocity sequence using FM-ODE.
    
    Args:
        model: Flow Matching model
        cond: Conditioning vector
        seq_len: Output sequence length
        T_SPAN: Integration time points
        solver_method: ODE solver method
        solver_tol: Solver tolerance
        noise_scale: Initial noise scale
        device: Compute device
    
    Returns:
        Velocity sequence [seq_len, 2]
    """
    dev = torch.device(device)
    model_dev = next(model.parameters()).device
    
    # Convert condition
    if not torch.is_tensor(cond):
        cond = torch.tensor(cond, dtype=torch.float32)
    cond = cond.to(model_dev).float()
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    
    # Create time span
    if T_SPAN is None:
        T_SPAN = torch.linspace(0.0, 1.0, 51, device=model_dev)
    else:
        T_SPAN = T_SPAN.to(model_dev)
    
    # Initialize from noise
    x0 = torch.randn(1, 2, seq_len, device=model_dev) * noise_scale
    
    # ODE function
    t_buf = torch.zeros(1, device=model_dev)
    
    def fm_ode_func(t, x_flat):
        x = x_flat.view(1, 2, seq_len)
        if torch.is_tensor(t):
            t_buf[0] = t.item() if t.dim() == 0 else t[0].item()
        else:
            t_buf[0] = float(t)
        with torch.no_grad():
            v = model(x, t_buf, cond=cond)
        return v.view(-1)
    
    # Integrate
    if HAS_TORCHDIFFEQ:
        sol = torchdiffeq.odeint(
            fm_ode_func,
            x0.view(-1),
            T_SPAN,
            method=solver_method,
            atol=solver_tol,
            rtol=solver_tol,
        )
        v_seq = sol[-1].view(2, seq_len).T.cpu().numpy()
    else:
        # Euler fallback
        x = x0.view(-1)
        for i in range(len(T_SPAN) - 1):
            dt = (T_SPAN[i + 1] - T_SPAN[i]).item()
            v = fm_ode_func(T_SPAN[i], x)
            x = x + dt * v
        v_seq = x.view(2, seq_len).T.cpu().numpy()
    
    return v_seq.astype(np.float64)

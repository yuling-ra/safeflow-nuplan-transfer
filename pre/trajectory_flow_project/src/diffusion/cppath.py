import torch
from typing import Tuple, Optional
from .data import Sampleable
from .schedule import Alpha, Beta
from abc import ABC, abstractmethod

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        num_samples = t.shape[0]
        z, _ = self.sample_conditioning_variable(num_samples)
        x = self.sample_conditional_path(z, t)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ...

class TrajectoryConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_simple: Sampleable, p_data: Sampleable, alpha: Alpha, beta: Beta):
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, num_samples: int):
        z, y = self.p_data.sample(num_samples)
        return z, y
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise, _ = self.p_simple.sample(z.shape[0])
        alpha_t = self.alpha(t).view(-1, 1, 1)
        beta_t  = self.beta(t).view(-1, 1, 1)
        x_t = alpha_t * z + beta_t * noise
        return x_t
    
    @torch.no_grad()
    def conditional_vector_field(self, x_t: torch.Tensor, z: torch.Tensor, t: torch.Tensor, eps: float = 1e-5):
        alpha_t   = self.alpha(t).view(-1, 1, 1)
        beta_t    = self.beta(t).view(-1, 1, 1)
        alpha_dt  = self.alpha.dt(t).view(-1, 1, 1)
        beta_dt   = self.beta.dt(t).view(-1, 1, 1)
        beta_safe = torch.clamp(beta_t, min=eps)
        coeff_x = (beta_dt / beta_safe)
        coeff_z = alpha_dt - (beta_dt / beta_safe) * alpha_t
        return coeff_z * z + coeff_x * x_t

import torch
from fmtorch.solver import Solver
from fmtorch.diff_eq import SDE

class EulerMaruyamaSolver(Solver):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        # randn_like is used to generate a random tensor with the same shape as the input tensor, 
        # but with elements following a standard normal distribution (mean 0, standard deviation 1)
        return xt + self.sde.drift_coefficient(xt,t) * dt + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(dt) * torch.randn_like(xt)
    
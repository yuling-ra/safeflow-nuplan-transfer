import torch
from fmtorch.solver import Solver
from fmtorch.diff_eq import ODE

class EulerSolver(Solver):
    def __init__(self, ode: ODE):
        # Initialize the simulator with an ODE instance
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        # Perform one step of the Euler method for ODEs
        # xt: current state at time t
        # t: current time
        # dt: time step size
        return xt + dt * self.ode.drift_coefficient(xt, t)
    
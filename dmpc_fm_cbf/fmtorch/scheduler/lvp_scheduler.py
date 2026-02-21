import torch
from torch import Tensor

from fmtorch.scheduler import Scheduler, SchedulerOutput


class LinearVPScheduler(Scheduler):
    """Linear Variance Preserving Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        epsilon = 1e-4
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=(1 - t**2) ** 0.5,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-t / (torch.sqrt(1 - t**2 + epsilon))
        )

    def snr_inverse(self, snr: Tensor) -> Tensor:
        return torch.sqrt(snr**2 / (1 + snr**2))

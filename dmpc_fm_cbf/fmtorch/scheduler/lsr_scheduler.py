import torch
from torch import Tensor

from fmtorch.scheduler import Scheduler, SchedulerOutput


class LinearSquareRootScheduler(Scheduler):
    """Scheduler combining LinearAlpha (alpha_t = t) and SquareRootBeta (sigma_t = sqrt(1 - t))."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        """
        Args:
            t: Time tensor with shape (num_samples, 1).
        Returns:
            SchedulerOutput containing alpha_t, sigma_t, and their derivatives.
        """
        epsilon = 1e-4

        return SchedulerOutput(
            # alpha_t = t (from LinearAlpha)
            alpha_t=t,
            # beta_t = sqrt(1 - t) (from SquareRootBeta, renamed from beta_t)
            beta_t=torch.ones_like(t),
            # d_alpha_t = 1 (from LinearAlpha)
            d_alpha_t=torch.sqrt(1 - t),
            # d_beta_t = -0.5 / sqrt(1 - t) (from SquareRootBeta, renamed from d_beta_t)
            d_beta_t=-0.5 / (torch.sqrt(1 - t) + epsilon)
        )
    
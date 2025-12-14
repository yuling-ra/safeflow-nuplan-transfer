import torch
from torch import Tensor

from fmtorch.scheduler import ConvexScheduler, SchedulerOutput


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            beta_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_beta_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa

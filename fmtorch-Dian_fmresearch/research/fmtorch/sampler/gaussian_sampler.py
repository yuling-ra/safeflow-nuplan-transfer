import torch
import torch.nn as nn
import torch.distributions as D
from fmtorch.distribution import Density
from fmtorch.sampler import Sampler

class GaussianSampler(nn.Module, Density, Sampler):
    """
    Two-dimensional Gaussian. Is a Density and a Sampleable. Wrapper around torch.distributions.MultivariateNormal
    """
    def __init__(self, mean, cov):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    # @property 是 Python 的一个装饰器，它将一个 方法 变成一个 属性，
    # 这样我们可以像访问变量一样调用它，而不需要加 ()。
    @property
    def dim(self) -> int:
        return self.mean.shape[0]
    
    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))

    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)
    
    @classmethod
    def isotropic(cls, dim: int, std: float) -> "GaussianSampler":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)
    
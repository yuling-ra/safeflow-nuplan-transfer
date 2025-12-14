import torch
from abc import ABC, abstractmethod
from torch.func import vmap, jacrev

class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(self(torch.zeros(1,1,1)), torch.zeros(1,1,1))
        assert torch.allclose(self(torch.ones(1,1,1)), torch.ones(1,1,1))
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ...

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1)

class Beta(ABC):
    def __init__(self):
        assert torch.allclose(self(torch.zeros(1,1,1)), torch.ones(1,1,1))
        assert torch.allclose(self(torch.ones(1,1,1)), torch.zeros(1,1,1))
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        ...

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1)

class LinearAlpha(Alpha):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

class LinearBeta(Beta):
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

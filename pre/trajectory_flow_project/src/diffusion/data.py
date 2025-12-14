import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Sampleable(ABC):
    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ...

class SimpleTrajectoryDataset(Sampleable):
    def __init__(self, seq_len: int = 50, dim: int = 2, num_classes: int = 3):
        self.seq_len = seq_len
        self.dim = dim
        self.num_classes = num_classes
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        labels = torch.randint(0, self.num_classes, (num_samples,))
        trajectories = []
        for label in labels:
            label_val = int(label.item())
            t = np.linspace(0, 2*np.pi, self.seq_len)
            if label_val == 0:  # circle
                x = np.cos(t) * 0.5
                y = np.sin(t) * 0.5
            elif label_val == 1:  # line
                x = np.linspace(-0.5, 0.5, self.seq_len)
                y = x * 0.5
            else:  # ellipse
                x = np.cos(t) * 0.7
                y = np.sin(t) * 0.3
            traj = np.stack([x, y], axis=-1)
            trajectories.append(traj)
        trajectories = torch.tensor(np.array(trajectories), dtype=torch.float32, device=device)
        return trajectories, labels.to(device)

class TrajectoryNoise(Sampleable):
    def __init__(self, seq_len: int = 50, dim: int = 2):
        self.seq_len = seq_len
        self.dim = dim
    
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        trajectories = torch.randn(num_samples, self.seq_len, self.dim, device=device)
        return trajectories, None

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from .data import SimpleTrajectoryDataset, TrajectoryNoise
from .schedule import LinearAlpha, LinearBeta
from .cppath import TrajectoryConditionalProbabilityPath
from .model import SimpleTrajectoryUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self):
        idx = 0
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            self.shadow[idx].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            idx += 1

    def copy_to(self, model):
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[idx])
            idx += 1

def train_diffusion_model(num_epochs: int = 30000, batch_size: int = 128, lr: float = 3e-4, use_ema: bool = True):
    data = SimpleTrajectoryDataset(seq_len=50, num_classes=3)
    noise = TrajectoryNoise(seq_len=50, dim=2)
    alpha = LinearAlpha()
    beta = LinearBeta()
    cond_path = TrajectoryConditionalProbabilityPath(noise, data, alpha, beta)

    # visualize real data once (optional, can be commented out in headless env)
    # real_trajs, real_labels = data.sample(12)
    # from .viz import plot_trajectories
    # plot_trajectories(real_trajs.cpu(), real_labels.cpu(), "Real trajectories")

    unet = SimpleTrajectoryUNet(num_classes=3).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    ema = EMA(unet, decay=0.999) if use_ema else None
    losses = []

    pbar = tqdm(range(num_epochs), desc="Training")
    for _ in pbar:
        t = torch.rand(batch_size, 1, 1, device=device) * 0.98 + 0.01  # [0.01, 0.99]
        z, y = cond_path.sample_conditioning_variable(batch_size)
        x_t = cond_path.sample_conditional_path(z, t)

        target_v = cond_path.conditional_vector_field(x_t, z, t)
        pred_v = unet(x_t, t, y)

        loss = F.mse_loss(pred_v, target_v)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()

        if ema:
            ema.update()

        losses.append(float(loss.item()))
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return unet, cond_path, losses, ema

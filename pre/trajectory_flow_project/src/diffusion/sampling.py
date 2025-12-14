import torch
from .model import SimpleTrajectoryUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def generate_trajectories(unet: SimpleTrajectoryUNet, cond_path, num_samples: int = 12, num_steps: int = 200, use_heun: bool = True):
    unet.eval()
    x = torch.randn(num_samples, 50, 2, device=device)
    labels = torch.arange(3, device=device).repeat(num_samples // 3 + 1)[:num_samples]

    dt = 1.0 / num_steps
    for i in range(num_steps):
        # midpoint time, clamped to [0.01, 0.99]
        t_val = min(0.99, max(0.01, (i + 0.5) / num_steps))
        t = torch.full((num_samples, 1, 1), t_val, device=device)

        v_t = unet(x, t, labels)
        if use_heun:
            x_pred = x + v_t * dt
            t_next_val = min(0.99, max(0.01, (i + 1 + 0.5) / num_steps))
            t_next = torch.full((num_samples, 1, 1), t_next_val, device=device)
            v_t_pred = unet(x_pred, t_next, labels)
            x = x + 0.5 * (v_t + v_t_pred) * dt
        else:
            x = x + v_t * dt
    return x.detach().cpu(), labels.detach().cpu()

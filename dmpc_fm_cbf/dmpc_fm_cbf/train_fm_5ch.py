"""
5-Channel Flow Matching Model Training
=====================================

训练 FM 模型输出 5-channel trajectory: [x, y, theta, a, delta_rate]

数据格式:
- states: (N, T, 5) = [x, y, theta, v, delta]
- controls: (N, T, 2) = [a, delta_rate]

FM 模型输入/输出:
- x_t: (B, 5, T) = [x, y, theta, a, delta_rate]
- v_pred: (B, 5, T) = velocity field
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")


# ===============================================================
# 1) Dataset: 构建 5-channel 训练数据
# ===============================================================
class Trajectory5ChDataset(Dataset):
    """
    5-channel trajectory dataset for FM training.
    
    Channels: [x, y, theta, a, delta_rate]
    """
    
    def __init__(self, data_path: str, normalize: bool = True):
        """
        Args:
            data_path: Path to .npy file
            normalize: Whether to normalize data
        """
        # Load data
        raw = np.load(data_path, allow_pickle=True)
        if hasattr(raw, 'item'):
            data = raw.item()
        else:
            data = raw
        
        print(f"[Dataset] Loading from {data_path}")
        print(f"[Dataset] Keys: {list(data.keys())}")
        
        # Extract states and controls
        states = data['states']      # (N, T, 5) = [x, y, theta, v, delta]
        controls = data['controls']  # (N, T, 2) = [a, delta_rate]
        
        N, T, _ = states.shape
        print(f"[Dataset] N={N}, T={T}")
        
        # 构建 5-channel: [x, y, theta, a, delta_rate]
        x = states[:, :, 0]          # (N, T)
        y = states[:, :, 1]          # (N, T)
        theta = states[:, :, 2]      # (N, T)
        a = controls[:, :, 0]        # (N, T)
        delta_rate = controls[:, :, 1]  # (N, T)
        
        # Stack to (N, 5, T)
        traj_5ch = np.stack([x, y, theta, a, delta_rate], axis=1)
        
        print(f"[Dataset] traj_5ch shape: {traj_5ch.shape}")
        print(f"[Dataset] Channel ranges before normalization:")
        for i, name in enumerate(['x', 'y', 'theta', 'a', 'delta_rate']):
            ch = traj_5ch[:, i, :]
            print(f"  {name}: [{ch.min():.3f}, {ch.max():.3f}]")
        
        # Normalization (per-channel)
        self.normalize = normalize
        if normalize:
            self.mean = traj_5ch.mean(axis=(0, 2), keepdims=True)  # (1, 5, 1)
            self.std = traj_5ch.std(axis=(0, 2), keepdims=True) + 1e-6
            traj_5ch = (traj_5ch - self.mean) / self.std
            print(f"[Dataset] Normalized. mean={self.mean.flatten()}, std={self.std.flatten()}")
        else:
            self.mean = np.zeros((1, 5, 1), dtype=np.float32)
            self.std = np.ones((1, 5, 1), dtype=np.float32)
        
        self.trajs = torch.tensor(traj_5ch, dtype=torch.float32)
        self.N = N
        self.T = T
        
        # Optional: conditioning (e.g., goals)
        if 'goals' in data:
            goals = data['goals']  # (N, 2)
            # Normalize goals too
            if normalize:
                goals_norm = (goals - self.mean[0, :2, 0]) / self.std[0, :2, 0]
            else:
                goals_norm = goals
            self.cond = torch.tensor(goals_norm, dtype=torch.float32)
        else:
            self.cond = None
    
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        traj = self.trajs[idx]  # (5, T)
        if self.cond is not None:
            return traj, self.cond[idx]
        return traj, None
    
    def denormalize(self, x):
        """Convert normalized data back to original scale."""
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std, device=x.device, dtype=x.dtype)
        else:
            mean, std = self.mean, self.std
        return x * std + mean


# ===============================================================
# 2) Simple 1D UNet for 5-channel
# ===============================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x, t_emb):
        h = self.norm1(F.silu(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.norm2(F.silu(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet1D_5ch(nn.Module):
    """
    Simple 1D UNet for 5-channel trajectory.
    
    Input: (B, 5, T) + time t
    Output: (B, 5, T) velocity field
    """
    
    def __init__(
        self,
        in_channels: int = 5,
        model_channels: int = 64,
        time_emb_dim: int = 64,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )
        
        # Conditioning embedding
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )
        else:
            self.cond_mlp = None
        
        # Encoder
        ch = model_channels
        self.conv_in = nn.Conv1d(in_channels, ch, 3, padding=1)
        
        self.down1 = ResBlock1D(ch, ch, time_emb_dim)
        self.down2 = ResBlock1D(ch, ch * 2, time_emb_dim)
        self.pool = nn.AvgPool1d(2)
        
        # Middle
        self.mid = ResBlock1D(ch * 2, ch * 2, time_emb_dim)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBlock1D(ch * 4, ch, time_emb_dim)  # concat skip
        self.up_conv2 = ResBlock1D(ch * 2, ch, time_emb_dim)
        
        self.conv_out = nn.Conv1d(ch, in_channels, 3, padding=1)
    
    def forward(self, x, t, cond=None):
        """
        Args:
            x: (B, 5, T) noisy trajectory
            t: (B,) time
            cond: (B, cond_dim) optional conditioning
        
        Returns:
            (B, 5, T) predicted velocity
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Add conditioning
        if self.cond_mlp is not None and cond is not None:
            t_emb = t_emb + self.cond_mlp(cond)
        
        # Encoder
        h = self.conv_in(x)
        h1 = self.down1(h, t_emb)
        h = self.pool(h1)
        h2 = self.down2(h, t_emb)
        h = self.pool(h2)
        
        # Middle
        h = self.mid(h, t_emb)
        
        # Decoder
        h = self.up1(h)
        # Handle size mismatch
        if h.shape[2] != h2.shape[2]:
            h = F.interpolate(h, size=h2.shape[2], mode='nearest')
        h = torch.cat([h, h2], dim=1)
        h = self.up_conv1(h, t_emb)
        
        h = self.up1(h)
        if h.shape[2] != h1.shape[2]:
            h = F.interpolate(h, size=h1.shape[2], mode='nearest')
        h = torch.cat([h, h1], dim=1)
        h = self.up_conv2(h, t_emb)
        
        return self.conv_out(h)


# ===============================================================
# 3) Flow Matching Trainer
# ===============================================================
class FMTrainer5ch:
    """Flow Matching trainer for 5-channel model."""
    
    def __init__(self, model, lr=1e-4, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.history = []
    
    def train_step(self, x_1, cond=None):
        """
        Single training step.
        
        x_1: (B, 5, T) target trajectory
        """
        B = x_1.shape[0]
        
        # Sample noise (source)
        x_0 = torch.randn_like(x_1)
        
        # Sample time uniformly
        t = torch.rand(B, device=self.device)
        
        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Target velocity: dx/dt = x_1 - x_0
        v_target = x_1 - x_0
        
        # Predict velocity
        v_pred = self.model(x_t, t, cond=cond)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            x_1, cond = batch
            x_1 = x_1.to(self.device)
            if cond is not None:
                cond = cond.to(self.device)
            
            self.optimizer.zero_grad()
            loss = self.train_step(x_1, cond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(self, dataset, epochs=100, batch_size=32, log_every=10):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        print(f"[Training] {epochs} epochs, batch_size={batch_size}")
        print(f"[Training] {len(dataset)} samples, {len(dataloader)} batches/epoch")
        
        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            self.history.append(loss)
            
            if (epoch + 1) % log_every == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss = {loss:.6f}")
        
        return self.history


# ===============================================================
# 4) Main Training Script
# ===============================================================
if __name__ == "__main__":
    # 1. Load dataset
    data_path = "car_ring_track_oc_dataset_v1_v0rand_with_vel.npy"
    
    # 检查文件是否存在
    if not Path(data_path).exists():
        # 尝试在 uploads 目录找
        uploads_path = Path("/mnt/user-data/uploads") / data_path
        if uploads_path.exists():
            data_path = str(uploads_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    dataset = Trajectory5ChDataset(data_path, normalize=True)
    
    # 2. Create model
    model = SimpleUNet1D_5ch(
        in_channels=5,
        model_channels=64,
        time_emb_dim=64,
        cond_dim=0,  # 设为 2 如果用 goals 做 conditioning
    )
    
    print(f"\n[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train
    trainer = FMTrainer5ch(model, lr=1e-4, device=str(device))
    history = trainer.train(dataset, epochs=200, batch_size=32, log_every=20)
    
    # 4. Save model
    save_path = "fm_5ch_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        'T': dataset.T,
    }, save_path)
    print(f"\n[Saved] {save_path}")
    
    # 5. Plot training curve
    plt.figure(figsize=(10, 4))
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('5-Channel FM Training')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()
    print("[Saved] training_curve.png")

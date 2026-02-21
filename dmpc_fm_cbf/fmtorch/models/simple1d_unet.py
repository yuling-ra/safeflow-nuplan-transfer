
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def forward(self, x):
        return x * th.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t: th.Tensor) -> th.Tensor:
        """
        t: (B,) or (B,1) or scalar
        returns: (B, dim) time embedding
        """
        device = t.device
        half_dim = self.dim // 2
        emb = th.log(th.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)

        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B,1)

        emb = t * emb[None, :]  # (B, half_dim)
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, channels),
        )
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            Mish(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            Mish(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: th.Tensor, t_emb: th.Tensor) -> th.Tensor:
        """
        x: (B, C, T)
        t_emb: (B, time_emb_dim)
        """
        h = self.block(x)
        # expand when t_emb has batch=1 but x has larger batch
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        h = h + self.time_mlp(t_emb)[:, :, None]
        return x + h


class SimpleUNet1D(nn.Module):
    def __init__(
        self,
        time_emb_dim: int = 128,
        hidden_dim: int = 128,
        cond_dim: int = 0,
        in_channels: int = 2,
        out_channels: int = 2,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim * 4),
                Mish(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )
        else:
            self.cond_mlp = None

        self.conv_in = nn.Conv1d(in_channels, hidden_dim, 3, padding=1)

        self.down1_block = nn.ModuleList(
            [
                ResBlock1D(hidden_dim, time_emb_dim),
                ResBlock1D(hidden_dim, time_emb_dim),
            ]
        )
        self.down1_sample = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)

        self.down2_block = nn.ModuleList(
            [
                ResBlock1D(hidden_dim, time_emb_dim),
                ResBlock1D(hidden_dim, time_emb_dim),
            ]
        )
        self.down2_sample = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)

        self.mid_block = nn.ModuleList(
            [
                ResBlock1D(hidden_dim, time_emb_dim),
                ResBlock1D(hidden_dim, time_emb_dim),
            ]
        )

        self.up2_sample = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.up2_block = nn.ModuleList(
            [
                ResBlock1D(hidden_dim * 2, time_emb_dim),
                ResBlock1D(hidden_dim * 2, time_emb_dim),
            ]
        )
        self.up2_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)

        self.up1_sample = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.up1_block = nn.ModuleList(
            [
                ResBlock1D(hidden_dim * 2, time_emb_dim),
                ResBlock1D(hidden_dim * 2, time_emb_dim),
            ]
        )
        self.up1_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            Mish(),
            nn.Conv1d(hidden_dim, out_channels, 3, padding=1),
        )

    def forward(self, x: th.Tensor, t: th.Tensor, cond: th.Tensor = None) -> th.Tensor:
        """
        x: (B, 2, T)
        t: (B,) or (B,1)
        cond: (B, cond_dim)
        """
        t_emb = self.time_mlp(t)

        if self.cond_mlp is not None and cond is not None:
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            if cond.shape[0] != x.shape[0]:
                cond = cond.expand(x.shape[0], -1)
            cond_emb = self.cond_mlp(cond)
            t_emb = t_emb + cond_emb

        h = self.conv_in(x)

        h1 = h
        for block in self.down1_block:
            h1 = block(h1, t_emb)
        h1_pooled = self.down1_sample(h1)

        h2 = h1_pooled
        for block in self.down2_block:
            h2 = block(h2, t_emb)
        h2_pooled = self.down2_sample(h2)

        h_mid = h2_pooled
        for block in self.mid_block:
            h_mid = block(h_mid, t_emb)

        h_up2 = self.up2_sample(h_mid)
        if h_up2.shape[2] != h2.shape[2]:
            min_len = min(h_up2.shape[2], h2.shape[2])
            h_up2 = h_up2[:, :, :min_len]
            h2_aligned = h2[:, :, :min_len]
        else:
            h2_aligned = h2

        h = th.cat([h_up2, h2_aligned], dim=1)
        for block in self.up2_block:
            h = block(h, t_emb)
        h = self.up2_conv(h)

        h_up1 = self.up1_sample(h)
        if h_up1.shape[2] != h1.shape[2]:
            min_len = min(h_up1.shape[2], h1.shape[2])
            h_up1 = h_up1[:, :, :min_len]
            h1_aligned = h1[:, :, :min_len]
        else:
            h1_aligned = h1

        h = th.cat([h_up1, h1_aligned], dim=1)
        for block in self.up1_block:
            h = block(h, t_emb)
        h = self.up1_conv(h)

        return self.conv_out(h)


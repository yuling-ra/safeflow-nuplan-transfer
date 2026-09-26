"""
Neural Network Models for Flow Matching.

Provides:
- SimpleUNet1D: 1D U-Net for trajectory/velocity sequence prediction
- VelocityMLP: Simple MLP for velocity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for time conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [B] or [B, 1] or scalar
        
        Returns:
            Time embeddings [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        emb = t * emb[None, :]  # [B, half_dim]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]
        return emb


class ResBlock1D(nn.Module):
    """Residual block for 1D convolutions with time conditioning."""
    
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
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, T]
            t_emb: Time embedding [B, time_emb_dim]
        
        Returns:
            Output [B, C, T]
        """
        h = self.block(x)
        # Expand t_emb if batch sizes don't match
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)
        h = h + self.time_mlp(t_emb)[:, :, None]
        return x + h


class SimpleUNet1D(nn.Module):
    """
    Simple 1D U-Net for sequence-to-sequence prediction.
    
    Used for Flow Matching velocity field prediction over trajectories.
    
    Architecture:
        - 2 downsampling blocks with residual connections
        - 1 middle block
        - 2 upsampling blocks with skip connections
        - Time conditioning via sinusoidal embeddings + MLP
        - Optional condition vector (e.g., goal distance, obstacle position)
    """
    
    def __init__(
        self,
        time_emb_dim: int = 128,
        hidden_dim: int = 128,
        cond_dim: int = 0,
        in_channels: int = 2,
        out_channels: int = 2,
    ):
        """
        Args:
            time_emb_dim: Dimension of time embedding
            hidden_dim: Hidden dimension for convolutions
            cond_dim: Dimension of conditioning vector (0 for unconditional)
            in_channels: Input channels (2 for xy)
            out_channels: Output channels (2 for xy velocity)
        """
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            Mish(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Condition embedding
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim * 4),
                Mish(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )
        else:
            self.cond_mlp = None
        
        # Input projection
        self.conv_in = nn.Conv1d(in_channels, hidden_dim, 3, padding=1)
        
        # Encoder
        self.down1_block = nn.ModuleList([
            ResBlock1D(hidden_dim, time_emb_dim),
            ResBlock1D(hidden_dim, time_emb_dim),
        ])
        self.down1_sample = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        
        self.down2_block = nn.ModuleList([
            ResBlock1D(hidden_dim, time_emb_dim),
            ResBlock1D(hidden_dim, time_emb_dim),
        ])
        self.down2_sample = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        
        # Middle
        self.mid_block = nn.ModuleList([
            ResBlock1D(hidden_dim, time_emb_dim),
            ResBlock1D(hidden_dim, time_emb_dim),
        ])
        
        # Decoder
        self.up2_sample = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.up2_block = nn.ModuleList([
            ResBlock1D(hidden_dim * 2, time_emb_dim),
            ResBlock1D(hidden_dim * 2, time_emb_dim),
        ])
        self.up2_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        
        self.up1_sample = nn.ConvTranspose1d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.up1_block = nn.ModuleList([
            ResBlock1D(hidden_dim * 2, time_emb_dim),
            ResBlock1D(hidden_dim * 2, time_emb_dim),
        ])
        self.up1_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            Mish(),
            nn.Conv1d(hidden_dim, out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequence [B, 2, T]
            t: Time [B] or [B, 1]
            cond: Conditioning vector [B, cond_dim] (optional)
        
        Returns:
            Predicted velocity field [B, 2, T]
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Add condition embedding
        if self.cond_mlp is not None and cond is not None:
            if cond.dim() == 1:
                cond = cond.unsqueeze(0)
            if cond.shape[0] != x.shape[0]:
                cond = cond.expand(x.shape[0], -1)
            cond_emb = self.cond_mlp(cond)
            t_emb = t_emb + cond_emb
        
        # Input
        h = self.conv_in(x)
        
        # Encoder
        h1 = h
        for block in self.down1_block:
            h1 = block(h1, t_emb)
        h1_pooled = self.down1_sample(h1)
        
        h2 = h1_pooled
        for block in self.down2_block:
            h2 = block(h2, t_emb)
        h2_pooled = self.down2_sample(h2)
        
        # Middle
        h_mid = h2_pooled
        for block in self.mid_block:
            h_mid = block(h_mid, t_emb)
        
        # Decoder with skip connections
        h_up2 = self.up2_sample(h_mid)
        # Handle size mismatch from pooling
        if h_up2.shape[2] != h2.shape[2]:
            min_len = min(h_up2.shape[2], h2.shape[2])
            h_up2 = h_up2[:, :, :min_len]
            h2_aligned = h2[:, :, :min_len]
        else:
            h2_aligned = h2
        
        h = torch.cat([h_up2, h2_aligned], dim=1)
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
        
        h = torch.cat([h_up1, h1_aligned], dim=1)
        for block in self.up1_block:
            h = block(h, t_emb)
        h = self.up1_conv(h)
        
        return self.conv_out(h)


class VelocityMLP(nn.Module):
    """
    Simple MLP for velocity prediction.
    
    Used when sequence structure is not needed.
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        cond_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        time_emb_dim: int = 64,
    ):
        """
        Args:
            state_dim: State dimension (2 for xy)
            cond_dim: Conditioning dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        
        input_dim = state_dim + cond_dim + time_emb_dim
        
        layers = [nn.Linear(input_dim, hidden_dim), Mish()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), Mish()])
        layers.append(nn.Linear(hidden_dim, state_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State [B, state_dim]
            t: Time [B] or [B, 1]
            cond: Conditioning [B, cond_dim]
        
        Returns:
            Velocity [B, state_dim]
        """
        t_emb = self.time_emb(t)
        
        if cond is not None:
            inp = torch.cat([x, cond, t_emb], dim=-1)
        else:
            inp = torch.cat([x, t_emb], dim=-1)
        
        return self.net(inp)


def infer_cond_dim(model: nn.Module) -> Optional[int]:
    """
    Infer the conditioning dimension from a model.
    
    Args:
        model: Neural network model
    
    Returns:
        Conditioning dimension or None if unconditional
    """
    if hasattr(model, "cond_dim") and model.cond_dim is not None:
        return int(model.cond_dim)
    if hasattr(model, "cond_mlp") and model.cond_mlp is not None:
        for layer in model.cond_mlp:
            if hasattr(layer, "in_features"):
                return int(layer.in_features)
    return None

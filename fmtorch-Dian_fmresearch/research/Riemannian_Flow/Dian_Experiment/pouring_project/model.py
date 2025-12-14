import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fmtorch.utils import ModelWrapper

# ==============================================================================
# Helper Functions & Basic Modules
# ==============================================================================

def fourier_time_embed(t, K=32):
    """Fourier time embedding"""
    device = t.device
    k = torch.arange(K, device=device, dtype=torch.float32)
    freqs = 2**k * math.pi
    ang = t * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

def sinusoidal_pe(seq_len, hidden_dim):
    """Sinusoidal positional encoding for (B, L, C) format"""
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim))
    pe = torch.zeros(1, seq_len, hidden_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ResidualFC(nn.Module):
    """全连接残差块（增强版）"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = Mish()
    
    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.act(out + residual)

# ==============================================================================
# CBAM Attention Modules for Sequences
# ==============================================================================

class SequenceChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        y = self.fc(y).unsqueeze(1)
        return x * y.expand_as(x)

class SequenceSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_t = x.transpose(1, 2)
        avg_out = torch.mean(x_t, dim=1, keepdim=True)
        max_out, _ = torch.max(x_t, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y.transpose(1, 2).expand_as(x)

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = SequenceChannelAttention(channels, reduction)
        self.spatial_attention = SequenceSpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ==============================================================================
# Windowed Transformer Encoder
# ==============================================================================

class WindowedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0, window_size=96, bridge_k=8, bridge_q=8):
        super().__init__()
        self.window_size = window_size
        self.bridge_k = bridge_k
        self.bridge_q = bridge_q
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.bridge_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, src):
        x = src
        x = x + self.dropout1(self._sa_block(self.norm1(x)))
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x

    def _sa_block(self, src):
        B, L, C = src.shape
        w = self.window_size
        pad_len = (w - L % w) % w
        
        padded_src = src
        if pad_len > 0:
            padded_src = F.pad(src, (0, 0, 0, pad_len))
        
        Lp = padded_src.shape[1]
        WN = Lp // w

        x_win = padded_src.view(B, WN, w, C)

        # --- Main window MHA ---
        core = x_win.reshape(B * WN, w, C)
        core, _ = self.self_attn(core, core, core, need_weights=False)
        core = core.view(B, WN, w, C)

        # --- Bridge Attention ---
        if self.bridge_k > 0 and self.bridge_q > 0 and WN > 1:
            prev_tail = x_win[:, :-1, -self.bridge_k:, :].reshape(-1, self.bridge_k, C)
            curr_head = core[:, 1:, :self.bridge_q, :].reshape(-1, self.bridge_q, C)

            bridged, _ = self.bridge_attn(curr_head, prev_tail, prev_tail, need_weights=False)
            bridged = bridged.view(B, WN - 1, self.bridge_q, C)

            core[:, 1:, :self.bridge_q, :] = core[:, 1:, :self.bridge_q, :] + bridged
        
        out = core.reshape(B, Lp, C)
        if pad_len > 0:
            out = out[:, :L, :]
        return out

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

# ==============================================================================
# Main Vector Field Network
# ==============================================================================

class EnhancedVectorFieldNet(ModelWrapper):
    def __init__(self, seq_len=479, data_dim=6, hidden_dim=512, num_layers=8, 
                 time_embed_dim=256, dropout=0.0, num_heads=8, reduction=16, window=96,
                 fourier_k=32, bridge_k=8, bridge_q=8, cond_dim=0):
        super(ModelWrapper, self).__init__() # Call nn.Module's init
        
        self.seq_len = seq_len
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        
        # Time and position embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(fourier_k * 2, time_embed_dim), Mish(),
            nn.Linear(time_embed_dim, time_embed_dim), Mish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.register_buffer("pos_emb", sinusoidal_pe(seq_len, hidden_dim), persistent=False)
        self.fourier_k = fourier_k

        # Projections
        self.input_proj = nn.Sequential(
            nn.Linear(data_dim, hidden_dim), nn.LayerNorm(hidden_dim), Mish()
        )
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)

        if self.cond_dim > 0:
            self.cond_proj = nn.Linear(self.cond_dim, hidden_dim)
        
        # Core blocks
        self.cbam_blocks = nn.ModuleList([
            CBAM1D(channels=hidden_dim, reduction=reduction) for _ in range(num_layers // 2)
        ])
        self.transformer_layers = nn.ModuleList([
            WindowedTransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 2, dropout=dropout, window_size=window,
                bridge_k=bridge_k, bridge_q=bridge_q
            ) for _ in range(num_layers)
        ])
        self.fc_blocks = nn.ModuleList([
            ResidualFC(hidden_dim, dropout=dropout) for _ in range(4)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            Mish(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, data_dim)
        )
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EnhancedVectorFieldNet Initialized: {total_params:,} trainable parameters.")

    def forward(self, x, t, condition=None, **kwargs):
        B, L, C = x.shape
        
        t_feat = fourier_time_embed(t.view(B, 1).float(), K=self.fourier_k)
        t_emb = self.time_embed(t_feat)
        
        h = self.input_proj(x) + self.time_proj(t_emb).unsqueeze(1)
        
        if condition is not None and self.cond_dim > 0:
            cond_emb = self.cond_proj(condition)
            h = h + cond_emb.unsqueeze(1)

        h = h + self.pos_emb[:, :L, :]
        
        for cbam in self.cbam_blocks:
            h = cbam(h)
        
        for layer in self.transformer_layers:
            h = layer(h)
        
        h_flat = h.reshape(B * L, -1)
        for fc_block in self.fc_blocks:
            h_flat = fc_block(h_flat)
        h = h_flat.reshape(B, L, -1)
        
        return self.output_proj(h)

from typing import List
import torch
import torch.nn as nn
from .time_embed import FourierEncoder

class SimpleTrajectoryUNet(nn.Module):
    def __init__(self, 
                 seq_len: int = 50,
                 input_dim: int = 2,
                 hidden_dims: List[int] = [64, 128],
                 t_embed_dim: int = 64,
                 y_embed_dim: int = 32,
                 num_classes: int = 3): 
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.y_embedder = nn.Embedding(num_classes, y_embed_dim)
        self.time_proj = nn.Linear(t_embed_dim, hidden_dims[0])
        self.y_proj = nn.Linear(y_embed_dim, hidden_dims[0])

        self.encoder_convs = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.encoder_convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i], kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dims[i], hidden_dims[i], kernel_size=3, padding=1),
                    nn.SiLU()
                )
            )
            self.downsample_layers.append(
                nn.Conv1d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1)
            )
        
        self.mid_conv = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.upsample_layers = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(hidden_dims[i], hidden_dims[i-1], kernel_size=3, padding=1)
                )
            )
            self.decoder_convs.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i-1], hidden_dims[i-1], kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dims[i-1], hidden_dims[i-1], kernel_size=3, padding=1),
                    nn.SiLU()
                )
            )
        
        self.output_proj = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bs, seq_len, input_dim = x.shape

        if t.dim() == 4:
            t = t.squeeze(-1)
        
        t_embed = self.time_embedder(t)
        y_embed = self.y_embedder(y)
        
        t_proj = self.time_proj(t_embed).unsqueeze(-1)
        y_proj = self.y_proj(y_embed).unsqueeze(-1)
        
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = x + t_proj + y_proj
        
        skip_connections = []
        for conv, downsample in zip(self.encoder_convs, self.downsample_layers):
            x = conv(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_conv(x)

        for upsample, conv in zip(self.upsample_layers, self.decoder_convs):
            x = upsample(x)
            skip = skip_connections.pop()
            x = x + skip
            x = conv(x)

        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x

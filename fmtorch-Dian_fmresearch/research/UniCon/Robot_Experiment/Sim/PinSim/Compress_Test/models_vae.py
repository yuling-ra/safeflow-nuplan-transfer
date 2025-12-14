#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
from typing import Tuple

class MLPVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(d, latent_dim)
        self.logvar = nn.Linear(d, latent_dim)

        # decoder
        dlayers = []
        d = latent_dim
        for i in range(num_layers):
            dlayers += [nn.Linear(d, hidden_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                dlayers += [nn.Dropout(dropout)]
            d = hidden_dim
        dlayers += [nn.Linear(d, in_dim)]
        self.decoder = nn.Sequential(*dlayers)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1e-3):
        # MSE 重建 + beta * KL
        recon_loss = torch.mean((recon_x - x) ** 2)
        # KL(N(mu, sigma) || N(0,1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss.detach(), kl.detach() 
# models/simple_autoencoder.py
from __future__ import annotations

import torch
import torch.nn as nn
from models.base_autoencoder import BaseAutoencoder

class SimpleAutoencoder(BaseAutoencoder):
    """Plain feed‑forward autoencoder (AE).

    Encoder/decoder are MLPs with a single hidden layer. Use MSE loss to
    reconstruct the input (x -> x').
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__(input_dim, latent_dim)
        # Encoder: D -> H -> Z
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder: Z -> H -> D
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

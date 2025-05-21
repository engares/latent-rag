# /models/denoising_autoencoder.py

import torch
import torch.nn as nn
from models.base_autoencoder import BaseAutoencoder
import torch.nn.functional as F

class DenoisingAutoencoder(BaseAutoencoder):
    """Feed‑forward Denoising Autoencoder.

    The dataset must supply *noisy* inputs; the model learns to reconstruct the
    clean version. Use `dae_loss` (MSE) during training.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super().__init__(input_dim, latent_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # assume inputs ∈ [0,1]; change if different scale
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

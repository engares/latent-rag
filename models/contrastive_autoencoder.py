import torch
import torch.nn as nn
from models.base_autoencoder import BaseAutoencoder

class ContrastiveAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super(ContrastiveAutoencoder, self).__init__(input_dim, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Useful if input vectors are normalized
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return torch.nn.functional.normalize(z, p=2, dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
      
    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

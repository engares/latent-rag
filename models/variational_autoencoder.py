# / models/variational_autoencoder.py
import torch
import torch.nn as nn
from models.base_autoencoder import BaseAutoencoder

class VariationalAutoencoder(BaseAutoencoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 512):
        super(VariationalAutoencoder, self).__init__(input_dim, latent_dim)
        
        # Encoder: proyecciones a la media y desviación estándar
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            # nn.Sigmoid()  # Asumimos entrada normalizada (0-1)
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu if not self.training else self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

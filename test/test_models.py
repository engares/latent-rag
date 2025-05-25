import pytest
import torch
from models.variational_autoencoder import VariationalAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
from models.contrastive_autoencoder import ContrastiveAutoencoder

# ------------------- Variational Autoencoder -------------------
def test_variational_autoencoder():
    model = VariationalAutoencoder(input_dim=16, latent_dim=4, hidden_dim=8)
    x = torch.randn(2, 16)
    x_reconstructed, mu, logvar = model(x)

    assert x_reconstructed.shape == x.shape, "Reconstructed output shape mismatch"
    assert mu.shape == (2, 4), "Latent mean shape mismatch"
    assert logvar.shape == (2, 4), "Latent logvar shape mismatch"

# ------------------- Denoising Autoencoder -------------------
def test_denoising_autoencoder():
    model = DenoisingAutoencoder(input_dim=16, latent_dim=4, hidden_dim=8)
    x_noisy = torch.randn(2, 16)
    x_reconstructed = model(x_noisy)

    assert x_reconstructed.shape == x_noisy.shape, "Reconstructed output shape mismatch"

# ------------------- Contrastive Autoencoder -------------------
def test_contrastive_autoencoder():
    model = ContrastiveAutoencoder(input_dim=16, latent_dim=4, hidden_dim=8)
    x = torch.randn(2, 16)
    x_reconstructed, z = model(x)

    assert x_reconstructed.shape == x.shape, "Reconstructed output shape mismatch"
    assert z.shape == (2, 4), "Latent embedding shape mismatch"

    # Test normalized embeddings
    z_norm = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(z_norm, torch.ones_like(z_norm), atol=1e-6), "Latent embeddings are not normalized"

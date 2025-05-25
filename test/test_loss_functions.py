import torch
import pytest
from training.loss_functions import vae_loss, dae_loss, contrastive_loss

# ------------------- VAE Loss -------------------
def test_vae_loss_basic():
    x_reconstructed = torch.randn(8, 16)
    x_target = torch.randn(8, 16)
    mu = torch.zeros(8, 4)
    logvar = torch.zeros(8, 4)
    loss = vae_loss(x_reconstructed, x_target, mu, logvar)
    assert loss.shape == (), "VAE loss should be a scalar"
    assert loss.item() >= 0

def test_vae_loss_beta():
    x_reconstructed = torch.randn(4, 10)
    x_target = torch.randn(4, 10)
    mu = torch.randn(4, 3)
    logvar = torch.randn(4, 3)
    loss1 = vae_loss(x_reconstructed, x_target, mu, logvar, beta=0.5)
    loss2 = vae_loss(x_reconstructed, x_target, mu, logvar, beta=2.0)
    assert loss1 != loss2

# ------------------- DAE Loss -------------------
def test_dae_loss_basic():
    x_reconstructed = torch.randn(5, 7)
    x_clean = torch.randn(5, 7)
    loss = dae_loss(x_reconstructed, x_clean)
    assert loss.shape == (), "DAE loss should be a scalar"
    assert loss.item() >= 0

# ------------------- Contrastive Loss -------------------
def test_contrastive_loss_hard_negatives():
    z_q = torch.randn(6, 8)
    z_pos = torch.randn(6, 8)
    loss = contrastive_loss(z_q, z_pos, hard_negatives=True)
    assert loss.shape == (), "Contrastive loss should be a scalar"
    assert loss.item() >= 0

def test_contrastive_loss_random_negatives():
    z_q = torch.randn(6, 8)
    z_pos = torch.randn(6, 8)
    loss = contrastive_loss(z_q, z_pos, hard_negatives=False)
    assert loss.shape == (), "Contrastive loss should be a scalar"
    assert loss.item() >= 0

def test_contrastive_loss_margin_effect():
    z_q = torch.randn(4, 5)
    z_pos = torch.randn(4, 5)
    loss1 = contrastive_loss(z_q, z_pos, margin=0.1)
    loss2 = contrastive_loss(z_q, z_pos, margin=1.0)
    assert loss1 != loss2

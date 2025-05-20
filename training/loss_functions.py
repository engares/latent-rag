import torch
import torch.nn.functional as F


def vae_loss(x_reconstructed: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE loss: BCE + KL divergence."""
    recon_loss = F.binary_cross_entropy(x_reconstructed, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def dae_loss(x_reconstructed: torch.Tensor, x_clean: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Mean‑squared error reconstruction loss for Denoising Autoencoders."""
    return F.mse_loss(x_reconstructed, x_clean, reduction=reduction)

def contrastive_loss(z_q: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Contrastive loss: encourages z_q to be closer to z_pos than to z_neg by at least the margin."""
    pos_dist = torch.norm(z_q - z_pos, dim=1)
    neg_dist = torch.norm(z_q - z_neg, dim=1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

# /training/loss_functions.py

import torch
import torch.nn.functional as F

###############################################################################
#  VAE                                                                        #
###############################################################################

import torch
import torch.nn.functional as F

def vae_loss(
    x_reconstructed: torch.Tensor,
    x_target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    mse_reduction: str = "mean",   # "mean" or "sum"
    beta: float = 1.0,             # β-VAE (β=1 → classic VAE)
) -> torch.Tensor:
    """VAE loss = reconstruction + β·KL  (KL normalized by batch).

    Args:
        x_reconstructed: output from the decoder  ― shape [B, D]
        x_target:        original embeddings ― shape [B, D]
        mu, logvar:      parameters of the latent distribution ― shape [B, Z]
        mse_reduction:   "mean" (recommended) or "sum"
        beta:            weight of the KL term (β-VAE)
    """
    cos = F.cosine_similarity(x_reconstructed, x_target, dim=-1)
    recon = (1.0 - cos).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl


###############################################################################
#  DAE                                                                        #
###############################################################################

def dae_loss(
    x_reconstructed: torch.Tensor,
    x_clean: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean‑squared error for Denoising Auto‑Encoders."""
    return F.mse_loss(x_reconstructed, x_clean, reduction=reduction)

###############################################################################
#  CONTRASTIVE                                                                #
###############################################################################

def contrastive_loss(
    z_q: torch.Tensor,
    z_pos: torch.Tensor,
    *,
    margin: float = 0.2,
    hard_negatives: bool = True,
) -> torch.Tensor:
    """Triplet loss with negative selection within the batch.

    If `hard_negatives` is True, uses the closest negative; otherwise,
    permutes `z_pos` to obtain a random negative.
    """
    z_q = F.normalize(z_q, p=2, dim=1)
    z_pos = F.normalize(z_pos, p=2, dim=1)

    if hard_negatives:
        dist_mat = torch.cdist(z_q, z_pos, p=2)
        mask = torch.eye(dist_mat.size(0), dtype=torch.bool, device=z_q.device)
        dist_mat = dist_mat.masked_fill(mask, float("inf"))  # ← corrected
        neg_dist, _ = dist_mat.min(dim=1)

    else:
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)
        neg_dist = torch.norm(z_q - z_pos[idx], dim=1)

    pos_dist = torch.norm(z_q - z_pos, dim=1)
    return F.relu(pos_dist - neg_dist + margin).mean()

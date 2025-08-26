# training/loss_functions.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

###############################################################################
#  VAE                                                                        #
###############################################################################

def vae_loss(
    x_reconstructed: Tensor,
    x_target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    *,
    beta: float = 1.0,
    return_parts: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
    """Compute VAE loss = reconstruction + beta * KL (batch-wise means).

    Args:
        x_reconstructed: Decoder outputs of shape [B, D].
        x_target: Original inputs of shape [B, D].
        mu: Latent mean of shape [B, Z].
        logvar: Latent log-variance of shape [B, Z].
        beta: Weight of the KL term.
        return_parts: If True, also return (recon, kl).

    Returns:
        loss if return_parts is False; otherwise (loss, recon, kl).
    """
    cos = F.cosine_similarity(x_reconstructed, x_target, dim=-1)
    recon = (1.0 - cos).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon + beta * kl
    if return_parts:
        return loss, recon, kl
    return loss


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

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
    mse_reduction: str = "mean",   # "mean" o "sum"
    beta: float = 1.0,             # β-VAE (β=1 → VAE clásico)
) -> torch.Tensor:
    """VAE loss = reconstruction + β·KL  (KL normalizado por batch).

    Args:
        x_reconstructed: output del decoder  ― shape [B, D]
        x_target:        embeddings originales ― shape [B, D]
        mu, logvar:      parámetros de la distribución latente ― shape [B, Z]
        mse_reduction:   "mean" (recomendado) o "sum"
        beta:            peso del término KL (β-VAE)
    """
    # ── 1. reconstruction error ─────────────────────────────────────────
    recon = F.mse_loss(x_reconstructed, x_target, reduction=mse_reduction)

    # ── 2. KL (normalized) ───────────────────────────────────────────────
    #   KL(q(z|x) || N(0,1))  =  -½ Σ_i (1 + logσ²_i − μ²_i − σ²_i)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()  # ← .mean() ≈ /B/Z

    return recon + beta * kl

###############################################################################
#  DAE                                                                        #
###############################################################################

def dae_loss(
    x_reconstructed: torch.Tensor,
    x_clean: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean‑squared error para Denoising Auto‑Encoders."""
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
    """Triplet loss con selección de negativos dentro del batch.

    Si `hard_negatives` es True, usa el negativo más cercano; de lo contrario,
    permuta `z_pos` para obtener un negativo aleatorio.
    """
    z_q = F.normalize(z_q, p=2, dim=1)
    z_pos = F.normalize(z_pos, p=2, dim=1)

    if hard_negatives:
        dist_mat = torch.cdist(z_q, z_pos, p=2)
        mask = torch.eye(dist_mat.size(0), dtype=torch.bool, device=z_q.device)
        dist_mat = dist_mat.masked_fill(mask, float("inf"))  # ← corregido
        neg_dist, _ = dist_mat.min(dim=1)

    else:
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)
        neg_dist = torch.norm(z_q - z_pos[idx], dim=1)

    pos_dist = torch.norm(z_q - z_pos, dim=1)
    return F.relu(pos_dist - neg_dist + margin).mean()

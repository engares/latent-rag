# /training/loss_functions.py

import torch
import torch.nn.functional as F


###############################################################################
#  VAE                                                                        #
###############################################################################

def vae_loss(
    x_reconstructed: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Pérdida estándar VAE = mse + KL.

    Parámetros
    ----------
    x_reconstructed : Tensor
        Salida del decodificador.
    x : Tensor
        Embedding original (target).
    mu, logvar : Tensor
        Parámetros de la distribución latente.
    reduction : str
        Reducción a emplear en MSE ("sum" o "mean").
    """
    # Reconstrucción (MSE)
    recon_loss = F.mse_loss(x_reconstructed, x, reduction=reduction)

    # Divergencia KL
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

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
        dist_mat.masked_fill_(mask, float("inf"))
        neg_dist, _ = dist_mat.min(dim=1)
    else:
        idx = torch.randperm(z_pos.size(0), device=z_pos.device)
        neg_dist = torch.norm(z_q - z_pos[idx], dim=1)

    pos_dist = torch.norm(z_q - z_pos, dim=1)
    return F.relu(pos_dist - neg_dist + margin).mean()

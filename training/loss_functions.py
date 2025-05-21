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
    z_neg: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Loss contrastiva con margen (triplet)."""
    pos_dist = torch.norm(z_q - z_pos, dim=1)
    neg_dist = torch.norm(z_q - z_neg, dim=1)
    return F.relu(pos_dist - neg_dist + margin).mean()

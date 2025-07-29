import torch



def evaluate_reconstruction_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, reduction: str = "mean") -> float:
    """Calcula el error de reconstrucción (MSE)."""
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    return loss_fn(x_reconstructed, x).item()
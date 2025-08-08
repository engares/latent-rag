# evaluation/autoencoder_metrics.py

from __future__ import annotations
import torch
import os
from typing import Dict

BYTES_F32 = 4


def evaluate_reconstruction_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, reduction: str = "mean") -> float:
    """Calcula el error de reconstrucción (MSE)."""
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    return loss_fn(x_reconstructed, x).item()

def compression_ratio(dim_in: int, dim_out: int) -> float:
    if dim_out <= 0:
        raise ValueError("dim_out must be > 0")
    return dim_in / float(dim_out)

def sizeof_file(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0

def estimate_storage(n_vectors: int, dim: int, bytes_per_elem: int = BYTES_F32) -> int:
    if n_vectors < 0 or dim <= 0:
        raise ValueError("invalid n_vectors or dim")
    return n_vectors * dim * bytes_per_elem

def summarise_sizes(index_path: str, n_vectors: int, dim_in: int, dim_out: int) -> Dict[str, float]:
    idx_bytes = sizeof_file(index_path)
    est_bytes = estimate_storage(n_vectors, dim_out)
    return {
        "index_mb": idx_bytes / (1024 ** 2),
        "estimated_embeddings_mb": est_bytes / (1024 ** 2),
        "compression_ratio": compression_ratio(dim_in, dim_out),
    }

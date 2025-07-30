"""Quick t-SNE experiment for compressed vs. original SBERT embeddings.

This script **recomputes** the latent codes in memory (no disk cache for
compressed embeddings).  It supports the three AE variants shipped in the
repo (`contrastive`, `dae`, `vae`).

Example:

    python -m utils.visualization_exp \
        --sbert-cache data/SQUAD/sbert_cache/sbert_2254a38d6b_all-MiniLM-L6-v2.pt \
        --checkpoint   models/checkpoints/contrastive_ae.pth \
        --sample-size  1000 \
        --out          tsne_squad.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch

from evaluation.embedding_visualization import visualize_compressed_vs_original

# ---------------------------------------------------------------------------
#  Auto-encoder loader (minimal, standalone)                                  
# ---------------------------------------------------------------------------

def _load_autoencoder(ckpt: str, *, device: torch.device | str = "cpu") -> torch.nn.Module:
    """Instantiate the right AE class and load weights from *ckpt*."""

    # Heuristics based on filename – adjust if you rename checkpoints
    name = Path(ckpt).name.lower()
    if "contrastive" in name or "cae" in name:
        from models.contrastive_autoencoder import ContrastiveAutoencoder as AE
    elif "dae" in name:
        from models.denoising_autoencoder import DenoisingAutoencoder as AE
    elif "vae" in name:
        from models.variational_autoencoder import VariationalAutoencoder as AE
    else:
        raise ValueError(f"Cannot infer AE type from checkpoint name: {ckpt}")

    # 384->64 is hard-coded in config; change if you train different dims
    model = AE(input_dim=384, latent_dim=64, hidden_dim=512)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()

# ---------------------------------------------------------------------------
#  Load SBERT pairs from cache                                               
# ---------------------------------------------------------------------------

def _load_sbert_pairs(path: str | Path, n: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (queries, positives) tensors [n, 384] on CPU."""
    emb = torch.load(path, map_location="cpu")
    assert emb.dim() == 2 and emb.size(0) % 2 == 0, "Cache must be [2N, D]"
    n_pairs = emb.size(0) // 2
    n = min(n, n_pairs)
    idx = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(seed))[:n]
    q_idx = 2 * idx
    d_idx = q_idx + 1
    return emb[q_idx], emb[d_idx]

# ---------------------------------------------------------------------------
#  Main                                                                       
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – CLI entry-point
    p = argparse.ArgumentParser("t-SNE visualisation of AE-compressed embeddings")
    p.add_argument("--sbert-cache", required=True, help=".pt file with cached SBERT [q, ctx] pairs interleaved")
    p.add_argument("--checkpoint",   required=True, help="Path to trained AE checkpoint (*.pth)")
    p.add_argument("--sample-size",  type=int, default=1000, help="Number of query–doc pairs to plot (default 1000)")
    p.add_argument("--perplexity",   type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--out",          default="tsne.png", help="Output figure file (PNG/PDF)")
    p.add_argument("--seed",         type=int, default=42, help="Random seed for sampling")
    args = p.parse_args()

    # 1. SBERT originals
    q_orig, d_orig = _load_sbert_pairs(args.sbert_cache, args.sample_size, seed=args.seed)

    # 2. AE latents (in memory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = _load_autoencoder(args.checkpoint, device=device)

    with torch.no_grad():
        q_comp = ae.encode(q_orig.to(device))
        if isinstance(q_comp, tuple):  # VAE returns (mu, logvar)
            q_comp = q_comp[0]
        d_comp = ae.encode(d_orig.to(device))
        if isinstance(d_comp, tuple):
            d_comp = d_comp[0]
        # back to CPU for t-SNE
        q_comp, d_comp = q_comp.cpu(), d_comp.cpu()

    # 3. Visualise
    metrics = visualize_compressed_vs_original(
        q_orig, d_orig, q_comp, d_comp,
        sample_size=1_000,
        max_lines=200,        # resalta las 200 parejas más cercanas
        color_by_dist=True,   # verde = muy próximas, rojo = alejadas
        save_path="tsne.png",
    )


    print("\nTrustworthiness:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Figure saved → {args.out}\n")


if __name__ == "__main__":
    main()

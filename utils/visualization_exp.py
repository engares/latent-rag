"""CLI helper — visualise compressed vs. original SBERT embeddings.

Supports both t‑SNE and PCA as low‑dimensional projections and 2‑D or 3‑D
scatter plots.  Latent embeddings are **recomputed in memory** (no cache for
compressed codes).  Works with the three AE checkpoints in the repo
(`contrastive`, `dae`, `vae`).

If the training script saved a meta JSON (see models/history/*.json), the
network architecture (input_dim, latent_dim, hidden_dim) is automatically
inferred from it. You can also pass it explicitly via --meta. Fallback defaults
are (384, 32, 1024).

Example

python -m utils.visualization_exp \
  --sbert-cache data/SQUAD/sbert_cache/sbert_8ebee4d368_all-MiniLM-L6-v2.pt \
  --checkpoint  models/checkpoints/cae_text.pth \
  --projection  tsne \
  --components  2 \
  --sample-size 1200 \
  --k-near 10 \

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import json
import torch

from evaluation.embedding_visualization import visualize_compressed_vs_original

# ---------------------------------------------------------------------------
#  Auto‑encoder loader                                                        
# ---------------------------------------------------------------------------

def _infer_meta_path(ckpt_path: str | Path) -> Path:
    """Given a checkpoint path, guess corresponding meta JSON path.

    The training scripts save a meta file in models/history/<stem>.json where
    <stem> matches the checkpoint filename (without extension).
    """
    ckpt_path = Path(ckpt_path)
    stem = ckpt_path.stem  # remove .pth
    return Path("models/history") / f"{stem}.json"

def _load_meta(meta_path: Optional[str | Path]) -> Dict[str, Any]:
    """Load architecture hyperparameters from a meta JSON.

    Returns empty dict if file missing / unreadable.
    """
    if not meta_path:
        return {}
    p = Path(meta_path)
    if not p.is_file():
        return {}
    try:
        with p.open("r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _load_autoencoder(
    ckpt: str,
    *,
    device: torch.device | str = "cpu",
    meta_path: Optional[str | Path] = None,
) -> torch.nn.Module:
    """Instantiate the proper AE subclass and load weights from *ckpt*.

    Tries to read architecture parameters (input_dim, latent_dim, hidden_dim)
    from a meta JSON. Resolution order:
      1. Explicit --meta path if provided
      2. Auto‑inferred models/history/<ckpt_stem>.json
      3. Hardcoded defaults (384, 32, 1024)
    """
    name = Path(ckpt).name.lower()
    if "cae" in name:
        from models.contrastive_autoencoder import ContrastiveAutoencoder as AE
    elif "dae" in name:
        from models.denoising_autoencoder import DenoisingAutoencoder as AE
    elif "vae" in name:
        from models.variational_autoencoder import VariationalAutoencoder as AE
    else:
        raise ValueError(f"Cannot infer AE type from checkpoint name: {ckpt}")

    # meta resolution
    if meta_path is None:
        inferred = _infer_meta_path(ckpt)
        meta = _load_meta(inferred)
    else:
        meta = _load_meta(meta_path)
        if not meta:  # fallback to inferred if explicit fails
            meta = _load_meta(_infer_meta_path(ckpt))

    input_dim = int(meta.get("input_dim", 384))
    latent_dim = int(meta.get("latent_dim", 32))
    hidden_dim = int(meta.get("hidden_dim", 1024))

    model = AE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()

# ---------------------------------------------------------------------------
#  SBERT pairs loader                                                         
# ---------------------------------------------------------------------------

def _load_sbert_pairs(path: str | Path, n: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(queries, positives)` tensors of shape [n, 384] (CPU)."""
    emb = torch.load(path, map_location="cpu")
    if emb.dim() != 2 or emb.size(0) % 2 != 0:
        raise ValueError("SBERT cache must have shape [2N, D]")
    n_pairs = emb.size(0) // 2
    n = min(n, n_pairs)
    idx = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(seed))[:n]
    q_idx = 2 * idx
    d_idx = q_idx + 1
    return emb[q_idx], emb[d_idx]

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _infer_ae_type(ckpt_name: str) -> str:
    """Return a short tag identifying the AE flavour."""
    lower = ckpt_name.lower()
    if "cae" in lower:
        return "cae"
    if "dae" in lower:
        return "dae"
    if "vae" in lower:
        return "vae"
    return "ae"


def _build_default_path(
    ckpt: str,
    *,
    projection: str,
    n_components: int,
    sample_size: int,
    k_near: int,
    perplexity: float,
) -> Path:
    """Compose a descriptive file name and ensure ``fig/`` exists.

    The directory is created with parents if missing.
    """
    ae_tag = _infer_ae_type(Path(ckpt).stem)
    fname = f"{ae_tag}_{projection}_{n_components}d_{sample_size}s_{k_near}k"
    if projection == "tsne":
        fname += f"_perp{int(perplexity)}"
    fname += ".png"

    fig_dir = Path("fig")  # ← default relative folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / fname


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        "Visualise AE-compressed vs. original embeddings"
    )

    # required paths
    parser.add_argument(
        "--sbert-cache",
        required=True,
        help=".pt file with SBERT cache (queries/ctx interleaved)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained AE checkpoint (.pth)",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="Optional path to meta JSON with architecture (auto-infer if omitted)",
    )

    # visual options
    parser.add_argument(
        "--projection",
        choices=["tsne", "pca"],
        default="tsne",
        help="Low-D projection method",
    )
    parser.add_argument(
        "--components",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of projection dimensions",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (ignored for PCA)",
    )
    parser.add_argument(
        "--k-near",
        type=int,
        default=5,
        help="Nearest-neighbour threshold for hits",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins for distance plot",
    )

    # sampling & io
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of query–doc pairs to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--out",
        default=None,  # ← None triggers automatic path generation
        help=(
            "Output figure path (PNG/PDF).  If omitted, the file is "
            "saved to ./fig/<params>.png"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load SBERT originals
    # ------------------------------------------------------------------ #
    q_orig, d_orig = _load_sbert_pairs(args.sbert_cache, args.sample_size, seed=args.seed)

    # ------------------------------------------------------------------ #
    # 2. Compress with AE (on-the-fly)
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = _load_autoencoder(args.checkpoint, device=device, meta_path=args.meta)

    with torch.no_grad():
        q_comp = ae.encode(q_orig.to(device))
        if isinstance(q_comp, tuple):  # VAE returns (mu, logvar)
            q_comp = q_comp[0]
        d_comp = ae.encode(d_orig.to(device))
        if isinstance(d_comp, tuple):
            d_comp = d_comp[0]
        q_comp, d_comp = q_comp.cpu(), d_comp.cpu()

    # ------------------------------------------------------------------ #
    # 3. Determine output path                                           #
    # ------------------------------------------------------------------ #
    if args.out is None:  # ← NEW
        args.out = _build_default_path(
            args.checkpoint,
            projection=args.projection,
            n_components=args.components,
            sample_size=args.sample_size,
            k_near=args.k_near,
            perplexity=args.perplexity,
        )

    # ------------------------------------------------------------------ #
    # 4. Visualise                                                       #
    # ------------------------------------------------------------------ #
    _ = visualize_compressed_vs_original(
        q_orig,
        d_orig,
        q_comp,
        d_comp,
        projection=args.projection,
        n_components=args.components,
        sample_size=args.sample_size,
        k_near=args.k_near,
        perplexity=args.perplexity,
        bins=args.bins,
        random_state=args.seed,
        save_path=str(args.out),  # ensure Path → str
        save_negatives_path=str(args.out).replace(".png", "_negatives_distribution.png"),  # Save negatives plot
    )

    print(f"Figure saved  {args.out}\n")


if __name__ == "__main__":
    main()
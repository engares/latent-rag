"""Embedding visualisation utilities — 2‑D *and* 3‑D, t‑SNE or PCA.

Features
========
* **Projection back‑end**: choose `'tsne'` (default) or `'pca'`.
* **Dimensionality**: 2‑D (scatter) or 3‑D (interactive rotation).
* **Binary hit/miss coding** with recall @ *k* and trustworthiness.
* **Histogram + CDF** of pairwise distances in the chosen sub‑space.

Public function
---------------
`visualize_compressed_vs_original(…, projection="tsne", n_components=2, …)`

Example
~~~~~~~
```python
visualize_compressed_vs_original(
    q_orig, d_orig, q_comp, d_comp,
    n_components=3,        # 3‑D scatter
    projection="pca",     # PCA instead of t‑SNE
    k_near=10,
    perplexity=35,         # ignored for PCA
    save_path="viz_3d.png",
)
```
"""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from matplotlib.ticker import PercentFormatter
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – necessary for 3‑D

Tensor = torch.Tensor

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Tensor):
    return x.detach().cpu().float().numpy()


def _rank_positive(q: Tensor, d: Tensor) -> Tensor:
    """1‑based rank of each positive doc (cosine similarity)."""
    sim = F.cosine_similarity(q.unsqueeze(1), d.unsqueeze(0), dim=-1)
    return sim.argsort(dim=1, descending=True).argsort(dim=1).diagonal() + 1


def _project(x: Tensor, *, method: str, n_components: int, perplexity: float, seed: int) -> Tensor:
    """Return low‑D embedding (torch.Tensor) via t‑SNE or PCA."""
    if method == "tsne":
        tsne = TSNE(n_components=n_components, perplexity=perplexity, metric="cosine", init="pca", max_iter=1_000, random_state=seed)
        return torch.from_numpy(tsne.fit_transform(_to_numpy(x))).float()
    elif method == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        return torch.from_numpy(pca.fit_transform(_to_numpy(x))).float()
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

# ---------------------------------------------------------------------------
#  Plotting primitives
# ---------------------------------------------------------------------------

def _scatter(ax, pts: Tensor, colour: str, marker: str, label: str, **kw):
    if pts.size(1) == 2:
        ax.scatter(pts[:, 0], pts[:, 1], c=colour, marker=marker, label=label, **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colour, marker=marker, label=label, **kw)


def _link(ax, p: Tensor, q: Tensor, colour: str):
    if p.numel() == 2:            # 2-D
        ax.plot([p[0], q[0]], [p[1], q[1]],
                color=colour, linewidth=0.8, alpha=0.8)
    else:                         # 3-D
        ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]],
                color=colour, linewidth=0.8, alpha=0.8)



def _scatter_pairs(ax, q_emb: Tensor, d_emb: Tensor, hit: Tensor):
    """Draw queries/docs and red links for misses."""
    dim = q_emb.size(1)
    if dim == 3:
        ax = ax  # 3‑D Axes
    # base cloud
    _scatter(ax, q_emb, "C0", "o", "Query", s=30, alpha=0.3)
    _scatter(ax, d_emb, "C1", "^", "Doc",   s=30, alpha=0.3)
    # hits green
    _scatter(ax, q_emb[hit], "forestgreen", "o", "Hit q", s=35, alpha=0.3, zorder=3)
    _scatter(ax, d_emb[hit], "forestgreen", "^", "Hit d", s=35, alpha=0.3, zorder=3)
    miss = ~hit
    _scatter(ax, q_emb[miss], "crimson", "o", "Miss q", s=35, alpha=0.3, zorder=4)
    _scatter(ax, d_emb[miss], "gold", "^", "Miss d", s=35, alpha=0.3, zorder=4)
    for p, q in zip(q_emb[miss], d_emb[miss]):
        _link(ax, p, q, "crimson")
    ax.set_xticks([])
    ax.set_yticks([])
    if dim == 3:
        ax.set_zticks([])
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def _hist_cdf(ax: plt.Axes, d1: Tensor, d2: Tensor, *, bins: int):
    ax.hist(d1.numpy(), bins=bins, alpha=0.5, label="Orig dist")
    ax.hist(d2.numpy(), bins=bins, alpha=0.5, label="Comp dist")
    ax.set_xlabel("Pair distance |q - d|")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False, fontsize=7)
    ax2 = ax.twinx()
    for data, lbl in [(d1, "Orig CDF"), (d2, "Comp CDF")]:
        sorted_vals = torch.sort(data).values
        cdf = torch.arange(1, len(data)+1) / len(data)
        ax2.plot(sorted_vals, cdf, label=lbl)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("CDF")
    ax2.legend(frameon=False, fontsize=7, loc="lower right")

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def visualize_compressed_vs_original(
    q_orig: Tensor,
    d_orig: Tensor,
    q_comp: Tensor,
    d_comp: Tensor,
    *,
    projection: str = "tsne",      # 'tsne' or 'pca'
    n_components: int = 2,          # 2 or 3
    sample_size: int = 1_000,
    k_near: int = 5,
    perplexity: float = 30.0,       # only for t‑SNE
    bins: int = 30,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> Dict[str, float]:

    if n_components not in {2, 3}:
        raise ValueError("n_components must be 2 or 3")
    if projection not in {"tsne", "pca"}:
        raise ValueError("projection must be 'tsne' or 'pca'")
    if not (len(q_orig) == len(d_orig) == len(q_comp) == len(d_comp)):
        raise ValueError("Input lengths mismatch")

    torch.manual_seed(random_state)
    N = len(q_orig)
    if sample_size < N:
        idx = torch.randperm(N)[:sample_size]
        q_o, d_o, q_c, d_c = q_orig[idx], d_orig[idx], q_comp[idx], d_comp[idx]
    else:
        q_o, d_o, q_c, d_c = q_orig, d_orig, q_comp, d_comp

    rank = _rank_positive(q_o, d_o)
    hit = rank <= k_near

    orig_proj = _project(torch.cat([q_o, d_o]), method=projection, n_components=n_components, perplexity=perplexity, seed=random_state)
    comp_proj = _project(torch.cat([q_c, d_c]), method=projection, n_components=n_components, perplexity=perplexity, seed=random_state)
    q_proj_o, d_proj_o = orig_proj[: len(q_o)], orig_proj[len(q_o) :]
    q_proj_c, d_proj_c = comp_proj[: len(q_o)], comp_proj[len(q_o) :]

    dist_o = torch.linalg.norm(q_proj_o - d_proj_o, dim=1)
    dist_c = torch.linalg.norm(q_proj_c - d_proj_c, dim=1)

    tq = trustworthiness(_to_numpy(torch.cat([q_o, d_o])), _to_numpy(orig_proj))
    tc = trustworthiness(_to_numpy(torch.cat([q_c, d_c])), _to_numpy(comp_proj))

    # ---- layout -------------------------------------------------------
    fig_height = 10 if n_components == 2 else 12
    fig = plt.figure(figsize=(14, fig_height))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2])

    # top‑left
    if n_components == 3:
        ax_t1 = fig.add_subplot(gs[0, 0], projection="3d")
    else:
        ax_t1 = fig.add_subplot(gs[0, 0])
    _scatter_pairs(ax_t1, q_proj_o, d_proj_o, hit)
    ax_t1.set_title(f"Original {projection.upper()} – Recall@{k_near}: {hit.float().mean():.1%} | Trust: {tq:.3f}")

    # top‑right
    if n_components == 3:
        ax_t2 = fig.add_subplot(gs[0, 1], projection="3d")
    else:
        ax_t2 = fig.add_subplot(gs[0, 1])
    _scatter_pairs(ax_t2, q_proj_c, d_proj_c, hit)
    ax_t2.set_title(f"Compressed {projection.upper()} – Recall@{k_near}: {hit.float().mean():.1%} | Trust: {tc:.3f}")

    # bottom histogram
    ax_hist = fig.add_subplot(gs[1, :])
    _hist_cdf(ax_hist, dist_o, dist_c, bins=bins)
    ax_hist.set_title(f"Pair distance distribution ({projection.upper()} {n_components}‑D)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {"trust_original": float(tq), "trust_compressed": float(tc)}

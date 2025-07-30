"""Embedding visualisation utilities — *extended*.

Adds distance‑aware styling and optional filtering to make the query↔doc
relationship visually obvious while keeping backward compatibility.

Public API (superset of original):
---------------------------------
* `plot_query_doc_pairs` now accepts:
    - `max_lines     : int | None`  – draw only the *k* shortest/longest links.
    - `color_by_dist : bool`        – map link colour to Euclidean distance.
* `visualize_compressed_vs_original` exposes those params via `kwargs`.

Example usage (unchanged defaults preserve previous behaviour):

```python
visualize_compressed_vs_original(
    q_orig, d_orig, q_comp, d_comp,
    sample_size=1_000,
    max_lines=200,              # draw only 200 closest pairs
    color_by_dist=True,         # green→red based on |q‑d|
    save_path="tsne.png",
)
```
"""
from __future__ import annotations

from typing import Dict, Optional, Literal

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from sklearn.manifold import TSNE, trustworthiness

Tensor = torch.Tensor  # alias for brevity

# ---------------------------------------------------------------------------
#  Low‑level helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: Tensor):  # -> np.ndarray (forward ref avoided)
    return x.detach().cpu().float().numpy()

# ---------------------------------------------------------------------------
#  t‑SNE
# ---------------------------------------------------------------------------

def tsne_projection(
    embeddings: Tensor,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Tensor:
    if embeddings.dim() != 2:
        raise ValueError("`embeddings` must be 2‑D [N, D].")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric="cosine",
        init="pca",
        n_iter=1_000,
        random_state=random_state,
    )
    return torch.from_numpy(tsne.fit_transform(_to_numpy(embeddings))).float()

# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def _compute_distances(a: Tensor, b: Tensor) -> Tensor:
    """Euclidean distance between corresponding rows."""
    return ((a - b).pow(2).sum(1)).sqrt()


def plot_query_doc_pairs(
    q_emb2d: Tensor,
    d_emb2d: Tensor,
    *,
    ax: Optional[plt.Axes] = None,
    connect: bool = True,
    alpha: float = 0.8,
    point_size: int = 40,
    max_lines: Optional[int] = None,
    line_selection: Literal["shortest", "longest"] = "shortest",
    color_by_dist: bool = False,
) -> plt.Axes:
    """Scatter of queries + positives with optional distance‑aware links.

    Args:
        max_lines: Draw at most *k* links; picks the `line_selection` shortest
            or longest.  `None` = draw all.
        color_by_dist: If *True*, colour links (and optional edge of markers)
            using a **linear** green→red colormap based on 2‑D distance.
    """
    if q_emb2d.shape != d_emb2d.shape:
        raise ValueError("Shapes mismatch.")
    if q_emb2d.size(1) != 2:
        raise ValueError("Expect [N, 2].")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # ---- base scatter --------------------------------------------------
    ax.scatter(q_emb2d[:, 0], q_emb2d[:, 1], c="C0", label="Query", alpha=alpha, s=point_size)
    ax.scatter(d_emb2d[:, 0], d_emb2d[:, 1], c="C1", marker="^", label="Positive doc", alpha=alpha, s=point_size)

    if connect:
        dist = _compute_distances(q_emb2d, d_emb2d)
        order = torch.arange(len(dist))
        if max_lines is not None and max_lines < len(dist):
            if line_selection == "shortest":
                order = dist.topk(max_lines, largest=False).indices
            else:  # longest
                order = dist.topk(max_lines, largest=True).indices
        cmap = mpl.cm.get_cmap("RdYlGn_r")
        norm = mpl.colors.Normalize(vmin=float(dist.min()), vmax=float(dist.max()))
        for i in order.tolist():
            q, d = q_emb2d[i], d_emb2d[i]
            color = cmap(norm(float(dist[i]))) if color_by_dist else "gray"
            ax.plot([q[0], d[0]], [q[1], d[1]], color=color, linewidth=0.8, alpha=0.7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    return ax

# ---------------------------------------------------------------------------
#  High‑level routine
# ---------------------------------------------------------------------------

def visualize_compressed_vs_original(
    q_original: Tensor,
    d_original: Tensor,
    q_compressed: Tensor,
    d_compressed: Tensor,
    *,
    sample_size: int = 1_000,
    perplexity: float = 30.0,
    n_neighbors: int = 5,
    random_state: int = 42,
    save_path: Optional[str] = None,
    # new visual params
    max_lines: Optional[int] = None,
    color_by_dist: bool = False,
    line_selection: Literal["shortest", "longest"] = "shortest",
) -> Dict[str, float]:

    if not (len(q_original) == len(d_original) == len(q_compressed) == len(d_compressed)):
        raise ValueError("All input batches must share the same length.")

    torch.manual_seed(random_state)
    n_samples = len(q_original)
    if sample_size < n_samples:
        idx = torch.randperm(n_samples)[:sample_size]
        q_orig, d_orig = q_original[idx], d_original[idx]
        q_comp, d_comp = q_compressed[idx], d_compressed[idx]
    else:
        q_orig, d_orig, q_comp, d_comp = q_original, d_original, q_compressed, d_compressed

    # 2‑D projection
    orig_2d = tsne_projection(torch.cat([q_orig, d_orig]), perplexity=perplexity, random_state=random_state)
    comp_2d = tsne_projection(torch.cat([q_comp, d_comp]), perplexity=perplexity, random_state=random_state)
    n = len(q_orig)
    q_orig_2d, d_orig_2d = orig_2d[:n], orig_2d[n:]
    q_comp_2d, d_comp_2d = comp_2d[:n], comp_2d[n:]

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_query_doc_pairs(
        q_orig_2d, d_orig_2d, ax=axes[0], connect=True, max_lines=max_lines,
        color_by_dist=color_by_dist, line_selection=line_selection
    )
    axes[0].set_title("Original embeddings")

    plot_query_doc_pairs(
        q_comp_2d, d_comp_2d, ax=axes[1], connect=True, max_lines=max_lines,
        color_by_dist=color_by_dist, line_selection=line_selection
    )
    axes[1].set_title("Compressed embeddings")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    tq = trustworthiness(_to_numpy(torch.cat([q_orig, d_orig])), _to_numpy(orig_2d), n_neighbors=n_neighbors)
    tc = trustworthiness(_to_numpy(torch.cat([q_comp, d_comp])), _to_numpy(comp_2d), n_neighbors=n_neighbors)
    return {"trust_original": float(tq), "trust_compressed": float(tc)}

from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.colors import Colormap
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3-D scatter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
Tensor = torch.Tensor  

import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


_MARKER_CYCLE: MutableSequence[str] = ["o","s","D","P","X","^","v","<",">", "o"]


def _to_numpy(x: Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    """Detach a tensor and move to CPU as *float32* NumPy array."""
    return x.detach().cpu().float().numpy()


def _rank_positive(q: Tensor, d: Tensor) -> Tensor:
    """Return **1-based** rank of each positive document using cosine similarity."""
    sim = F.cosine_similarity(q.unsqueeze(1), d.unsqueeze(0), dim=-1)
    return sim.argsort(dim=1, descending=True).argsort(dim=1).diagonal() + 1


def _project(
    x: Tensor,
    *,
    method: str,
    n_components: int,
    perplexity: float,
    seed: int,
) -> Tensor:
    """Return a low-dimensional projection via t-SNE or PCA."""
    if method == "tsne":
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            max_iter=1_000,
            random_state=seed,
        )
        return torch.from_numpy(tsne.fit_transform(_to_numpy(x))).float()
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=seed)
        return torch.from_numpy(pca.fit_transform(_to_numpy(x))).float()
    raise ValueError("method must be 'tsne' or 'pca'")


def _build_marker_map(groups: Iterable[str]) -> Mapping[str, str]:
    """Assign each *group* a marker shape cycling through `_MARKER_CYCLE`."""
    cycle_iter = cycle(_MARKER_CYCLE)
    return {grp: next(cycle_iter) for grp in sorted(set(groups))}


# ---------------------------------------------------------------------------
# Plotting primitives
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


def _scatter_pairs(
    ax,
    q_emb: Tensor,
    d_emb: Tensor,
    dist: Tensor,
    *,
    cmap_name: str = "viridis_r",
    alpha: float = 0.3,
    s: int = 30,
    max_links: int = 10,
):
    """
    Dibuja las parejas (query-doc) coloreadas según la distancia del coseno.

    Params
    ------
    q_emb, d_emb : (N, 2|3)  Embeddings proyectados de queries y docs.
    dist         : (N,)      Distancias del coseno ||q - d|| en el espacio original.
    max_links    : int       Número máximo de enlaces (los más disimilares).
    """
    dim = q_emb.size(1)
    norm = Normalize(vmin=float(dist.min()), vmax=float(dist.max()))
    cmap = cm.get_cmap(cmap_name)

    # Matriz RGBA con alpha constante
    colors = cmap(norm(_to_numpy(dist)))
    colors[:, -1] = alpha

    # Dispersión de queries
    ax.scatter(
        q_emb[:, 0], q_emb[:, 1],
        *(q_emb[:, 2].T,) if dim == 3 else (),
        c=colors, marker="o", s=s, label="Query"
    )
    # Dispersión de docs
    ax.scatter(
        d_emb[:, 0], d_emb[:, 1],
        *(d_emb[:, 2].T,) if dim == 3 else (),
        c=colors, marker="^", s=s, label="Doc"
    )

    # Índices de las distancias más altas
    top_idx = dist.topk(min(max_links, len(dist))).indices

    # Enlaces coloreados (solo top-k)
    for i in top_idx:
        _link(ax, q_emb[i], d_emb[i], colour=colors[i])

    ax.set_xticks([]); ax.set_yticks([])
    if dim == 3:
        ax.set_zticks([])
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def _hist_cdf(ax: "plt.Axes", d1: Tensor, d2: Tensor, *, bins: int) -> None:
    """Histogram + CDF of two 1-D distance distributions."""
    cmap = cm.get_cmap("viridis")  # Use the viridis colormap
    color1 = cmap(0.3)  # A color from the lower range of viridis
    color2 = cmap(0.7)  # A color from the higher range of viridis

    # Histogram
    ax.hist(d1.numpy(), bins=bins, alpha=0.5, label="Original dist.", color=color1)
    ax.hist(d2.numpy(), bins=bins, alpha=0.5, label="Compressed dist.", color=color2)
    ax.set_xlabel("Pair cosine distance ")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False, fontsize=7)

    # CDF
    ax2 = ax.twinx()
    for data, lbl, color in ((d1, "Original CDF", color1), (d2, "Compressed CDF", color2)):
        sorted_vals = torch.sort(data).values
        cdf = torch.arange(1, len(data) + 1) / len(data)
        ax2.plot(sorted_vals, cdf, label=lbl, color=color)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylabel("CDF")
    ax2.legend(frameon=False, fontsize=7, loc="lower right")


def visualize_compressed_vs_original(
    q_orig: Tensor,
    d_orig: Tensor,
    q_comp: Tensor,
    d_comp: Tensor,
    *,
    projection: str = "tsne",
    n_components: int = 2,
    sample_size: int = 1_000,
    k_near: int = 5,
    perplexity: float = 30.0,
    bins: int = 30,
    random_state: int = 42,
    q_groups: Optional[Sequence[str]] = None,
    k_near_colour_cut: Optional[int] = None,
    save_path: Optional[str] = None,
    save_negatives_path: Optional[str] = None,
) -> Dict[str, float]:
    """Visualise original vs. compressed embeddings and guardar dos figuras:
      1) scatter + hist/CDF
      2) dist positives vs. negatives."""

    torch.manual_seed(random_state)
    N = len(q_orig)
    if sample_size < N:
        idx = torch.randperm(N)[:sample_size]
        q_o, d_o, q_c, d_c = q_orig[idx], d_orig[idx], q_comp[idx], d_comp[idx]
    else:
        q_o, d_o, q_c, d_c = q_orig, d_orig, q_comp, d_comp


    # --- Métricas de recall
    rank_orig = _rank_positive(q_o, d_o)
    recall_orig = (rank_orig <= k_near).float().mean()
    rank_comp = _rank_positive(q_c, d_c)
    recall_comp = (rank_comp <= k_near).float().mean()

    # --- Proyecciones (solo para la figura 1)
    orig_proj = _project(torch.cat([q_o, d_o]), method=projection,
                         n_components=n_components, perplexity=perplexity, seed=random_state)
    comp_proj = _project(torch.cat([q_c, d_c]), method=projection,
                         n_components=n_components, perplexity=perplexity, seed=random_state)
    q_proj_o, d_proj_o = orig_proj[: len(q_o)], orig_proj[len(q_o):]
    q_proj_c, d_proj_c = comp_proj[: len(q_o)], comp_proj[len(q_o):]

    # --- Distancias del coseno
    dist_o = 1 - F.cosine_similarity(q_o, d_o, dim=1)
    dist_c = 1 - F.cosine_similarity(q_c, d_c, dim=1)

    # ========== FIGURA 1: scatter + hist/CDF ==========
    fig = plt.figure(figsize=(14, 10 if n_components==2 else 12))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1,1,0.05], height_ratios=[3,2])

    ax1 = fig.add_subplot(gs[0,0], projection="3d" if n_components==3 else None)
    ax2 = fig.add_subplot(gs[0,1], projection="3d" if n_components==3 else None)
    cbar_ax = fig.add_subplot(gs[0,2])
    ax_hist = fig.add_subplot(gs[1,:2])

    _scatter_pairs(ax1, q_proj_o, d_proj_o, dist_o)
    _scatter_pairs(ax2, q_proj_c, d_proj_c, dist_c)
    ax1.set_title(f"Original {projection.upper()} – Recall@{k_near}: {recall_orig:.1%}", fontsize=10)
    ax2.set_title(f"Compressed {projection.upper()} – Recall@{k_near}: {recall_comp:.1%}", fontsize=10)
    _hist_cdf(ax_hist, dist_o, dist_c, bins=bins)
    ax_hist.set_title(f"Pair distance distribution ({projection.upper()} {n_components}-D)")

    # colorbar única
    sm = cm.ScalarMappable(norm=Normalize(vmin=float(min(dist_o.min(), dist_c.min())),
                                         vmax=float(max(dist_o.max(), dist_c.max()))),
                           cmap="viridis_r")
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label("Cosine distance", rotation=270, labelpad=15)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    plot_positive_vs_negative_distances(
        q_o, d_o, q_c, d_c,
        bins=bins,
        save_negatives_path=save_negatives_path
    )

    return {
        "recall_original": float(recall_orig),
        "recall_compressed": float(recall_comp),
    }


def plot_positive_vs_negative_distances(
    q_o: Tensor,
    d_o: Tensor,
    q_c: Tensor,
    d_c: Tensor,
    *,
    bins: int,
    save_negatives_path: Optional[str] = None
) -> None:
    """Plot positive vs. negative distances for original and compressed embeddings."""
    # Muestreo de negativos (una permutación aleatoria)
    perm = torch.randperm(len(d_o))
    neg_idx = torch.where(perm == torch.arange(len(d_o)), (perm + 1) % len(d_o), perm)
    d_neg_o = d_o[neg_idx]
    d_neg_c = d_c[neg_idx]

    # Calcular distancias del coseno para positivos y negativos
    dist_o = 1 - F.cosine_similarity(q_o, d_o, dim=1)
    dist_c = 1 - F.cosine_similarity(q_c, d_c, dim=1)
    dist_neg_o = 1 - F.cosine_similarity(q_o, d_neg_o, dim=1)
    dist_neg_c = 1 - F.cosine_similarity(q_c, d_neg_c, dim=1)

    fig, (axp, axn) = plt.subplots(1, 2, figsize=(12, 5))
    # Distribución originales
    axp.hist(dist_o.numpy(), bins=bins, alpha=0.6, label="Positives", color="C0")
    axp.hist(dist_neg_o.numpy(), bins=bins, alpha=0.6, label="Negatives", color="C1")
    axp.set_title("Original: q–d⁺ vs q–d⁻")
    axp.set_xlabel("Cosine distance")
    axp.set_ylabel("Frequency")
    axp.legend(frameon=False)

    # Distribución comprimidas
    axn.hist(dist_c.numpy(), bins=bins, alpha=0.6, label="Positives", color="C0")
    axn.hist(dist_neg_c.numpy(), bins=bins, alpha=0.6, label="Negatives", color="C1")
    axn.set_title("Compressed: q–d⁺ vs q–d⁻")
    axn.set_xlabel("Cosine distance")
    axn.legend(frameon=False)

    fig.tight_layout()
    if save_negatives_path:
        fig.savefig(save_negatives_path, dpi=300, bbox_inches="tight")
    plt.show()

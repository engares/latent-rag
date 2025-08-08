# retrieval/bruteforce.py
from __future__ import annotations
from typing import Sequence, Tuple, List, Literal
import time

import torch
import torch.nn.functional as F

from retrieval.common import (
    normalize_l2_torch,
    StatsTracker,
)

Similarity = Literal["cosine", "euclidean"]


class BruteForceRetriever:
    """
    Recuperador por fuerza bruta (exacto) con métricas de rendimiento.

    - Soporta 'cosine' (recomendado) y 'euclidean'.
    - Expone get_stats(reset=False) con build_time_s, search_time_s,
      search_calls y per_query_ms, homogéneo con FAISS.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,   # [N × D] CPU o GPU; se moverá a CPU
        texts: Sequence[str],
        doc_ids: Sequence[int] | None = None,
        metric: Similarity = "cosine",
    ):
        if doc_ids is not None:
            assert len(texts) == len(doc_ids), "len mismatch (texts vs doc_ids)"

        # Asegurar CPU
        if embeddings.device.type != "cpu":
            embeddings = embeddings.cpu()

        self.texts = list(texts)
        self.doc_ids = list(doc_ids) if doc_ids is not None else list(range(len(texts)))
        self.metric = metric

        # Métricas
        self._stats = StatsTracker()

        # "Construcción": normalización si coseno (medimos el coste)
        t0 = time.perf_counter()
        if metric == "cosine":
            self.emb = normalize_l2_torch(embeddings, dim=1).contiguous()
        elif metric == "euclidean":
            self.emb = embeddings.contiguous()
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        self._stats.add_build_time(time.perf_counter() - t0)

    # API por lotes (coincidir con FAISS.search)
    def search(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        if queries.device.type != "cpu":
            queries = queries.cpu()

        # Normalización de queries si coseno
        t0 = time.perf_counter()
        if self.metric == "cosine":
            q = normalize_l2_torch(queries, dim=1)
            # scores = Q @ E^T
            scores = q @ self.emb.T                       # [B, N]
        else:
            # -||q - e||_2 para ranking descendente
            # (vectorizado: ||q||^2 + ||e||^2 - 2 q·e)
            q = queries
            q2 = (q * q).sum(dim=1, keepdim=True)         # [B,1]
            e2 = (self.emb * self.emb).sum(dim=1).unsqueeze(0)  # [1,N]
            scores = -(q2 + e2 - 2.0 * (q @ self.emb.T))  # [B,N]
        dt = time.perf_counter() - t0
        self._stats.add_search_batch(batch_size=len(queries), seconds=dt)

        # top‑k para cada fila
        k = min(k, scores.size(1))
        vals, idxs = torch.topk(scores, k=k, dim=1)       # [B,k]
        return vals.numpy(), idxs.numpy()

    # Conveniencia: una sola query
    def retrieve(self, query_emb: torch.Tensor, top_k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        D, I = self.search(query_emb, top_k)
        idxs = I[0].tolist()
        texts = [self.texts[i] for i in idxs]
        scores = D[0].tolist()
        docids = [self.doc_ids[i] for i in idxs]
        return texts, scores, docids

    def get_stats(self, reset: bool = False):
        return self._stats.get_stats(reset=reset)

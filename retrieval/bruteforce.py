# retrieval/bruteforce.py
from __future__ import annotations
from typing import Sequence, Tuple, List, Literal

import torch
import torch.nn.functional as F

Similarity = Literal["cosine", "euclidean"]

class BruteForceRetriever:
    def __init__(
        self,
        embeddings: torch.Tensor,   # [N × D] CPU float32
        texts: Sequence[str],
        doc_ids: Sequence[int] | None = None,
        metric: Similarity = "cosine",
    ):
        if doc_ids is not None:
            assert len(texts) == len(doc_ids), "len mismatch (texts vs doc_ids)"

        if embeddings.device.type != "cpu":
            embeddings = embeddings.cpu()

        self.texts = list(texts)
        self.doc_ids = list(doc_ids) if doc_ids is not None else list(range(len(texts)))
        self.metric = metric

        self.emb = embeddings
        if metric == "cosine":
            self.emb = F.normalize(self.emb, p=2, dim=1)

    def retrieve(
        self, query_emb: torch.Tensor, top_k: int = 10
    ) -> Tuple[List[str], List[float], List[int]]:
        if query_emb.device.type != "cpu":
            query_emb = query_emb.cpu()

        if self.metric == "cosine":
            q = F.normalize(query_emb, p=2, dim=0)
            scores = torch.mv(self.emb, q)            # [N]
        elif self.metric == "euclidean":
            diff = self.emb - query_emb
            scores = -torch.norm(diff, dim=1)
        else:
            raise ValueError(self.metric)

        top_k = min(top_k, scores.numel())
        vals, idxs = torch.topk(scores, k=top_k)
        idxs = idxs.tolist()
        return (
            [self.texts[i] for i in idxs],
            vals.tolist(),
            [self.doc_ids[i] for i in idxs],
        )

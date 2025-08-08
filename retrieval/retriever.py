# /retrieval/retriever.py

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from typing import List, Tuple, Literal


# FAISS

from typing import Sequence
from retrieval.FAISSEmbeddingRetriever import FAISSEmbeddingRetriever

def build_retriever(
    embeddings: torch.Tensor,
    texts: Sequence[str],
    doc_ids: Sequence[int],
    cfg: dict,
):
    if cfg.get("backend", "faiss") == "faiss":
        ret = FAISSEmbeddingRetriever(
            embedding_dim=embeddings.size(1),
            index_path=cfg.get("index_path"),
            index_type=cfg.get("index_type", "hnsw"),
            use_gpu=cfg.get("use_gpu", False),
        )
        ret.build(embeddings, texts, doc_ids, train=True)
        return ret
    else:
        from retrieval.bruteforce import BruteForceRetriever
        return BruteForceRetriever(embeddings, texts, doc_ids)

from __future__ import annotations

"""Unit tests for FAISSEmbeddingRetriever and BruteForceRetriever.

Run with::

    pytest -q tests/test_retrievers.py

These tests rely on FAISS (CPU build). They will be skipped automatically if
`import faiss` fails.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

try:
    import faiss  # noqa: F401  # pylint: disable=W0611
except ModuleNotFoundError:  # pragma: no cover – handled by pytest
    faiss = None  # type: ignore

# Project imports ----------------------------------------------------------
from retrieval.FAISSEmbeddingRetriever import FAISSEmbeddingRetriever
from retrieval.bruteforce import BruteForceRetriever

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _make_embeddings(num: int, dim: int = 64, seed: int = 7) -> torch.Tensor:
    """Return L2‑normalised random embeddings on CPU (float32)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return torch.from_numpy(x)


def _build_retrievers(
    embeddings: torch.Tensor,
    texts: List[str],
    doc_ids: List[int],
):
    """Return FAISS (FlatIP) and Brute‑force retrievers over the same corpus."""
    bf = BruteForceRetriever(embeddings.clone(), texts, doc_ids, metric="cosine")
    fr = FAISSEmbeddingRetriever(
        embedding_dim=embeddings.size(1),
        index_type="flatip",
        use_gpu=False,
    )
    fr.build(embeddings.clone(), texts, doc_ids, train=False)
    return fr, bf

# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------


@pytest.mark.skipif(faiss is None, reason="FAISS library not available")
@pytest.mark.parametrize("num_docs, num_queries, dim", [(100, 10, 64), (1000, 50, 32)])
def test_faiss_matches_bruteforce(num_docs: int, num_queries: int, dim: int):
    """Top‑k doc_ids returned by FAISS must match brute‑force for small k=5."""

    k = 5
    emb = _make_embeddings(num_docs, dim)
    texts = [f"doc_{i}" for i in range(num_docs)]
    doc_ids = list(range(num_docs))

    fr, bf = _build_retrievers(emb, texts, doc_ids)

    q_emb = emb[:num_queries]  # use first *num_queries* docs as queries

    for q in q_emb:
        docs_f, _, ids_f = fr.retrieve(q, top_k=k)
        docs_b, _, ids_b = bf.retrieve(q, top_k=k)

        # Compare ordered doc_id lists (they should be identical)
        assert ids_f == ids_b
        # Sanity: returned texts correspond to ids
        assert docs_f == [texts[i] for i in ids_f]
        assert docs_b == [texts[i] for i in ids_b]


@pytest.mark.skipif(faiss is None, reason="FAISS library not available")
def test_faiss_index_persistence(tmp_path: Path):
    """FAISS index should persist to disk and reload with identical results."""

    emb = _make_embeddings(200, 48)
    texts = [f"chunk_{i}" for i in range(200)]
    ids = list(range(200))

    # 1) Build and save index
    idx_path = tmp_path / "test.faiss"
    retr = FAISSEmbeddingRetriever(
        embedding_dim=emb.size(1),
        index_path=idx_path,
        index_type="flatip",
        use_gpu=False,
    )
    retr.build(emb, texts, ids, train=False)

    # 2) Reload new instance from disk
    retr2 = FAISSEmbeddingRetriever(
        embedding_dim=emb.size(1),
        index_path=idx_path,
        index_type="flatip",
        use_gpu=False,
    )
    # No call to .build(): index is mmap‑loaded in constructor

    q = emb[0]
    docs1, scores1, ids1 = retr.retrieve(q, top_k=10)
    docs2, scores2, ids2 = retr2.retrieve(q, top_k=10)

    assert ids1 == ids2
    assert docs1 == docs2
    np.testing.assert_allclose(scores1, scores2, rtol=1e-6)


def test_bruteforce_len_mismatch():
    """BruteForceRetriever must raise if len(texts) ≠ len(doc_ids)."""

    emb = _make_embeddings(10)
    texts = ["a"] * 10
    with pytest.raises(AssertionError):
        BruteForceRetriever(emb, texts, doc_ids=[0, 1])

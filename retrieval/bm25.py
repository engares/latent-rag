from __future__ import annotations

# NOTE: Pyserini dependency removed. This retriever now uses rank-bm25 with
# lazy imports so the heavy dependency is only needed if actually used.
# Add `rank-bm25>=0.2.2` to requeriments.txt if not present.

from retrieval.base import BaseRetriever
from typing import Sequence, List, Callable, Optional
import math, time

Tokenizer = Callable[[str], List[str]]

class BM25Retriever(BaseRetriever):
    """Lightweight BM25 retriever using rank-bm25 (pure Python).

    Differences vs previous version:
      - No temporary Lucene index; fully in-memory.
      - Lazy import of rank_bm25 (only inside build_index()).
      - retrieve() returns List[int] (doc IDs) to match evaluation.benchmark.
    """

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        bm25_variant: str = "okapi",  # okapi | plus
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenizer = tokenizer or (lambda s: s.lower().split())
        self.variant = bm25_variant.lower()
        self.k1 = k1
        self.b = b
        self._bm25 = None  # underlying rank_bm25 object
        self._doc_tokens: List[List[str]] = []
        self._texts: List[str] = []
        self._stats = {
            "variant": self.variant,
            "k1": k1,
            "b": b,
            "n_docs": 0,
            "build_time_s": 0.0,
            "avg_doc_len": 0.0,
        }

    # ------------------------------------------------------------------
    def build_index(self, corpus: Sequence[str]):  # type: ignore[override]
        try:  # lazy import
            if self.variant == "plus":
                from rank_bm25 import BM25Plus as _BM25  # type: ignore
            else:
                from rank_bm25 import BM25Okapi as _BM25  # type: ignore
        except ImportError as e:  # pragma: no cover - env issue
            raise ImportError(
                "rank-bm25 not installed. Add 'rank-bm25>=0.2.2' to requirements."  # noqa: E501
            ) from e

        t0 = time.perf_counter()
        self._texts = list(corpus)
        self._doc_tokens = [self.tokenizer(d) for d in self._texts]
        if not self._doc_tokens:
            raise ValueError("Empty corpus passed to BM25Retriever.build_index")
        self._bm25 = _BM25(self._doc_tokens, k1=self.k1, b=self.b)
        build_time = time.perf_counter() - t0

        # Stats
        total_len = sum(len(toks) for toks in self._doc_tokens)
        self._stats.update({
            "n_docs": len(self._texts),
            "build_time_s": build_time,
            "avg_doc_len": (total_len / max(1, len(self._doc_tokens))),
        })

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int) -> List[int]:  # type: ignore[override]
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_tokens = self.tokenizer(query)
        scores = self._bm25.get_scores(q_tokens)  # np.ndarray[ndocs]
        # Manual partial top-k (avoid full sort for large corpora)
        if k >= len(scores):
            ranked = list(range(len(scores)))
            ranked.sort(key=lambda i: scores[i], reverse=True)
        else:
            # nth_element style: use argsort on partial via enumerate sorting slice
            # For simplicity (k is usually small), just sort all indices.
            ranked = list(range(len(scores)))
            ranked.sort(key=lambda i: scores[i], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    def get_stats(self, reset: bool = False):  # pragma: no cover - simple getter
        return dict(self._stats)

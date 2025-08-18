from retrieval.base import BaseRetriever
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from typing import List, Sequence

class DPRRetriever(BaseRetriever):
    """Dense Passage Retrieval (dual-encoder) returning doc ID lists.

    Now aligns with BM25Retriever and evaluation.benchmark expectations: retrieve(query,k)
    returns List[int] (indices into original corpus list passed to build_index).
    """
    def __init__(self,
                 q_model: str = "facebook-dpr-question_encoder-single-nq-base",
                 p_model: str = "facebook-dpr-ctx_encoder-single-nq-base",
                 device: str | None = None):
        self.q_encoder = SentenceTransformer(q_model, device=device)
        self.p_encoder = SentenceTransformer(p_model, device=device)
        self.index = None
        self.docs: List[str] = []
        self._stats = {"index_type": "hnsw", "efConstruction": 200}

    def build_index(self, corpus: Sequence[str]):  # type: ignore[override]
        self.docs = list(corpus)
        emb = self.p_encoder.encode(self.docs,
                                    batch_size=64,
                                    convert_to_numpy=True,
                                    normalize_embeddings=True)
        d = emb.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(emb.astype("float32"))
        self._stats.update({"n_docs": len(self.docs), "dim": d})

    def retrieve(self, query: str, k: int) -> List[int]:  # type: ignore[override]
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        q_emb = self.q_encoder.encode([query],
                                      convert_to_numpy=True,
                                      normalize_embeddings=True)
        dist, idx = self.index.search(q_emb.astype("float32"), k)
        # idx[0] already are integer positions (doc IDs)
        return idx[0].tolist()

    def get_stats(self, reset: bool = False):  # pragma: no cover
        return dict(self._stats)

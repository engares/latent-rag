from retrieval.base import BaseRetriever
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

class DPRRetriever(BaseRetriever):
    """Dense Passage Retrieval (dual-encoder)."""
    def __init__(self,
                 q_model: str = "facebook-dpr-question_encoder-single-nq-base",
                 p_model: str = "facebook-dpr-ctx_encoder-single-nq-base",
                 device: str | None = None):
        self.q_encoder = SentenceTransformer(q_model, device=device)
        self.p_encoder = SentenceTransformer(p_model, device=device)
        self.index = None
        self.docs  = []

    def build_index(self, corpus):
        self.docs = list(corpus)
        emb = self.p_encoder.encode(self.docs,
                                    batch_size=64,
                                    convert_to_numpy=True,
                                    normalize_embeddings=True)
        d = emb.shape[1]
        self.index = faiss.IndexHNSWFlat(d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(emb.astype("float32"))

    def retrieve(self, query, k):
        q_emb = self.q_encoder.encode([query],
                                      convert_to_numpy=True,
                                      normalize_embeddings=True)
        dist, idx = self.index.search(q_emb.astype("float32"), k)
        return [(self.docs[i], float(-dist[0][j])) for j, i in enumerate(idx[0])]

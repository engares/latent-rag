import pytest
import torch
from retrieval.embedder import EmbeddingCompressor
from retrieval.retriever import compute_similarity, retrieve_top_k

# ------------------- EmbeddingCompressor -------------------
def test_embedding_compressor_without_autoencoder():
    embedder = EmbeddingCompressor()
    texts = ["This is a test.", "Another test."]
    embeddings = embedder.encode_text(texts, compress=False)

    assert embeddings.shape == (2, 384), "Embedding shape mismatch"

def test_embedding_compressor_with_autoencoder():
    class DummyAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def encode(self, x):
            return x * 0.5

    autoencoder = DummyAutoencoder()
    embedder = EmbeddingCompressor(autoencoder=autoencoder)
    texts = ["This is a test.", "Another test."]
    embeddings = embedder.encode_text(texts, compress=True)

    assert embeddings.shape == (2, 384), "Compressed embedding shape mismatch"

# ------------------- Retriever -------------------
def test_compute_similarity():
    queries = torch.randn(2, 384)
    docs = torch.randn(5, 384)

    cosine_sim = compute_similarity(queries, docs, metric="cosine")
    assert cosine_sim.shape == (2, 5), "Cosine similarity shape mismatch"

    euclidean_sim = compute_similarity(queries, docs, metric="euclidean")
    assert euclidean_sim.shape == (2, 5), "Euclidean similarity shape mismatch"

    mahalanobis_sim = compute_similarity(queries, docs, metric="mahalanobis")
    assert mahalanobis_sim.shape == (2, 5), "Mahalanobis similarity shape mismatch"

def test_retrieve_top_k():
    query = torch.randn(384)
    docs = torch.randn(10, 384)
    doc_texts = [f"Document {i}" for i in range(10)]

    top_k = retrieve_top_k(query, docs, doc_texts, k=3, metric="cosine")
    assert len(top_k) == 3, "Top-k retrieval length mismatch"
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_k), "Top-k format mismatch"

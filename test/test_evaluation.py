import pytest
import torch
import numpy as np
from evaluation.retrieval_metrics import recall_at_k, mrr, ndcg_at_k, evaluate_retrieval
from evaluation.generation_metrics import compute_bleu, compute_rouge_l, evaluate_generation_bootstrap
from evaluation.autoencoder_metrics import evaluate_reconstruction_loss

# ------------------- Retrieval Metrics -------------------
def test_recall_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [3, 4, 6]
    assert recall_at_k(retrieved, relevant, k=3) == 1 / 3, "Recall@k mismatch"

def test_mrr():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [3, 4, 6]
    assert mrr(retrieved, relevant) == 1 / 3, "MRR mismatch"

def test_ndcg_at_k():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [3, 4, 6]
    assert np.isclose(ndcg_at_k(retrieved, relevant, k=3), 0.23463936301137822, atol=1e-6), "NDCG@k mismatch"

def test_evaluate_retrieval():
    retrieved_batch = [[1, 2, 3], [4, 5, 6]]
    relevant_batch = [[3, 4], [5, 6]]
    metrics = ["recall@2", "mrr"]
    results = evaluate_retrieval(retrieved_batch, relevant_batch, metrics)
    assert "recall@2" in results and "mrr" in results, "Evaluation metrics missing"

# ------------------- Generation Metrics -------------------
def test_compute_bleu():
    refs = ["this is a test"]
    cands = ["this is a test"]
    assert np.isclose(compute_bleu(cands, refs), 100.0, atol=1e-6), "BLEU mismatch"

def test_compute_rouge_l():
    refs = ["this is a test"]
    cands = ["this is a test"]
    assert compute_rouge_l(cands, refs) == 100.0, "ROUGE-L mismatch"

def test_evaluate_generation_bootstrap():
    refs = ["this is a test"] * 30
    cands = ["this is a test"] * 30
    results = evaluate_generation_bootstrap(refs, cands, metrics=["BLEU", "ROUGE-L"], n_samples=100)
    assert "BLEU" in results and "ROUGE-L" in results, "Bootstrap metrics missing"

# ------------------- Autoencoder Metrics -------------------
def test_evaluate_reconstruction_loss():
    x = torch.randn(10, 16)
    x_reconstructed = x + torch.randn(10, 16) * 0.1
    loss = evaluate_reconstruction_loss(x, x_reconstructed, reduction="mean")
    assert loss > 0, "Reconstruction loss should be positive"

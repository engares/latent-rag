import pytest
import torch
import os
from evaluation.embedding_visualization import visualize_compressed_vs_original

# Small synthetic dataset: 10 query–doc pairs
@pytest.fixture
def dummy_embeddings():
    torch.manual_seed(42)
    N, D = 10, 384
    q_orig = torch.randn(N, D)
    d_orig = torch.randn(N, D)

    # compressed = slight noise + projection
    q_comp = q_orig[:, :64] + 0.01 * torch.randn(N, 64)
    d_comp = d_orig[:, :64] + 0.01 * torch.randn(N, 64)

    return q_orig, d_orig, q_comp, d_comp


def test_visualization_outputs(tmp_path, dummy_embeddings):
    q_orig, d_orig, q_comp, d_comp = dummy_embeddings

    save_path = tmp_path / "viz.png"
    neg_path  = tmp_path / "viz_neg.png"

    metrics = visualize_compressed_vs_original(
        q_orig,
        d_orig,
        q_comp,
        d_comp,
        projection="pca",
        n_components=2,
        sample_size=10,
        k_near=3,
        bins=5,
        random_state=42,
        save_path=str(save_path),
        save_negatives_path=str(neg_path),
    )

    # Check files exist
    assert save_path.exists(), "Main figure not saved"
    assert neg_path.exists(), "Negative distribution figure not saved"

    # Check metrics returned
    assert isinstance(metrics, dict), "Output is not a dictionary"
    assert "recall_original" in metrics
    assert "recall_compressed" in metrics

    # Check reasonable range
    assert 0.0 <= metrics["recall_original"] <= 1.0
    assert 0.0 <= metrics["recall_compressed"] <= 1.0

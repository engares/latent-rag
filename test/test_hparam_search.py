# tests/test_hparam_search.py
import os
import re
import csv
import json
import types
import shutil
import importlib
import sys
from pathlib import Path

import pytest
import torch
import optuna


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    """Force CPU so the test runs anywhere."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


class _DummyLogger:
    def __init__(self):
        # emulate .main.info/.debug/... usage in your code
        self.main = self

    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _write_yaml(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _mock_train(tmpdir):
    """
    Returns a fake trainer that:
      - Prunes trial #1 (by parsing the trial number from model_save_path)
      - Creates a checkpoint file
      - Writes a matching meta JSON with best_val_loss
    """
    def train_dae(**kwargs):
        ckpt_path = kwargs["model_save_path"]
        # parse trial number from "..._t{num}_..."
        m = re.search(r"_t(\d+)_", Path(ckpt_path).name)
        trial_num = int(m.group(1)) if m else 0

        # simulate a prune on trial #1 to exercise pruning path
        if trial_num == 1:
            # Signal some progress before pruning
            cb = kwargs.get("report_cb")
            if cb:
                for ep in range(3):
                    try:
                        cb(ep, 1.0, 1.0 - 0.01 * ep, kwargs.get("lr", 1e-3))
                    except optuna.TrialPruned:
                        raise
            raise optuna.TrialPruned("synthetic prune")

        # "successful" trial: write ckpt + meta
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(ckpt_path).write_bytes(b"FAKE_WEIGHTS")

        stem = Path(ckpt_path).with_suffix("").name
        meta_dir = tmpdir / "models" / "history"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{stem}.json"

        # make best_val_loss depend on latent_dim so trials rank deterministically
        latent = int(kwargs.get("latent_dim", 64))
        best_val = 1.0 / max(1, latent)  # smaller is better

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"best_val_loss": best_val}, f)

        return ckpt_path

    return train_dae


def _mock_data_loader():
    """
    Returns small, deterministic eval data.
    - queries[i] is relevant to corpus_docs[i]
    """
    def load_evaluation_data(dataset_name: str, max_samples: int = 2000):
        n = min(12, max_samples)
        queries = [f"q{i}" for i in range(n)]
        docs =    [f"d{i}" for i in range(n)]
        relevant = [[docs[i]] for i in range(n)]  # identity relevance
        return queries, docs, relevant
    return load_evaluation_data


def _mock_chunker():
    # We'll keep chunking disabled in config, but provide a stub anyway.
    def prepare_inference_chunks(docs, **kwargs):
        import pandas as pd
        # identity mapping: one "chunk" per doc
        df = pd.DataFrame({"doc_id": list(range(len(docs)))})
        return list(docs), df
    return prepare_inference_chunks


def _mock_embedder():
    class FakeEC:
        def __init__(self, base_model_name, autoencoder, device):
            self.base_model_name = base_model_name
            self.autoencoder = autoencoder
            self.device = device
        def encode_text(self, texts, compress=False):
            # Return SBERT-sized embeddings when compress=False
            D = 384 if not compress else 96
            return torch.randn(len(texts), D)
    return FakeEC


def _mock_pipeline_runner():
    class FakeRunner:
        def __init__(
            self, cfg, model, log,
            pre_corpus_texts, pre_corpus_doc_ids,
            base_corpus_embeddings, base_query_embeddings,
            checkpoint_override=None
        ):
            # sanity checks similar to your real runner
            assert isinstance(pre_corpus_texts, list)
            assert len(pre_corpus_texts) == len(pre_corpus_doc_ids)
            assert isinstance(base_corpus_embeddings, torch.Tensor)
            assert isinstance(base_query_embeddings, torch.Tensor)
            self.cfg = cfg
            self.model = model
            self.checkpoint = checkpoint_override
            self.log = log

        def process(self, queries, corpus, relevant_docs=None, generate=False):
            # deterministic "good" metrics; primary is ndcg@10 by default
            return {
                "retrieval_metrics": {
                    "Recall@10": {"mean": 0.80, "std": 0.01},
                    "MRR@10":    {"mean": 0.60, "std": 0.01},
                    "nDCG@10":   {"mean": 0.70, "std": 0.01},
                },
                "retriever_stats": {},
                "dim_in": 384,
                "dim_out": 96,
                "n_corpus": len(corpus),
                "n_queries": len(queries),
                "ae_type": "dae",
            }
    return FakeRunner


def test_hparam_search_end_to_end(tmp_path, monkeypatch, capsys):
    """
    Hard integration-style test that:
      * runs the Optuna loop with 3 trials (1 pruned),
      * writes checkpoints + meta,
      * creates trials CSV,
      * runs retrieval re-rank via a mocked PipelineRunner,
      * writes retrieval CSV,
      * creates bm_ and bmret_ copies.
    """
    # Work inside an isolated cwd so relative paths like 'models/history' go here.
    monkeypatch.chdir(tmp_path)

    # --- Write minimal config & sweep ---
    cfg_path = tmp_path / "config" / "config.yaml"
    _write_yaml(
        cfg_path,
        """
paths:
  checkpoints_dir: "models/checkpoints"
logging: {}
data:
  dataset: "toyset"
embedding_model:
  name: "dummy-model"
chunking:
  enabled: false
training:
  seed: 123
  deterministic: false
  epochs: 3
  batch_size: 8
  learning_rate: 0.001
retrieval:
  top_k: 10
evaluation:
  retrieval_metrics: ["Recall@10","MRR@10","nDCG@10"]
        """.strip()
    )
    sweep_path = tmp_path / "sweeps" / "cae.yaml"
    _write_yaml(
        sweep_path,
        """
space:
  latent_dim:      {type: int,   low: 32, high: 96, step: 64}
  hidden_dim:      {type: int,   low: 256, high: 1024, step: 768}
  lr:              {type: float, low: 0.0005, high: 0.001, log: false}
  batch_size:      {type: categorical, choices: [8, 16]}
  # the rest are optional; defaults in code will fill them if absent
constants: {}
        """.strip()
    )

    # --- Import the module under test ---
    import hparam_search
    importlib.reload(hparam_search)  # ensure a clean module state

    # --- Patch utilities & heavy deps with deterministic fakes ---
    monkeypatch.setattr(hparam_search, "init_logger", lambda *_: _DummyLogger())
    monkeypatch.setattr(hparam_search, "load_config", lambda _: {
        # We still return a parsed config object (not re-reading YAML) to keep it simple.
        "paths": {"checkpoints_dir": "models/checkpoints"},
        "logging": {},
        "data": {"dataset": "toyset"},
        "embedding_model": {"name": "dummy-model"},
        "chunking": {"enabled": False},
        "training": {"seed": 123, "deterministic": False, "epochs": 3, "batch_size": 8, "learning_rate": 0.001},
        "retrieval": {"top_k": 10},
        "evaluation": {"retrieval_metrics": ["Recall@10", "MRR@10", "nDCG@10"]},
    })
    monkeypatch.setattr(hparam_search, "set_seed", lambda *a, **k: None)
    monkeypatch.setattr(hparam_search, "prepare_datasets", lambda *a, **k: "DATASET_OK")

    # Eval-data path (aligned with your current code path)
    monkeypatch.setattr(hparam_search, "load_evaluation_data", _mock_data_loader())
    monkeypatch.setattr(hparam_search, "prepare_inference_chunks", _mock_chunker())
    monkeypatch.setattr(hparam_search, "EmbeddingCompressor", _mock_embedder())
    monkeypatch.setattr(hparam_search, "PipelineRunner", _mock_pipeline_runner())

    # Trainer
    fake_trainer = _mock_train(tmp_path)
    monkeypatch.setattr(hparam_search, "train_dae", fake_trainer)
    # Keep registry consistent
    hparam_search.REGISTRY["dae"] = fake_trainer

    # --- Environment knobs for the re-rank phase ---
    monkeypatch.setenv("HPO_EVAL_TOPK", "2")         # re-evaluate top-2 finishing trials
    monkeypatch.setenv("HPO_SAMPLE_QUERIES", "10")   # small but > 0
    monkeypatch.setenv("HPO_PRIMARY", "ndcg@10")     # match mocked metrics

    # --- Run main with fixed study name for predictable file names ---
    argv = [
        "prog",
        "--config", str(cfg_path),
        "--model", "dae",
        "--sweep", str(sweep_path),
        "--n_trials", "3",
        "--study", "unittest",
        "--pruner", "none",         # we'll prune one trial manually inside the trainer
        "--sampler", "random",
    ]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(sys, "argv", argv)

    # Execute
    hparam_search.main()

    # --- Assertions: trials CSV exists and is well-formed ---
    trials_csv = tmp_path / "models" / "history" / "hpo_unittest.csv"
    assert trials_csv.exists(), f"Missing trials CSV: {trials_csv}"
    rows = list(csv.DictReader(trials_csv.open()))
    # 3 trials were scheduled, 1 was pruned → at least 2 rows present (Optuna logs all trials)
    assert len(rows) == 3
    # At least one COMPLETE trial with a numeric value
    assert any(r["state"].endswith("COMPLETE") and r["value"] for r in rows)

    # --- bm_ copy should exist for the best val_loss trial ---
    ckpt_dir = tmp_path / "models" / "checkpoints"
    bm_candidates = list(ckpt_dir.glob("bm_*.pth"))
    assert bm_candidates, "Best-model (bm_) copy not created."

    # --- Retrieval re-rank CSV exists and contains the primary metric ---
    ret_csv = tmp_path / "models" / "history" / "hpo_dae_unittest_retrieval.csv"
    assert ret_csv.exists(), f"Missing retrieval CSV: {ret_csv}"
    ret_rows = list(csv.DictReader(ret_csv.open()))
    assert len(ret_rows) >= 1
    # Columns should include the lower-cased metrics keys used in the code
    header = ret_rows[0].keys()
    assert "ndcg@10" in header and "recall@10" in header and "mrr@10" in header

    # --- bmret_ copy should exist for the best retrieval model ---
    bmret_candidates = list(ckpt_dir.glob("bmret_*.pth"))
    assert bmret_candidates, "Best-retrieval (bmret_) copy not created."

    # --- Sanity: stdout mentions the two CSVs (helps catch regressions) ---
    out = capsys.readouterr().out
    assert "[HPO] Trials CSV:" in out
    assert "[HPO][RET] Metrics CSV:" in out
    assert "Best by ndcg@10" in out

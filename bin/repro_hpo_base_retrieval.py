#!/usr/bin/env python3
"""
Reproduce hparam_search post-eval retrieval table for the 'base' model without retraining.

It evaluates the provided checkpoints with the exact retrieval setup used in hparam_search
(pre-chunked corpus reuse, base SBERT embeddings reuse, PRIMARY metric @K control) and
writes a CSV with columns: ckpt,mrr@K,ndcg@K,recall@K,trial

Usage examples:
  - Default (uses hardcoded ckpt list below):
      python bin/repro_hpo_base_retrieval.py
  - With custom list from CSV (headers: ckpt,trial):
      python bin/repro_hpo_base_retrieval.py --ckpt_csv /path/to/list.csv --out models/history/repro.csv

Environment variables (same defaults as hparam_search):
  HPO_PRIMARY=ndcg@10
  HPO_EVAL_TOPK=5            (only impacts selection in hparam_search; here we just set metric@K)
  HPO_SAMPLE_QUERIES=2000
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from typing import List, Dict, Any

import torch

from utils.load_config import load_config, init_logger
from utils.data_utils import load_evaluation_data, prepare_inference_chunks
from main import PipelineRunner
from retrieval.embedder import EmbeddingCompressor


# Default list (exact ckpts and trials from the provided table)
DEFAULT_CKPTS = [
    {"ckpt": "./models/checkpoints/base_20250825_224508_t10_lat192_hid1024_lr0.000558009_bs128_tr0.0008_val0.0006.pth", "trial": 10},
    {"ckpt": "./models/checkpoints/base_20250825_224508_t24_lat128_hid768_lr0.00081828_bs256_tr0.0010_val0.0007.pth",  "trial": 24},
    {"ckpt": "./models/checkpoints/base_20250825_224508_t0_lat96_hid1024_lr0.000945431_bs128_tr0.0009_val0.0008.pth",   "trial": 0},
    {"ckpt": "./models/checkpoints/base_20250825_224508_t11_lat96_hid1024_lr0.000266354_bs128_tr0.0010_val0.0008.pth",  "trial": 11},
    {"ckpt": "./models/checkpoints/base_20250825_224508_t3_lat96_hid256_lr0.000702626_bs512_tr0.0008_val0.0008.pth",    "trial": 3},
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Reproduce HPO retrieval metrics for base model")
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--ckpt_csv", default=None, help="Optional CSV with columns: ckpt,trial")
    ap.add_argument("--out", default=None, help="Output CSV path; default mirrors hparam_search naming in models/history")
    ap.add_argument("--dataset", choices=["squad"], default=None, help="Override dataset (defaults to cfg.data.dataset)")
    ap.add_argument("--max_samples", type=int, default=None, help="Override HPO_SAMPLE_QUERIES")
    return ap.parse_args()


def _load_ckpt_list(path: str | None) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_CKPTS
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("ckpt"):
                continue
            trial = row.get("trial")
            try:
                t = int(float(trial)) if trial is not None and str(trial).strip() != "" else -1
            except Exception:
                t = -1
            rows.append({"ckpt": row["ckpt"], "trial": t})
    return rows


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    log = init_logger(cfg.get("logging", {}))

    # PRIMARY metric and K (matches hparam_search defaults)
    PRIMARY = os.getenv("HPO_PRIMARY", "ndcg@10")
    mK = re.search(r"@(\d+)$", PRIMARY)
    primary_k = int(mK.group(1)) if mK else 10
    SAMPLE_Q = int(os.getenv("HPO_SAMPLE_QUERIES", str(args.max_samples or 2000)))

    # Dataset & chunking
    dataset_name = (args.dataset or cfg.get("data", {}).get("dataset", "squad")).lower()
    queries, corpus_docs, relevant = load_evaluation_data(dataset_name, max_samples=SAMPLE_Q)

    ch_cfg = cfg.get("chunking", {})
    use_chunking = bool(ch_cfg.get("enabled", False))
    corpus_texts = corpus_docs
    corpus_doc_ids: List[int]
    if use_chunking:
        chunks, chunk_index = prepare_inference_chunks(
            corpus_texts,
            mode=ch_cfg.get("mode", "sliding"),
            max_tokens=ch_cfg.get("max_tokens", 128),
            stride=ch_cfg.get("stride", 64),
            min_tokens=ch_cfg.get("min_tokens", 48),
            tokenizer_name=ch_cfg.get("tokenizer_name", cfg["embedding_model"]["name"]),
            index_out=ch_cfg.get("index_out"),
            store_chunk_text=ch_cfg.get("store_chunk_text", True),
        )
        corpus_texts = chunks
        corpus_doc_ids = chunk_index["doc_id"].astype(int).tolist()
    else:
        corpus_doc_ids = list(range(len(corpus_texts)))

    # Base compressor and base embeddings (no AE)
    base_compressor = EmbeddingCompressor(
        base_model_name=cfg["embedding_model"]["name"],
        autoencoder=None,
        device=cfg.get("training", {}).get("device") or ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    with torch.inference_mode():
        base_corpus_emb = base_compressor.encode_text(list(corpus_texts), compress=False)
        base_query_emb = base_compressor.encode_text(list(queries), compress=False)

    # cfg_eval aligned to PRIMARY's @K like in hparam_search
    cfg_eval = {k: v for k, v in cfg.items()}
    cfg_eval.setdefault("evaluation", {})
    cfg_eval["evaluation"]["retrieval_metrics"] = [f"Recall@{primary_k}", f"MRR@{primary_k}", f"nDCG@{primary_k}"]
    cfg_eval.setdefault("retrieval", {})
    cfg_eval["retrieval"]["top_k"] = int(primary_k)
    if use_chunking:
        cur = int(cfg_eval["retrieval"].get("candidate_k", primary_k * 3))
        cfg_eval["retrieval"]["candidate_k"] = int(max(cur, primary_k))

    rows = _load_ckpt_list(args.ckpt_csv)

    # Output CSV path
    out_csv = args.out or os.path.join(
        cfg.get("paths", {}).get("history_dir", "./models/history"),
        f"repro_base_{PRIMARY}_retrieval.csv",
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Fixed header/order to match provided table
    header = ["ckpt", f"mrr@{primary_k}", f"ndcg@{primary_k}", f"recall@{primary_k}", "trial"]

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for item in rows:
            ckpt = item["ckpt"]
            trial = item.get("trial", -1)
            try:
                runner = PipelineRunner(
                    cfg_eval, "base", log,
                    pre_corpus_texts=corpus_texts,
                    pre_corpus_doc_ids=corpus_doc_ids,
                    base_corpus_embeddings=base_corpus_emb,
                    base_query_embeddings=base_query_emb,
                    checkpoint_override=ckpt,
                )
                # IMPORTANT: replicate hparam_search call (pass corpus_texts)
                result = runner.process(
                    queries,
                    corpus_texts,
                    relevant_docs=relevant,
                    generate=False,
                )
                retm = result.get("retrieval_metrics", {})
                metrics = {}
                for k, v in retm.items():
                    kl = k.lower()
                    try:
                        metrics[kl] = float(v.get("mean", 0.0))
                    except Exception:
                        try:
                            metrics[kl] = float(v)
                        except Exception:
                            pass
                row = {
                    "ckpt": ckpt,
                    f"mrr@{primary_k}": metrics.get(f"mrr@{primary_k}", 0.0),
                    f"ndcg@{primary_k}": metrics.get(f"ndcg@{primary_k}", 0.0),
                    f"recall@{primary_k}": metrics.get(f"recall@{primary_k}", 0.0),
                    "trial": trial,
                }
                w.writerow(row)
                print(f"[OK] {ckpt} → MRR@{primary_k}={row[f'mrr@{primary_k}']:.6f} NDCG@{primary_k}={row[f'ndcg@{primary_k}']:.6f} Recall@{primary_k}={row[f'recall@{primary_k}']:.6f}")
            except Exception as e:
                print(f"[FAIL] {ckpt}: {e}")

    print("[DONE] CSV:", out_csv)


if __name__ == "__main__":
    main()

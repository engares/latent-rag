# utils/benchmark_utils.py
from __future__ import annotations
import os, json
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, List
import numpy as np
import csv

def percentile(values: Sequence[float], q: float) -> float:
    """Return q-th percentile of a list of floats."""
    if not values:
        return float("nan")
    return float(np.percentile(values, q))

def build_metrics_row(
    cfg: Dict[str, Any],
    args: Any,
    ae: str,
    result: Dict[str, Any],
    *,
    baseline_dir: str = "logs/benchmarks",
) -> Dict[str, Any]:
    """Build a CSV row from a retrieval benchmark result.

    Also updates or reads the baseline file for speedup/delta calculation.

    Returns:
        Dict ready to be appended to a CSV.
    """
    ret = result.get("retrieval_metrics", {})

    def _m(name: str) -> float:
        d = ret.get(name) or {}
        return float(d.get("mean")) if "mean" in d else float("nan")

    stats = result.get("retriever_stats", {})
    perq = sorted(stats.get("per_query_ms", []))
    p50 = percentile(perq, 50.0)
    p95 = percentile(perq, 95.0)
    qps = (1000.0 / p50) if p50 and p50 > 0 else float("nan")

    dim_in = int(result.get("dim_in", 0))
    dim_out = int(result.get("dim_out", 0))
    cr = (float(dim_in) / float(dim_out)) if dim_out else float("nan")

    retr_cfg = cfg.get("retrieval", {})
    embm = cfg.get("embedding_model", {})
    data = cfg.get("data", {})

    # Create the row dictionary directly with optimized logic
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": getattr(args, "benchmark_tag", None),
        "dataset": data.get("dataset", getattr(args, "dataset", None)),
        "split": "validation",
        "max_samples": int(data.get("max_samples", getattr(args, "max_samples", 0))),
        "embedder": embm.get("name", "?"),
        "ae_type": ae,
        "latent_dim": dim_out,
        "dim_in": dim_in,
        "compression_ratio": cr,
        "retriever": retr_cfg.get("backend", "faiss"),
        "index_type": retr_cfg.get("index_type", "hnsw"),
        "use_gpu": bool(retr_cfg.get("use_gpu", False)),
        "top_k": int(retr_cfg.get("top_k", 10)),
        "candidate_k": int(retr_cfg.get("candidate_k", 10)),
        "n_corpus": int(result.get("n_corpus", 0)),
        "Recall@10": _m("Recall@10"),
        "MRR@10": _m("MRR@10"),
        "nDCG@10": _m("nDCG@10"),
        "build_time_s": float(stats.get("build_time_s", 0.0)),
        "search_time_s": float(stats.get("search_time_s", 0.0)),
        "search_calls": int(stats.get("search_calls", 0)),
        "query_p50_ms": p50,
        "query_p95_ms": p95,
        "qps": qps,
    }

    # --- baseline speedup calc
    baseline_key = f"{row['dataset']}_{row['split']}_{row['embedder']}_{row['retriever']}_{row['index_type']}_k{row['top_k']}"
    baseline_path = os.path.join(baseline_dir, f"baseline_{baseline_key}.json")

    if ae == "none":
        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({"p50": p50, "p95": p95}, f, indent=2)
    elif os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            base = json.load(f)
        bp50 = float(base.get("p50", float("nan")))
        bp95 = float(base.get("p95", float("nan")))
        row.update({
            "speedup_p50": (bp50 / p50) if p50 and p50 > 0 else float("nan"),
            "speedup_p95": (bp95 / p95) if p95 and p95 > 0 else float("nan"),
            "delta_ms_p50": (bp50 - p50) if not (np.isnan(bp50) or np.isnan(p50)) else float("nan"),
            "delta_ms_p95": (bp95 - p95) if not (np.isnan(bp95) or np.isnan(p95)) else float("nan"),
        })

    return row


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Devuelve el percentil p (0–100) sobre una lista YA ordenada; NaN si vacía."""
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    i = min(max(int(round(p / 100.0 * (n - 1))), 0), n - 1)
    return sorted_vals[i]

def _append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Append-only CSV; crea cabecera si el archivo no existe."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

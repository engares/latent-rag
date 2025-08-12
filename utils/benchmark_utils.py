# utils/benchmark_utils.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ------------------------------- helpers ---------------------------------- #

def percentile(values: Sequence[float], q: float) -> float:
    """Return the q-th percentile for a list of floats (NaN if empty)."""
    if not values:
        return float("nan")
    return float(np.percentile(values, q))


def _append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Append a dict row to CSV, creating header on first write."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Safe nested get: _cfg_get(cfg, "chunking", "enabled", default=False)."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_ckpt_path(checkpoint: Optional[str], cfg_paths: Mapping[str, Any]) -> Optional[Path]:
    """Resolve checkpoint to absolute Path using paths.checkpoints_dir when relative."""
    if not checkpoint:
        return None
    p = Path(checkpoint)
    if p.is_absolute():
        return p
    base = Path(_cfg_get(cfg_paths, "checkpoints_dir", default="./models/checkpoints"))
    return (base / p).resolve()


def _size_mb(path: Path) -> Optional[float]:
    try:
        return float(path.stat().st_size) / 1_000_000.0
    except Exception:
        return None


def _probe_index_size_mb(cfg: Mapping[str, Any]) -> Optional[float]:
    """Best-effort FAISS index size (only if persisted)."""
    # common locations
    explicit = _cfg_get(cfg, "retrieval", "index_path")
    if explicit:
        return _size_mb(Path(str(explicit)))
    # default repo layout: data/index/faiss_chunks.faiss
    candidate = Path("data/index/faiss_chunks.faiss")
    if candidate.exists():
        return _size_mb(candidate)
    return None


def _probe_embedding_sizes_mb(dataset: str, ae: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (emb_disk_mb_original, emb_disk_mb_compressed, storage_saving_pct)."""
    # Repo layout observed in tree:
    # data/SQUAD/sbert_cache/*.pt    (original SBERT)
    # data/SQUAD/squad_{vae|dae|contrastive}_embeddings.pt (compressed)
    root = Path("data") / dataset.upper()

    # original: sum of cache files if present
    orig_mb: Optional[float] = None
    cache_dir = root / "sbert_cache"
    if cache_dir.exists():
        total = 0.0
        for p in cache_dir.glob("*.pt"):
            sz = _size_mb(p)
            if sz:
                total += sz
        orig_mb = total if total > 0 else None

    # compressed: single file per AE (if it exists)
    ae_map = {"vae": "vae", "dae": "dae", "contrastive": "contrastive"}
    comp_mb: Optional[float] = None
    if ae in ae_map:
        comp_path = root / f"{dataset.lower()}_{ae_map[ae]}_embeddings.pt"
        if comp_path.exists():
            comp_mb = _size_mb(comp_path)

    saving_pct: Optional[float] = None
    if orig_mb and comp_mb:
        saving_pct = max(0.0, (1.0 - (comp_mb / orig_mb)) * 100.0)

    return orig_mb, comp_mb, saving_pct


def _extract_retriever_metric(cfg: Mapping[str, Any]) -> Tuple[str, bool]:
    """Infer (metric, normalize_l2) from retrieval config/backends used in this repo."""
    backend = str(_cfg_get(cfg, "retrieval", "backend", default="faiss")).lower()
    metric_cfg = _cfg_get(cfg, "retrieval", "metric")  # optional for bruteforce
    if backend == "faiss":
        # Our FAISS uses IP + L2 normalisation (cosine) by design.
        return "ip", True
    # BruteForce default is cosine in our implementation.
    metric = (metric_cfg or "cosine").lower()
    normalize = bool(metric == "cosine")
    return metric, normalize


def _extract_ae_hparams(cfg: Mapping[str, Any], ae: str) -> Dict[str, Any]:
    """Collect AE hyper-parameters from config and optional sidecar metadata."""
    mcfg = _cfg_get(cfg, "models", ae, default={}) or {}
    ckpt_rel = mcfg.get("checkpoint")
    ckpt_abs = _resolve_ckpt_path(ckpt_rel, _cfg_get(cfg, "paths", default={}) or {})
    meta = {"epochs_trained": None, "early_stop_epoch": None}
    if ckpt_abs:
        meta_path = ckpt_abs.with_suffix(ckpt_abs.suffix + ".meta.json")
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    j = json.load(f)
                meta["epochs_trained"] = j.get("epochs_trained")
                meta["early_stop_epoch"] = j.get("early_stop_epoch")
            except Exception:
                pass

    beta = None
    if ae == "vae":
        beta = _cfg_get(cfg, "models", "vae", "beta", default=1.0)

    return {
        "ae_checkpoint": str(ckpt_abs) if ckpt_abs else ckpt_rel,
        "hidden_dim": mcfg.get("hidden_dim"),
        "beta": beta,
        "epochs_trained": meta["epochs_trained"],
        "early_stop_epoch": meta["early_stop_epoch"],
    }


# ------------------------------ public API -------------------------------- #

def build_metrics_row(
    cfg: Dict[str, Any],
    args: Any,
    ae: str,
    result: Dict[str, Any],
    *,
    baseline_dir: str = "logs/benchmarks",
) -> Dict[str, Any]:
    """Build a flat CSV row from a retrieval/generation benchmark result.

    It also updates/reads a baseline JSON (for ae='none') to compute speedups.
    """
    # --- retrieval metrics summary
    ret = result.get("retrieval_metrics", {}) or {}
    def _m(name: str) -> float:
        d = ret.get(name) or {}
        return float(d.get("mean")) if "mean" in d else float("nan")

    stats = result.get("retriever_stats", {}) or {}
    perq = sorted(stats.get("per_query_ms", []))
    p50 = percentile(perq, 50.0)
    p95 = percentile(perq, 95.0)
    qps = (1000.0 / p50) if p50 and p50 > 0 else float("nan")

    dim_in = int(result.get("dim_in", 0))
    dim_out = int(result.get("dim_out", 0))
    cr = (float(dim_in) / float(dim_out)) if dim_out else float("nan")

    retr_cfg = cfg.get("retrieval", {}) or {}
    embm = cfg.get("embedding_model", {}) or {}
    data = cfg.get("data", {}) or {}
    ch = cfg.get("chunking", {}) or {}

    metric, normalize_l2 = _extract_retriever_metric(cfg)
    ae_h = _extract_ae_hparams(cfg, ae)

    # Disk sizes (index + embeddings)
    index_size_mb = _probe_index_size_mb(cfg)
    emb_orig_mb, emb_comp_mb, saving_pct = _probe_embedding_sizes_mb(
        dataset=str(data.get("dataset", getattr(args, "dataset", "squad"))),
        ae=ae,
    )

    # Optional generation metrics (if runner provided them)
    gen = result.get("generation_metrics", {}) or {}
    rougeL = gen.get("rougeL", {}).get("mean")
    rougeL_lo = gen.get("rougeL", {}).get("ci_lower")
    rougeL_hi = gen.get("rougeL", {}).get("ci_upper")
    bleu = gen.get("bleu", {}).get("mean")
    meteor = gen.get("meteor", {}).get("mean")

    # chunking
    chunking_enabled = bool(ch.get("enabled", False))
    row: Dict[str, Any] = {
        # identity
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": getattr(args, "benchmark_tag", None),
        "dataset": data.get("dataset", getattr(args, "dataset", None)),
        "split": "validation",
        "max_samples": int(data.get("max_samples", getattr(args, "max_samples", 0))),

        # embedder & AE
        "embedder": embm.get("name", "?"),
        "ae_type": ae,
        "latent_dim": dim_out,
        "dim_in": dim_in,
        "compression_ratio": cr,
        "ae_checkpoint": ae_h.get("ae_checkpoint"),
        "hidden_dim": ae_h.get("hidden_dim"),
        "beta": ae_h.get("beta"),
        "epochs_trained": ae_h.get("epochs_trained"),
        "early_stop_epoch": ae_h.get("early_stop_epoch"),

        # retrieval configuration (incl. metric/normalisation)
        "retriever": retr_cfg.get("backend", "faiss"),
        "index_type": retr_cfg.get("index_type", "hnsw"),
        "metric": metric,
        "normalize_l2": bool(normalize_l2),
        "use_gpu": bool(retr_cfg.get("use_gpu", False)),
        "top_k": int(retr_cfg.get("top_k", 10)),
        "candidate_k": int(retr_cfg.get("candidate_k", 10)),
        "n_corpus": int(result.get("n_corpus", 0)),

        # chunking footprint
        "chunking_enabled": chunking_enabled,
        "chunk_mode": ch.get("mode") if chunking_enabled else None,
        "chunk_max_tokens": ch.get("max_tokens") if chunking_enabled else None,
        "chunk_stride": ch.get("stride") if chunking_enabled else None,
        "chunk_min_tokens": ch.get("min_tokens") if chunking_enabled else None,

        # accuracy
        "Recall@10": _m("Recall@10"),
        "MRR@10": _m("MRR@10"),
        "nDCG@10": _m("nDCG@10"),

        # latency/throughput
        "build_time_s": float(stats.get("build_time_s", 0.0)),
        "search_time_s": float(stats.get("search_time_s", 0.0)),
        "search_calls": int(stats.get("search_calls", 0)),
        "query_p50_ms": p50,
        "query_p95_ms": p95,
        "qps": qps,

        # disk/compression & ablations
        "index_size_mb": index_size_mb,
        "emb_disk_mb_original": emb_orig_mb,
        "emb_disk_mb_compressed": emb_comp_mb,
        "storage_saving_pct": saving_pct,

        # generation (optional)
        "rougeL": rougeL,
        "rougeL_ci95_low": rougeL_lo,
        "rougeL_ci95_high": rougeL_hi,
        "bleu": bleu,
        "meteor": meteor,
    }

    # --- baseline speedup calc
    baseline_key = f"{row['dataset']}_{row['split']}_{row['embedder']}_{row['retriever']}_{row['index_type']}_k{row['top_k']}"
    baseline_path = os.path.join(baseline_dir, f"baseline_{baseline_key}.json")

    if ae == "none":
        Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
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
            "baseline_key": baseline_key,
        })

    return row

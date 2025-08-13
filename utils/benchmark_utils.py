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
    """Append a dict row to CSV, creating header on first write or if file is empty."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or (path.stat().st_size == 0)
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

    # ------------------------------------------------------------------
    # Size estimation (optimized): use ONLY theoretical sizes.z
    # ------------------------------------------------------------------
    index_size_mb = None  # no disk probe
    n_corpus = int(result.get("n_corpus", 0))
    dim_in = int(result.get("dim_in", 0))  # may overwrite earlier local dim_in (kept consistent)
    dim_out = int(result.get("dim_out", 0))
    orig_dtype = _cfg_get(cfg, "embedding_model", "dtype", default="float32")
    comp_dtype = _cfg_get(cfg, "models", ae, "dtype", default="float32")
    emb_orig_mb = emb_comp_mb = saving_pct = float("nan")
    if n_corpus > 0 and dim_in > 0 and dim_out > 0:
        emb_orig_mb, emb_comp_mb, saving_pct = compute_per_run_sizes(
            n_corpus=n_corpus,
            dim_in=dim_in,
            dim_out=dim_out,
            orig_dtype=orig_dtype,
            comp_dtype=comp_dtype,
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
    # safeguard numeric casts
    max_samples_val = data.get("max_samples", getattr(args, "max_samples", 0))
    if max_samples_val is None:
        max_samples_val = getattr(args, "max_samples", 0) or 0
    # build row
    row: Dict[str, Any] = {
        # identity
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": getattr(args, "benchmark_tag", None),
        "dataset": data.get("dataset", getattr(args, "dataset", None)),
        "split": "validation",
        "max_samples": _safe_int(max_samples_val, 0),
        "n_queries": _safe_int(result.get("n_queries")),  # NEW: actual evaluated query count

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
        "top_k": _safe_int(retr_cfg.get("top_k", 10), 10),
        "candidate_k": _safe_int(retr_cfg.get("candidate_k", 10), 10),
        "n_corpus": _safe_int(result.get("n_corpus", 0), 0),

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
        "build_time_s": _safe_float(stats.get("build_time_s", 0.0), 0.0),
        "search_time_s": _safe_float(stats.get("search_time_s", 0.0), 0.0),
        "search_calls": _safe_int(stats.get("search_calls", 0), 0),
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

# utils/benchmark_utils.py (añade después de los imports)

def _bytes_per_elem(dtype: Any) -> int:
    """Return bytes per element for a given dtype label or numpy dtype."""
    # Acepta strings o np.dtype; default razonable = float32
    mapping = {
        "float32": 4, "float": 4, np.float32: 4,
        "float16": 2, "half": 2, np.float16: 2,
        "float64": 8, "double": 8, np.float64: 8,
        "int8": 1, np.int8: 1, "uint8": 1, np.uint8: 1,
        "int16": 2, np.int16: 2, "uint16": 2, np.uint16: 2,
        "int32": 4, np.int32: 4, "uint32": 4, np.uint32: 4,
    }
    return int(mapping.get(dtype, 4))


def _theoretical_size_mb(n: int, d: int, dtype: Any = "float32") -> float:
    """Theoretical size in MB for an [n, d] dense matrix with given dtype (no container overhead)."""
    if n <= 0 or d <= 0:
        return float("nan")
    return (n * d * _bytes_per_elem(dtype)) / 1_000_000.0


def compute_per_run_sizes(
    n_corpus: int,
    dim_in: int,
    dim_out: int,
    *,
    orig_dtype: Any = "float32",
    comp_dtype: Any = "float32",
) -> Tuple[float, float, float]:
    """Return (emb_disk_mb_original, emb_disk_mb_compressed, storage_saving_pct) for THIS run.

    - Homogéneo: ambos tamaños se calculan con el MISMO N (n_corpus) y en el MISMO dtype supuesto.
    - Sin I/O: tamaño teórico; evita medir artefactos ajenos (caches, índices, checkpoints).
    """
    mb_orig = _theoretical_size_mb(n_corpus, dim_in, orig_dtype)
    mb_comp = _theoretical_size_mb(n_corpus, dim_out, comp_dtype)
    saving = None
    if not (np.isnan(mb_orig) or np.isnan(mb_comp)) and mb_orig > 0:
        saving = max(0.0, (1.0 - (mb_comp / mb_orig)) * 100.0)
    return mb_orig, mb_comp, saving


def persist_array_and_get_size_mb(arr: np.ndarray, path: Path, *, npz: bool = False) -> float:
    """(Opcional) Guarda el array y devuelve su tamaño real en MB. Útil si quieres 'prueba física'.

    Nota: si usas .npz (comprimido), ya NO comparas a igualdad de contenedor con .npy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if npz:
        np.savez_compressed(path, arr=arr)
    else:
        np.save(path, arr)
    return float(path.stat().st_size) / 1_000_000.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

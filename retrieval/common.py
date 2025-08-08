# retrieval/common.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F

try:
    import faiss 
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# ---------- Normalización / conversión --------------------------------------

def as_float32_cpu_np(t: torch.Tensor) -> np.ndarray:
    """Convierte tensor a numpy float32 en CPU y contiguo."""
    return np.ascontiguousarray(t.detach().cpu().numpy().astype("float32", copy=False))

def normalize_l2_np_inplace(x: np.ndarray) -> None:
    """Normaliza L2 in-place un array numpy de forma vectorizada."""
    if _HAS_FAISS:
        faiss.normalize_L2(x)
    else:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x /= n

def normalize_l2_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Devuelve una copia L2-normalizada (torch)."""
    return F.normalize(x, p=2, dim=dim)


# ---------- Métricas de rendimiento -----------------------------------------

@dataclass
class StatsTracker:
    build_time_s: float = 0.0
    search_time_s: float = 0.0
    search_calls: int = 0
    per_query_ms: List[float] = field(default_factory=list)

    def add_build_time(self, seconds: float) -> None:
        self.build_time_s += float(seconds)

    def add_search_batch(self, batch_size: int, seconds: float) -> None:
        self.search_time_s += float(seconds)
        self.search_calls += 1
        ms_per_query = (seconds / max(1, int(batch_size))) * 1000.0
        self.per_query_ms.append(ms_per_query)

    def get_stats(self, reset: bool = False) -> Dict[str, object]:
        out = {
            "build_time_s": float(self.build_time_s),
            "search_time_s": float(self.search_time_s),
            "search_calls": int(self.search_calls),
            "per_query_ms": list(self.per_query_ms),
        }
        if reset:
            self.build_time_s = 0.0
            self.search_time_s = 0.0
            self.search_calls = 0
            self.per_query_ms.clear()
        return out

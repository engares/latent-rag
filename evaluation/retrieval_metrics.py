# evaluation/retrieval_metrics.py

from __future__ import annotations

import numpy as np
from typing import List, Sequence, Dict, Tuple, Union
              
ID = Union[int, str]

###############################################################################
#  Métricas elementales (1 consulta)
###############################################################################

def recall_at_k(retrieved: Sequence[ID], relevant: Sequence[ID], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def mrr(retrieved: Sequence[ID], relevant: Sequence[ID]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: Sequence[ID], relevant: Sequence[ID], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2) if d in relevant else 0.0
        for i, d in enumerate(retrieved[:k])
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0

###############################################################################
#  Vectorización sobre lotes
###############################################################################

def _parse_metric(m: str) -> Tuple[str, int | None]:
    return (m.split("@")[0], int(m.split("@")[1])) if "@" in m else (m, None)

def _score_single(
    retrieved: Sequence[ID],
    relevant: Sequence[ID],
    name: str,
    k: int | None,
) -> float:
    name = name.lower()
    if name == "recall" and k is not None:
        return recall_at_k(retrieved, relevant, k)
    if name == "mrr":
        return mrr(retrieved[: (k or len(retrieved))], relevant)
    if name == "ndcg" and k is not None:
        return ndcg_at_k(retrieved, relevant, k)
    raise ValueError(f"Métrica '{name}' mal especificada.")

def evaluate_retrieval(retrieved_batch: List[Sequence[ID]] | Sequence[ID],
    relevant_batch: List[Sequence[ID]] | Sequence[ID],
    metrics: List[str] | None = None,
    *,
    return_per_query: bool = False,
) -> Dict[str, Dict[str, float]] | Tuple[Dict[str, Dict[str, float]],
                                         List[Dict[str, float]]]:
    
    # ── Normalizar entrada a lote ──────────────────────────────────────────
    single = isinstance(retrieved_batch[0], (str, int))  # type: ignore[index]
    if single:
        retrieved_batch = [retrieved_batch]              # type: ignore[assignment]
        relevant_batch  = [relevant_batch]               # type: ignore[assignment]

    assert len(retrieved_batch) == len(relevant_batch), \
        "retrieved_batch y relevant_batch deben tener la misma longitud."

    if not metrics:
        raise ValueError("No se especificaron métricas de evaluación.")

    Q = len(retrieved_batch)
    per_query: List[Dict[str, float]] = [{} for _ in range(Q)]
    summary: Dict[str, Dict[str, float]] = {}

    for m in metrics:
        name, k = _parse_metric(m)
        vals = [
            _score_single(r, rel, name, k)
            for r, rel in zip(retrieved_batch, relevant_batch)
        ]
        summary[m] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if Q > 1 else 0.0,
        }
        for d, v in zip(per_query, vals):
            d[m] = v

    if return_per_query:
        return summary, per_query
    if single:
        return {k: v["mean"] for k, v in summary.items()}      # compat.
    return summary

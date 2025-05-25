from __future__ import annotations

"""Métricas de generación con *bootstrap* y prueba de significancia emparejada.

Uso principal:
--------------
>>> mean_ci = evaluate_generation_bootstrap(refs, cands, metrics=["BLEU", "ROUGE-L"])
>>> pval = paired_bootstrap_test(refs, sys_a, sys_b, metric="BLEU")
"""

from collections.abc import Callable
from typing import List, Dict, Tuple
import numpy as np
from sacrebleu.metrics import BLEU as _BLEUMetric
from rouge_score import rouge_scorer
import random

###############################################################################
#  MÉTRICAS BÁSICAS                                                           #
###############################################################################

_bleu = _BLEUMetric()
_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_bleu(candidates: List[str], references: List[str]) -> float:
    """BLEU corpus‐level sacreBLEU (0‑100). Handles both flat and nested reference lists."""
    # Flatten references if it's a list of lists (shouldn't be for single-ref BLEU)
    if references and isinstance(references[0], list):
        # If already nested, flatten one level
        references = [ref for sublist in references for ref in (sublist if isinstance(sublist, list) else [sublist])]
    return _bleu.corpus_score(candidates, [references]).score


def compute_rouge_l(candidates: List[str], references: List[str]) -> float:
    """Promedio de ROUGE‑L (F1) ×100."""
    def to_str(x):
        if isinstance(x, list):
            return " ".join(map(str, x))
        return str(x)
    scores = [
        _scorer.score(to_str(ref), to_str(cand))["rougeL"].fmeasure * 100
        for ref, cand in zip(references, candidates)
    ]
    return float(np.mean(scores))


_metric_fn: Dict[str, Callable[[List[str], List[str]], float]] = {
    "BLEU": compute_bleu,
    "ROUGE-L": compute_rouge_l,
}

###############################################################################
#  BOOTSTRAP                                                                  #
###############################################################################

def _bootstrap_ci(
    func: Callable[[List[str], List[str]], float],
    refs: List[str],
    cands: List[str],
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """Devuelve media, límite inferior y superior del IC al (1‑alpha)."""
    if seed is not None:
        random.seed(seed)
    N = len(refs)
    stats = []
    for _ in range(n_samples):
        idx = [random.randint(0, N - 1) for _ in range(N)]
        stats.append(func([cands[i] for i in idx], [refs[i] for i in idx]))
    stats_np = np.array(stats)
    mean = stats_np.mean()
    lower = np.percentile(stats_np, 100 * alpha / 2)
    upper = np.percentile(stats_np, 100 * (1 - alpha / 2))
    return mean, lower, upper


def evaluate_generation_bootstrap(
    references: List[str],
    candidates: List[str],
    metrics: List[str] | None = None,
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Calcula métricas + IC al 95 % mediante bootstrap.

    Retorna: {metric: {"mean": m, "ci_lower": l, "ci_upper": u}}
    """
    if metrics is None:
        metrics = ["BLEU", "ROUGE-L"]

    assert len(references) == len(candidates) >= 30, ( 
        "Se requieren al menos 30 pares ref‑cand para un IC mínimo; se recomienda ≥1000."
    )

    results: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        if m not in _metric_fn:
            raise ValueError(f"Non soported metric '{m}'")
        mean, lo, hi = _bootstrap_ci(_metric_fn[m], references, candidates, n_samples, alpha, seed)
        results[m] = {"mean": mean, "ci_lower": lo, "ci_upper": hi}
    return results

###############################################################################
#  PAIRED BOOTSTRAP SIGNIFICANCE TEST                                         #
###############################################################################

def paired_bootstrap_test(
    references: List[str],
    sys_a: List[str],
    sys_b: List[str],
    metric: str = "BLEU",
    n_samples: int = 10000,
    seed: int | None = None,
) -> Dict[str, float]:
    """Prueba de significancia emparejada (bootstrap) entre dos sistemas.

    Devuelve un dict: {"diff_mean": d, "ci_lower": lo, "ci_upper": hi, "p_value": p}
    p‑value ≈ proporción de muestras con diferencia ≤0 (o ≥0, según el signo).
    """
    assert len(references) == len(sys_a) == len(sys_b)
    if seed is not None:
        random.seed(seed)

    if metric not in _metric_fn:
        raise ValueError(f"Métrica '{metric}' no soportada.")
    fn = _metric_fn[metric]

    N = len(references)
    diffs = []
    for _ in range(n_samples):
        idx = [random.randint(0, N - 1) for _ in range(N)]
        a_score = fn([sys_a[i] for i in idx], [references[i] for i in idx])
        b_score = fn([sys_b[i] for i in idx], [references[i] for i in idx])
        diffs.append(a_score - b_score)

    diffs_np = np.array(diffs)
    diff_mean = diffs_np.mean()
    ci_lower = np.percentile(diffs_np, 2.5)
    ci_upper = np.percentile(diffs_np, 97.5)
    # p‑value: H0: diff <= 0  (si diff_mean>0) ó diff >=0 (si diff_mean<0)
    if diff_mean >= 0:
        p_val = (diffs_np <= 0).mean()
    else:
        p_val = (diffs_np >= 0).mean()

    return {
        "diff_mean": diff_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }

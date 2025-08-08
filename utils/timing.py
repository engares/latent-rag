# utils/timing.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Iterable, List

@contextmanager
def stopwatch(accumulator: Dict[str, float], key: str):
    """Accumulates elapsed seconds under 'key'."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        accumulator[key] = accumulator.get(key, 0.0) + (time.perf_counter() - t0)

def percentiles(values: Iterable[float], ps: Iterable[float]) -> List[float]:
    """Return empirical percentiles in the range [0, 100]."""
    arr = sorted(values)
    if not arr:
        return [float("nan") for _ in ps]
    n = len(arr)
    out: List[float] = []
    for p in ps:
        i = min(max(int(round(p / 100.0 * (n - 1))), 0), n - 1)
        out.append(arr[i])
    return out

# evaluation/benchmark.py
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from evaluation.retrieval_metrics import evaluate_retrieval
from retrieval.bm25 import BM25Retriever
from retrieval.dpr import DPRRetriever
from utils.load_config import load_config


def _as_str_matrix(rows: Sequence[Sequence[int | str]]) -> List[List[str]]:
    """Ensure retrieved/relevant IDs are strings, as expected by metrics."""
    return [[str(x) for x in row] for row in rows]


def run_benchmark(
    queries: Sequence[str],
    corpus: Sequence[str],
    relevant: Sequence[Sequence[str]],
    cfg_path: str = "./config/config.yaml",
    retrievers: Iterable[str] = ("bm25", "dpr"),
) -> Dict[str, dict]:
    """Run BM25/DPR over the corpus and return retrieval metrics per method.

    Args:
        queries: Input questions.
        corpus:  Documents (no chunking).
        relevant: For each query, list of gold documents (as full texts).
        cfg_path: Path to YAML configuration.
        retrievers: Which baselines to evaluate.

    Returns:
        Mapping {retriever_name: {"retrieval_metrics": {...}, "retriever_stats": {...}}}
        ready to be appended into the benchmark CSV by main.py.
    """
    cfg = load_config(cfg_path)
    metrics = cfg.get("evaluation", {}).get(
        "retrieval_metrics", ["Recall@10", "MRR@10", "nDCG@10"]
    )
    top_k = int(cfg.get("retrieval", {}).get("top_k", 10))

    # Map gold texts -> doc IDs to align with retriever outputs
    text2docid = {t: i for i, t in enumerate(corpus)}
    relevant_ids: List[List[int]] = []
    missing = 0
    for rel_list in relevant:
        ids = []
        for ctx in rel_list:
            did = text2docid.get(ctx)
            if did is not None:
                ids.append(did)
            else:
                missing += 1
        relevant_ids.append(ids)
    if missing:
        # Silent handling; main.py does logging. Keep function pure.
        pass

    results: Dict[str, dict] = {}

    for name in retrievers:
        if name == "bm25":
            retr = BM25Retriever()
        elif name == "dpr":
            retr = DPRRetriever()
        else:
            raise ValueError(f"Unsupported retriever: {name}")

        retr.build_index(corpus)
        retrieved_ids: List[List[int]] = [
            retr.retrieve(q, k=top_k) for q in queries
        ]

        # Metrics need string IDs
        retrieved_as_str = _as_str_matrix(retrieved_ids)
        relevant_as_str = _as_str_matrix(relevant_ids)
        summary = evaluate_retrieval(retrieved_as_str, relevant_as_str, metrics)

        results[name] = {
            "retrieval_metrics": summary,  # contains mean/std per metric
            "retriever_stats": getattr(retr, "get_stats", lambda: {})(),
        }

    return results

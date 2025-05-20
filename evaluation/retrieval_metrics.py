import numpy as np
from typing import List, Union, Dict
import yaml
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

def load_config(path: str = DEFAULT_CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

config = load_config()

def recall_at_k(retrieved: List[Union[str, int]], relevant: List[Union[str, int]], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = len(set(retrieved_k) & set(relevant))
    return hits / len(relevant)

def mrr(retrieved: List[Union[str, int]], relevant: List[Union[str, int]]) -> float:
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0

def ndcg_at_k(retrieved: List[Union[str, int]], relevant: List[Union[str, int]], k: int) -> float:
    retrieved_k = retrieved[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) if doc in relevant else 0.0 
        for i, doc in enumerate(retrieved_k)
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate_retrieval(
    retrieved: List[Union[str, int]], 
    relevant: List[Union[str, int]], 
    metrics: List[str] = None
) -> Dict[str, float]:

    if metrics is None:
        metrics = config.get("evaluation", {}).get("retrieval_metrics", [])

    results = {}
    for metric in metrics:
        if "@" in metric:
            name, k_str = metric.split("@")
            k = int(k_str)
        else:
            name = metric
            k = None

        name_lower = name.lower()
        if name_lower == "recall" and k is not None:
            results[f"Recall@{k}"] = recall_at_k(retrieved, relevant, k)
        elif name_lower == "mrr":
            cutoff = k if k else len(retrieved)
            results[f"MRR@{cutoff}"] = mrr(retrieved[:cutoff], relevant)
        elif name_lower == "ndcg" and k is not None:
            results[f"nDCG@{k}"] = ndcg_at_k(retrieved, relevant, k)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results

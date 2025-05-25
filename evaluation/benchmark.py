from typing import Dict, Sequence
from retrieval.bm25 import BM25Retriever
from retrieval.dpr  import DPRRetriever
from retrieval.sbert import SBERTRetriever
from utils.load_config import load_config
from evaluation.retrieval_metrics import evaluate_retrieval, paired_bootstrap_test

Retrievers = {
    "bm25":  BM25Retriever(),
    "dpr":   DPRRetriever(),
    "sbert": SBERTRetriever(),          # sin AE
    "vae":   SBERTRetriever(ae_type="vae"),
    "dae":   SBERTRetriever(ae_type="dae"),
    "cae":   SBERTRetriever(ae_type="contrastive"),
}

def run_benchmark(queries: Sequence[str],
                  corpus:  Sequence[str],
                  gold:    Sequence[Sequence[str]],
                  cfg_path="./config/config.yaml") -> None:
    cfg = load_config(cfg_path)
    metrics = cfg["evaluation"]["retrieval_metrics"]
    results: Dict[str, Dict[str, float]] = {}

    for name, retr in Retrievers.items():
        retr.build_index(corpus)
        retrieved = [retr.retrieve(q, k=cfg["retrieval"]["top_k"]) for q in queries]
        summary   = evaluate_retrieval(retrieved, gold, metrics)   # media + sd
        results[name] = {m: v["mean"] for m, v in summary.items()}
        print(f"{name.upper():<6}", summary)

    # Significancia (ejemplos)
    _print_sig(results, queries, gold, metrics[0])

def _print_sig(res, queries, gold, metric):
    from itertools import combinations
    for a, b in combinations(res.keys(), 2):
        pa, pb = res[a][metric], res[b][metric]
        # bootstrap paired significance (aproveche evaluate_generation_bootstrap si lo desea)
        print(f"{a} vs {b}: Δ={pa-pb:+.4f}")

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from dotenv import load_dotenv
from utils.data_utils import load_evaluation_data

# Third‑party
from rich import print as rprint
from sentence_transformers import SentenceTransformer  # lazy‑loaded by embedder

# First‑party (repository) -----------------------------------------------------
from utils.load_config import init_logger, load_config
from utils.training_utils import resolve_device, set_seed
from retrieval.embedder import EmbeddingCompressor
from retrieval.retriever import retrieve_top_k
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import (
    evaluate_generation_bootstrap as eval_generation,
)
from generation.generator import RAGGenerator
from models.variational_autoencoder import VariationalAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
from models.contrastive_autoencoder import ContrastiveAutoencoder

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _load_autoencoder(
    cfg_models: Dict[str, Dict[str, Any]],
    ae_type: str,
    device: str,
) -> Optional[torch.nn.Module]:
    """Instantiate and load the requested auto‑encoder.

    Args:
        cfg_models: Dict extracted from YAML under `models`.
        ae_type:    "vae", "dae", "contrastive" or "none".
        device:     "cpu" | "cuda".

    Returns:
        A `torch.nn.Module` in eval mode, or *None* if `ae_type == "none"`.
    """

    if ae_type == "none":
        return None

    if ae_type not in cfg_models:
        raise ValueError(
            f"[CONFIG] Auto‑encoder '{ae_type}' not found under 'models' in config."
        )

    mcfg = cfg_models[ae_type]
    input_dim = mcfg.get("input_dim", 384)
    latent_dim = mcfg.get("latent_dim", 64)
    hidden_dim = mcfg.get("hidden_dim", 512)
    checkpoint = mcfg.get("checkpoint")

    if ae_type == "vae":
        model: torch.nn.Module = VariationalAutoencoder(  # type: ignore[assignment]
            input_dim, latent_dim, hidden_dim
        )
    elif ae_type == "dae":
        model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim)
    elif ae_type == "contrastive":
        model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim)
    else:
        raise RuntimeError("Unreachable branch – ae_type already validated.")

    if checkpoint and Path(checkpoint).exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        raise FileNotFoundError(f"Checkpoint for '{ae_type}' not found: {checkpoint}")

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _encode_corpus(
    compressor: EmbeddingCompressor,
    texts: Sequence[str],
    compress: bool = True,
) -> torch.Tensor:
    """Return document embeddings as `[N × D]` float32 CPU tensor."""

    return compressor.encode_text(list(texts), compress=compress)


def _retrieve_documents(
    query_emb: torch.Tensor,
    doc_emb: torch.Tensor,
    corpus: Sequence[str],
    retr_cfg: Dict[str, Any],
) -> Tuple[List[str], List[float]]:
    """Retrieve top‑k docs and similarity scores for a *single* query."""

    top_k = retr_cfg.get("top_k", 10)
    metric = retr_cfg.get("similarity_metric", "cosine")
    results = retrieve_top_k(query_emb, doc_emb, list(corpus), k=top_k, metric=metric)
    docs, scores = zip(*results)
    return list(docs), list(scores)


def _evaluate_retrieval(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Sequence[str]] | Sequence[str],
    metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """Wrapper around `evaluate_retrieval` with sensible defaults."""

    return evaluate_retrieval(retrieved, relevant, metrics=metrics)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

class PipelineRunner:  # noqa: D101 – simple orchestrator
    def __init__(self, cfg: Dict[str, Any], ae_type: str, logger):
        self.cfg = cfg
        self.ae_type = ae_type
        self.logger = logger

        device = resolve_device(cfg.get("training", {}).get("device"))
        self.device = device
        self.logger.main.info("Device resolved → %s", device)

        # Auto‑encoder & compressor ------------------------------------------------
        ae_model = _load_autoencoder(cfg["models"], ae_type, device)
        self.compressor = EmbeddingCompressor(
            base_model_name=cfg["embedding_model"]["name"],
            autoencoder=ae_model,
            device=device,
        )
        self.logger.main.info("Compressor ready (AE = %s)", ae_type)

        # Retriever ---------------------------------------------------------------
        self.retr_cfg = cfg.get("retrieval", {})

        # Generator ---------------------------------------------------------------
        self.generator = RAGGenerator(cfg)

    # ---------------------------------------------------------------------
    def process(
        self,
        queries: Sequence[str],
        corpus: Sequence[str],
        relevant_docs: Optional[Sequence[str]] = None,
        generate: bool = False,
    ) -> None:
        """Run encode → retrieve → generate → evaluate for *all* queries."""

        self.logger.main.info("Running pipeline: |queries|=%d |docs|=%d", len(queries), len(corpus))
        doc_embeddings = _encode_corpus(self.compressor, corpus, compress=True)
        query_embeddings = _encode_corpus(self.compressor, queries, compress=True)

        all_retrieved: List[Sequence[str]] = []
        answers: List[str] = []

        for idx, (q, q_emb) in enumerate(zip(queries, query_embeddings)):
            docs_k, _ = _retrieve_documents(q_emb, doc_embeddings, corpus, self.retr_cfg)
            all_retrieved.append(docs_k)
            if generate:
                ans = self.generator.generate(q, docs_k)
                answers.append(ans)
                self.logger.main.debug("[%d] Q: %s | A: %s", idx, q, ans[:60] + "…")

        # ----------------------------------------------------------------- EVAL
        eval_cfg = self.cfg.get("evaluation", {})
        if relevant_docs:
            ret_metrics = _evaluate_retrieval(
                all_retrieved,
                relevant_docs,
                metrics=eval_cfg.get("retrieval_metrics", ["Recall@5"]),
            )
            rprint("[bold magenta]\n[Retrieval evaluation]\n[/]")
            for k, v in ret_metrics.items():
                rprint(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

        if generate and relevant_docs and len(queries) >= 30: 
            # If the user provided refs with ≥30 samples we can bootstrap.
            gen_metrics = eval_generation(
                references=list(relevant_docs),
                candidates=answers,
                metrics=eval_cfg.get("generation_metrics", ["ROUGE-L"]),
            )
            rprint("[bold magenta]\n[Generation evaluation]\n[/]")
            for m, d in gen_metrics.items():
                rprint(f"{m}: {d['mean']:.2f} (CI 95%: {d['ci_lower']:.2f}–{d['ci_upper']:.2f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:  # noqa: D401
    """Return command‑line arguments."""

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="./config/config.yaml")
    known, _ = pre_parser.parse_known_args(sys.argv[1:])

    cfg = load_config(known.config)
    valid_ae = list(cfg.get("models", {}).keys()) + ["none", "all"]

    parser = argparse.ArgumentParser(description="Run RAG‑AE experimental pipeline")
    parser.add_argument("--config", default="./config/config.yaml", help="Path to YAML config")
    parser.add_argument("--ae_type", default="vae", choices=valid_ae, help="Select auto‑encoder variant")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--dataset", choices=["squad", "uda"], default="squad",
                    help="Dataset for evaluation (SQuAD or UDA)")
    parser.add_argument("--max_samples", type=int, default=200,
                        help="Maximum number of queries to use")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare against BM25, DPR, SBERT, AE...")
    parser.add_argument("--generate", action="store_true", help="Run generation step (RAG)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – standard script
    args = _parse_args()

    cfg = load_config(args.config)
    log = init_logger(cfg.get("logging", {}))
    set_seed(args.seed, cfg.get("training", {}).get("deterministic", False), logger=log.train)
    load_dotenv()

    ae_variants = (
        [args.ae_type]
        if args.ae_type != "all"
        else [k for k in cfg.get("models", {}).keys() if k in {"vae", "dae", "contrastive"}]
    )

    # --------------------------------------------------------------------- Toy corpus (replace with real dataset) --
    queries, corpus, relevant = load_evaluation_data(args.dataset, max_samples=args.max_samples)

    # --------------------------------------------------------------------- Run each variant
    for ae in ae_variants:
        rprint(f"\n[bold cyan]==== PIPELINE ({ae.upper()}) ====\n[/]")
        runner = PipelineRunner(cfg, ae, log)
        runner.process(queries, corpus, relevant_docs=relevant, generate=args.generate)


if __name__ == "__main__":
    main()

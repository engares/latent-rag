"""Main pipeline for RAG-AE experiments.

This script orchestrates the retrieval-augmented generation (RAG) pipeline, including encoding, retrieval, optional generation, and evaluation.
"""
from __future__ import annotations

import argparse
from html import parser
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import runner

import torch
from dotenv import load_dotenv
from utils.data_utils import load_evaluation_data

# Third‑party
from rich import print as rprint
from utils.benchmark_utils import build_metrics_row, _append_csv_row

# First‑party (repository) -----------------------------------------------------
from utils.load_config import init_logger, load_config
from utils.training_utils import resolve_device, set_seed
from retrieval.embedder import EmbeddingCompressor
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import (
    evaluate_generation_bootstrap as eval_generation,
)
from generation.generator import RAGGenerator
from models.variational_autoencoder import VariationalAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
from models.contrastive_autoencoder import ContrastiveAutoencoder

from retrieval.retriever import build_retriever          
from retrieval.FAISSEmbeddingRetriever import FAISSEmbeddingRetriever
import time
from typing import Any

import csv
import json
from datetime import datetime

from retrieval.common import StatsTracker


def _print_run_card(cfg: Dict[str, Any], ae_type: str, *, generate: bool) -> None:
    """Print a summary of the experiment configuration.

    Args:
        cfg: Configuration dictionary.
        ae_type: Type of autoencoder used.
        generate: Whether the generation step is enabled.
    """
    retr = cfg.get("retrieval", {})
    ch = cfg.get("chunking", {})
    data = cfg.get("data", {})
    embm = cfg.get("embedding_model", {})
    gen = cfg.get("generation", {})

    use_chunking = bool(ch.get("enabled"))
    top_k = int(retr.get("top_k", 10))
    cand_k = int(retr.get("candidate_k", top_k * 3 if use_chunking else top_k))

    lines = [
        "Experiment Configuration",
        f"  Dataset: {data.get('dataset', '?')} / split=validation / max_samples={data.get('max_samples')}",
        f"  Embedding: {embm.get('name', '?')} (max_length={embm.get('max_length', '?')})",
        f"  Autoencoder: {ae_type}",
        f"  Retrieval: backend={retr.get('backend', 'faiss')} index_type={retr.get('index_type', 'hnsw')} "
        f"use_gpu={bool(retr.get('use_gpu'))} top_k={top_k} candidate_k={cand_k} max_chunks_per_doc={retr.get('max_chunks_per_doc', 2)}",
        (
            f"  Chunking: enabled={use_chunking} mode={ch.get('mode', 'sliding')} "
            f"max_tokens={ch.get('max_tokens', 128)} stride={ch.get('stride', 64)} "
            + (f"min_tokens={ch.get('min_tokens', 48)} " if ch.get('mode', 'sliding') != 'sliding' else "")
            + f"tokenizer={ch.get('tokenizer_name', embm.get('name', '?'))}"
        ) if use_chunking else "  Chunking: disabled",
        f"  Evaluation: {', '.join(cfg.get('evaluation', {}).get('retrieval_metrics', ['Recall@10', 'MRR@10', 'nDCG@10']))}",
        (
            f"  Generation: provider={gen.get('provider')} model={gen.get('model')} "
            f"temperature={gen.get('temperature', 0.3)} max_tokens={gen.get('max_tokens', 256)}"
        ) if generate else "  Generation: disabled",
    ]
    from rich import print as rprint
    rprint("\n" + "\n".join(lines) + "\n")



# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _resolve_ckpt_path(checkpoint: str | None, cfg_paths: Dict[str, Any]) -> Path:
    """Returns the absolute path of the checkpoint.
    If relative, it is concatenated with paths.checkpoints_dir.
    """
    if not checkpoint:
        return Path()  # invalid
    p = Path(checkpoint)
    if p.is_absolute():
        return p
    base = Path(cfg_paths.get("checkpoints_dir", "./models/checkpoints"))
    return (base / p).resolve()

def _load_autoencoder(
    cfg_models: Dict[str, Dict[str, Any]],
    ae_type: str,
    device: str,
    cfg_paths: Dict[str, Any] | None = None,
) -> Optional[torch.nn.Module]:
    """Instantiate and load the requested autoencoder."""
    if ae_type == "none":
        return None

    if ae_type not in cfg_models:
        raise ValueError(f"[CONFIG] Auto‑encoder '{ae_type}' not found in 'models'.")

    mcfg = cfg_models[ae_type]
    input_dim  = mcfg.get("input_dim", 384)
    latent_dim = mcfg.get("latent_dim", 64)
    hidden_dim = mcfg.get("hidden_dim", 512)

    # --- Factory by ae_type (without requiring 'class' in YAML)
    if ae_type == "vae":
        model: torch.nn.Module = VariationalAutoencoder(input_dim, latent_dim, hidden_dim)
    elif ae_type == "dae":
        model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim)
    elif ae_type == "cae":
        model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim)
    else:
        raise RuntimeError("Unrecognized AE type.")

    # --- Resolve checkpoint relative to paths.checkpoints_dir if needed
    ckpt = _resolve_ckpt_path(mcfg.get("checkpoint"), cfg_paths or {})
    if ckpt and ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
    else:
        raise FileNotFoundError(
            f"Checkpoint for '{ae_type}' not found: {ckpt} "
            f"(check 'paths.checkpoints_dir' and 'models.{ae_type}.checkpoint')"
        )

    return model.to(device).eval()



# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _safe_dim_from_tensor(x: torch.Tensor) -> int:
    """Devuelve la segunda dimensión de un tensor [N, D]; si no cumple, intenta inferir."""
    if isinstance(x, torch.Tensor) and x.ndim == 2:
        return x.size(1)
    raise ValueError("Expected a 2D tensor [N, D] to infer embedding dimension.")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Orchestrates the RAG pipeline: encode → retrieve → (optional) generate → evaluate."""

    def __init__(self, cfg: Dict[str, Any], ae_type: str, logger):
        """Initialize the pipeline runner.

        Args:
            cfg: Configuration dictionary.
            ae_type: Type of autoencoder used.
            logger: Logger instance.
        """
        self.cfg = cfg
        self.ae_type = ae_type
        self.logger = logger

        self.device = resolve_device(cfg.get("training", {}).get("device"))
        self.logger.main.info("Device resolved → %s", self.device)

        # Compressor (SBERT ± AE)
        ae_model = _load_autoencoder(cfg["models"], ae_type, self.device, cfg.get("paths", {}))
        self.compressor = EmbeddingCompressor(
            base_model_name=cfg["embedding_model"]["name"],
            autoencoder=ae_model,
            device=self.device,
        )
        self.logger.main.info("Compressor ready (AE = %s)", ae_type)

        # Retrieval configuration
        self.retr_cfg = cfg.get("retrieval", {})
        self.retriever = None  # Set in _build_retriever

        # Generator
        self.generator = RAGGenerator(cfg)

    # ------------------------------------------------------------------ #
    def process(
        self,
        queries: Sequence[str],
        corpus: Sequence[str],
        relevant_docs: Optional[Sequence[Sequence[str]]] = None,
        generate: bool = False,
    ) -> None:
        """Run the pipeline: encode → retrieve → (optional) generate → evaluate.

        Args:
            queries: List of query strings.
            corpus: List of document strings.
            relevant_docs: Ground-truth relevant documents for evaluation.
            generate: Whether to run the generation step.
        """
        self.logger.main.info(
            "Running pipeline: |queries|=%d |corpus|=%d", len(queries), len(corpus)
        )

        # Immutable copy for doc-level evaluation
        orig_docs = list(corpus)
        context2docid: Dict[str, int] = {t: i for i, t in enumerate(orig_docs)}

        # Optional chunking for inference
        ch_cfg = self.cfg.get("chunking", {})
        use_chunking = bool(ch_cfg.get("enabled", False))
        if use_chunking:
            from utils.data_utils import prepare_inference_chunks
            chunks, chunk_index = prepare_inference_chunks(
                orig_docs,
                mode=ch_cfg.get("mode", "sliding"),
                max_tokens=ch_cfg.get("max_tokens", 128),
                stride=ch_cfg.get("stride", 64),
                min_tokens=ch_cfg.get("min_tokens", 48),
                tokenizer_name=ch_cfg.get("tokenizer_name", self.cfg["embedding_model"]["name"]),
                index_out=ch_cfg.get("index_out"),
                store_chunk_text=ch_cfg.get("store_chunk_text", True),
            )
            corpus = chunks
            corpus_doc_ids: List[int] = chunk_index["doc_id"].astype(int).tolist()  # chunk → doc
            self.logger.main.info("Chunking enabled: |docs|=%d → |chunks|=%d", len(orig_docs), len(corpus))
            self.logger.main.debug("Chunking configuration: %s", ch_cfg)
        else:
            corpus_doc_ids = list(range(len(corpus)))

        # Encode corpus once
        doc_embeddings = self.compressor.encode_text(list(corpus), compress=True)

        # Build or load retrieval index
        t0 = time.perf_counter()
        self.retriever = build_retriever(
            embeddings=doc_embeddings,
            texts=corpus,
            doc_ids=corpus_doc_ids,
            cfg=self.retr_cfg,
        )
        self.logger.main.info(
            "Retriever backend '%s' initialised in %.2f s",
            self.retr_cfg.get("backend", "faiss"),
            time.perf_counter() - t0,
        )

        # Encode queries
        query_embeddings = self.compressor.encode_text(list(queries), compress=True)

        # Retrieve with doc-level MaxSim aggregation
        top_k = int(self.retr_cfg.get("top_k", 10))
        cand_k = int(self.retr_cfg.get("candidate_k", top_k * 3 if use_chunking else top_k))

        all_retrieved_docids: List[List[int]] = []
        answers: List[str] = []

        for idx, (q, q_emb) in enumerate(zip(queries, query_embeddings)):
            texts_k, scores_k, docids_k = self.retriever.retrieve(q_emb, top_k=cand_k)

            # Aggregation: doc-level MaxSim
            agg: Dict[int, float] = {}
            for did, sc in zip(docids_k, scores_k):
                prev = agg.get(did)
                if (prev is None) or (sc > prev):
                    agg[did] = sc

            # Re-rank docs by max score (desc) and truncate to top_k unique
            ranked_docids = sorted(agg, key=agg.get, reverse=True)[:top_k]
            all_retrieved_docids.append(ranked_docids)

            # Optional: context for LLM prioritizing chunks of top docs
            if generate:
                per_doc_cap = int(self.retr_cfg.get("max_chunks_per_doc", 2))
                used: Dict[int, int] = {d: 0 for d in ranked_docids}
                selected_chunks: List[str] = []
                for t, d in zip(texts_k, docids_k):
                    if d in used and used[d] < per_doc_cap:
                        selected_chunks.append(t)
                        used[d] += 1
                    if len(selected_chunks) >= max(1, per_doc_cap * len(ranked_docids)):
                        break
                ctx_for_llm = selected_chunks if selected_chunks else texts_k[:top_k]
                ans = self.generator.generate(q, ctx_for_llm)
                answers.append(ans)
                self.logger.main.debug("[%d] Q: %s | A: %s", idx, q, (ans[:60] + "…") if ans else "")

        # Evaluation (doc-level, using doc_ids)
        ret_metrics = {}
        if relevant_docs:
            relevant_doc_ids: List[List[int]] = []
            missing = 0
            for rel_list in relevant_docs:
                ids = []
                for ctx in rel_list:
                    did = context2docid.get(ctx)
                    if did is not None:
                        ids.append(did)
                    else:
                        missing += 1
                relevant_doc_ids.append(ids)
            if missing:
                self.logger.main.warning("Relevant items missing from mapping: %d", missing)

            retrieved_as_str = [[str(did) for did in row] for row in all_retrieved_docids]
            relevant_as_str = [[str(did) for did in row] for row in relevant_doc_ids]

            eval_cfg = self.cfg.get("evaluation", {})
            ret_metrics = evaluate_retrieval(
                retrieved_as_str,
                relevant_as_str,
                metrics=eval_cfg.get("retrieval_metrics", ["Recall@5"]),
            )
            rprint("\n[Retrieval evaluation]\n")
            for k, v in ret_metrics.items():
                rprint(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")

        # Optional generation metrics (requires many samples)
        if generate and relevant_docs and len(queries) >= 100:
            gen_metrics = eval_generation(
                references=[r[0] for r in relevant_docs],
                candidates=answers,
                metrics=self.cfg.get("evaluation", {}).get("generation_metrics", ["ROUGE-L"]),
            )
            rprint("\n[Generation evaluation]\n")
            for m, d in gen_metrics.items():
                rprint(f"{m}: {d['mean']:.2f} (CI 95%: {d['ci_lower']:.2f}–{d['ci_upper']:.2f})")

        # --- NEW: devolver paquete de resultados + estadísticas del retriever
        retr_stats = {}
        if hasattr(self.retriever, "get_stats"):
            retr_stats = self.retriever.get_stats(reset=False)

        # dim_out lo inferimos del embedding del corpus; dim_in del config/model
        if not (isinstance(doc_embeddings, torch.Tensor) and doc_embeddings.ndim == 2):
            raise ValueError("Expected a 2D tensor [N, D] to infer embedding dimension.")
        dim_out = int(doc_embeddings.size(1))
        dim_in  = int(self.compressor.input_dim if hasattr(self.compressor, "input_dim")
                      else self.cfg.get("embedding_model", {}).get("dim", dim_out))
        n_corpus = int(len(corpus))

        return {
            "retrieval_metrics": ret_metrics,   # dict[str] -> {"mean": x, "std": y}
            "retriever_stats":  retr_stats,     # build_time_s, per_query_ms, etc.
            "dim_in": dim_in,
            "dim_out": dim_out,
            "n_corpus": n_corpus,
            "ae_type": self.ae_type,
        }


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
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Maximum number of queries to use")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare against BM25, DPR, SBERT, AE...")
    parser.add_argument("--generate", action="store_true", help="Run generation step (RAG)")

    parser.add_argument("--metrics_csv", default="logs/benchmarks/experiments.csv",
                    help="Ruta del CSV donde añadir una fila por run")
    parser.add_argument("--benchmark_tag", default="",
                    help="Etiqueta libre para identificar el experimento (columna 'tag')")


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
        _print_run_card(cfg, ae, generate=args.generate)
        runner = PipelineRunner(cfg, ae, log)
        result = runner.process(queries, corpus, relevant_docs=relevant, generate=args.generate)
        row = build_metrics_row(cfg, args, ae, result)
        _append_csv_row(args.metrics_csv, row)

                


if __name__ == "__main__":
    main()

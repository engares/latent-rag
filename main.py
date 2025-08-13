"""Main pipeline for RAG-AE experiments.

This script orchestrates the retrieval-augmented generation (RAG) pipeline, including encoding, retrieval, optional generation, and evaluation.
"""
from __future__ import annotations

import argparse
import sys  # needed for parse_known_args
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import time

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
import time
from typing import Any


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
        f"  Dataset: {data.get('dataset', '?')} / split=validation / max_samples={data.get('max_samples')} We are using validation as the test set",
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
    """Orchestrates the RAG pipeline: encode → retrieve → (optional) generate → evaluate.

    Supports reuse of precomputed (chunked) corpus and base SBERT embeddings for multiple AE variants.
    """

    def __init__(self, cfg: Dict[str, Any], ae_type: str, logger,
                 *,
                 pre_corpus_texts: Optional[Sequence[str]] = None,
                 pre_corpus_doc_ids: Optional[Sequence[int]] = None,
                 base_corpus_embeddings: Optional[torch.Tensor] = None,
                 base_query_embeddings: Optional[torch.Tensor] = None,
                 ):  # noqa: D401
        """Initialize the pipeline runner.

        Args:
            cfg: Configuration dictionary.
            ae_type: Type of autoencoder used.
            logger: Logger instance.
            pre_corpus_texts: Optional precomputed chunked corpus texts.
            pre_corpus_doc_ids: Optional precomputed document IDs for the corpus.
            base_corpus_embeddings: Optional precomputed base embeddings for the corpus.
            base_query_embeddings: Optional precomputed base embeddings for the queries.
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
        # Store precomputed reuse artifacts
        self.pre_corpus_texts = list(pre_corpus_texts) if pre_corpus_texts is not None else None
        self.pre_corpus_doc_ids = list(pre_corpus_doc_ids) if pre_corpus_doc_ids is not None else None
        self.base_corpus_embeddings = base_corpus_embeddings
        self.base_query_embeddings = base_query_embeddings

                # --- NEW: explicit dimension banner + invariant
        self.logger.main.info(
            "[Compressor] SBERT_dim=%d | latent_dim=%d | compressed=%s",
            int(self.compressor.input_dim),
            int(self.compressor.latent_dim),
            str(self.compressor.latent_dim < self.compressor.input_dim),
        )
        if self.compressor.latent_dim > self.compressor.input_dim:
            raise ValueError(
                f"Invalid dims: latent_dim={self.compressor.latent_dim} "
                f"> input_dim={self.compressor.input_dim}"
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
    ) -> Dict[str, Any]:
        """Run the pipeline end-to-end and return metrics and footprint."""
        self.logger.main.info(
            "Running pipeline: |queries|=%d |corpus|=%d", len(queries), len(corpus)
        )

        # Reuse pre-chunked corpus if provided
        if self.pre_corpus_texts is not None:
            corpus = self.pre_corpus_texts
            corpus_doc_ids = self.pre_corpus_doc_ids or list(range(len(corpus)))
            use_chunking = bool(self.cfg.get("chunking", {}).get("enabled", False))
            # Mapping for evaluation (chunk texts treated as docs)
            orig_docs = list(corpus)
            context2docid: Dict[str, int] = {t: i for i, t in enumerate(orig_docs)}
        else:
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
                corpus_doc_ids = chunk_index["doc_id"].astype(int).tolist()
                self.logger.main.info("Chunking enabled: |docs|=%d → |chunks|=%d", len(orig_docs), len(corpus))
                self.logger.main.debug("Chunking configuration: %s", ch_cfg)
            else:
                corpus_doc_ids = list(range(len(corpus)))
        # ------------------------- Encode corpus (COMPRESSED) -------------------------
        if self.base_corpus_embeddings is not None:
            # Reuse base SBERT embeddings → apply AE encode if needed
            with torch.inference_mode():
                if self.compressor.autoencoder:
                    x = self.base_corpus_embeddings.to(self.device)
                    enc = self.compressor.autoencoder.encode(x)
                    if isinstance(enc, tuple):
                        enc = enc[0]
                    doc_embeddings = enc.detach().cpu().contiguous()
                else:
                    doc_embeddings = self.base_corpus_embeddings.detach().cpu().contiguous()
        else:
            with torch.inference_mode():
                doc_embeddings = self.compressor.encode_text(list(corpus), compress=True)
        if not (isinstance(doc_embeddings, torch.Tensor) and doc_embeddings.ndim == 2):
            raise ValueError("Corpus embeddings must be a 2D tensor [N, D].")
        d_corpus = int(doc_embeddings.size(1))
        # --- NEW: assert compression path is really used
        if d_corpus != int(self.compressor.latent_dim):
            raise RuntimeError(
                f"Unexpected corpus dim: got {d_corpus}, expected latent_dim={self.compressor.latent_dim}. "
                "Are you sure compress=True and the AE checkpoint matches the configured latent_dim?"
            )
        cr = float(self.compressor.input_dim) / float(d_corpus)
        self.logger.main.info("[Embeddings] corpus_dim=%d (CR=%.2f× from %d)",
                              d_corpus, cr, int(self.compressor.input_dim))

        # ------------------------- Build retriever -------------------------
        t0 = time.perf_counter()
        self.retriever = build_retriever(
            embeddings=doc_embeddings,   # D must be == latent_dim
            texts=corpus,
            doc_ids=corpus_doc_ids,
            cfg=self.retr_cfg,
        )
        init_secs = time.perf_counter() - t0
        self.logger.main.info(
            "Retriever backend '%s' initialised in %.2f s",
            self.retr_cfg.get("backend", "faiss"), init_secs
        )
        # --- NEW: sanity log if FAISS exposes 'd' and 'ntotal'
        try:
            faiss_d = int(getattr(getattr(self.retriever, "index", None), "d", d_corpus))
            faiss_n = int(getattr(getattr(self.retriever, "index", None), "ntotal", len(corpus)))
            self.logger.main.info("[FAISS] d=%d ntotal=%d (expect d=%d, ntotal=%d)",
                                  faiss_d, faiss_n, d_corpus, len(corpus))
            if faiss_d != d_corpus:
                raise RuntimeError(
                    f"FAISS dimension mismatch: index.d={faiss_d} vs embeddings D={d_corpus}"
                )
            if faiss_n != len(corpus):
                raise RuntimeError(
                    f"FAISS ntotal mismatch: index.ntotal={faiss_n} vs |corpus|={len(corpus)}"
                )
        except Exception:
            # Fail fast: dimension/size mismatch leads to corrupt retrieval quality
            raise

        # ------------------------- Encode queries (COMPRESSED) -------------------------
        if self.base_query_embeddings is not None:
            with torch.inference_mode():
                if self.compressor.autoencoder:
                    qx = self.base_query_embeddings.to(self.device)
                    qenc = self.compressor.autoencoder.encode(qx)
                    if isinstance(qenc, tuple):
                        qenc = qenc[0]
                    query_embeddings = qenc.detach().cpu().contiguous()
                else:
                    query_embeddings = self.base_query_embeddings.detach().cpu().contiguous()
        else:
            with torch.inference_mode():
                query_embeddings = self.compressor.encode_text(list(queries), compress=True)
        if not (isinstance(query_embeddings, torch.Tensor) and query_embeddings.ndim == 2):
            raise ValueError("Query embeddings must be a 2D tensor [N, D].")
        d_queries = int(query_embeddings.size(1))
        if d_queries != d_corpus:
            raise RuntimeError(
                f"Query/Corpus dim mismatch: queries D={d_queries} vs corpus D={d_corpus}."
            )

        # ------------------------- Retrieval loop (batched FAISS search) -------------------------
        top_k = int(self.retr_cfg.get("top_k", 10))
        cand_k = int(self.retr_cfg.get("candidate_k", top_k * 3 if use_chunking else top_k))
        per_doc_cap = int(self.retr_cfg.get("max_chunks_per_doc", 2)) if generate else 0

        # Batch search once
        D, I = self.retriever.search(query_embeddings, cand_k)  # D: [Q, cand_k]
        texts_store = getattr(self.retriever, "_texts", [])
        doc_ids_store = getattr(self.retriever, "_doc_ids", [])

        all_retrieved_docids: List[List[int]] = []
        answers: List[str] = []
        for qi in range(I.shape[0]):
            idxs = I[qi]
            scores = D[qi]
            # Aggregation MaxSim per original doc id
            agg: Dict[int, float] = {}
            # Build ranked_docids
            for didx, sc in zip(idxs.tolist(), scores.tolist()):
                if didx < 0:
                    continue
                doc_id = doc_ids_store[didx]
                if (doc_id not in agg) or (sc > agg[doc_id]):
                    agg[doc_id] = sc
            ranked_docids = sorted(agg, key=agg.get, reverse=True)[:top_k]
            all_retrieved_docids.append(ranked_docids)

            if generate:
                # Select candidate texts matching ranked_docids (respect per_doc_cap)
                used = {d: 0 for d in ranked_docids}
                selected: List[str] = []
                for didx in idxs.tolist():
                    if didx < 0:
                        continue
                    d_id = doc_ids_store[didx]
                    if d_id in used and used[d_id] < per_doc_cap:
                        selected.append(texts_store[didx])
                        used[d_id] += 1
                    if len(selected) >= max(1, per_doc_cap * len(ranked_docids)):
                        break
                ctx_for_llm = selected if selected else [texts_store[d] for d in idxs[:top_k].tolist()]
                ans = self.generator.generate(queries[qi], ctx_for_llm)
                answers.append(ans)
                # Debug logging kept concise
                self.logger.main.debug("[%d] Generated answer len=%d", qi, len(ans) if ans else 0)

        # ------------------------- Retrieval evaluation -------------------------
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

        # ------------------------- Generation evaluation -------------------------
        if generate and relevant_docs and len(queries) >= 100:
            gen_metrics = eval_generation(
                references=[r[0] for r in relevant_docs],
                candidates=answers,
                metrics=self.cfg.get("evaluation", {}).get("generation_metrics", ["ROUGE-L"]),
            )
            rprint("\n[Generation evaluation]\n")
            for m, d in gen_metrics.items():
                rprint(f"{m}: {d['mean']:.2f} (CI 95%: {d['ci_lower']:.2f}–{d['ci_upper']:.2f})")

        # ------------------------- Metrics packing -------------------------
        retr_stats = {}
        if hasattr(self.retriever, "get_stats"):
            retr_stats = self.retriever.get_stats(reset=False)

        # Dimensions reported out
        dim_out = d_corpus
        dim_in = int(self.compressor.input_dim)  # --- NEW: always SBERT native dim
        n_corpus = int(len(corpus))

        result = {
            "retrieval_metrics": ret_metrics,
            "retriever_stats":  retr_stats,
            "dim_in": dim_in,
            "dim_out": dim_out,
            "n_corpus": n_corpus,
            "n_queries": int(len(queries)),  # NEW: actual number of query samples used
            "ae_type": self.ae_type,
        }
        # --- NEW: terse banner for CSV cross-check
        self.logger.main.info("[Result] dim_in=%d dim_out=%d CR=%.2f× n=%d",
                              dim_in, dim_out, (dim_in / max(1, dim_out)), n_corpus)
        return result

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
    parser.add_argument("--max_samples", type=int, default=None,
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
        else [k for k in cfg.get("models", {}).keys() if k in {"vae", "dae", "cae", "none"}]
    )
    # Data
    queries, corpus, relevant = load_evaluation_data(args.dataset, max_samples=args.max_samples)
    # Pre-chunk + base embeddings reuse
    ch_cfg = cfg.get("chunking", {})
    use_chunking = bool(ch_cfg.get("enabled", False))
    corpus_texts = corpus
    corpus_doc_ids: List[int]
    if use_chunking:
        from utils.data_utils import prepare_inference_chunks
        chunks, chunk_index = prepare_inference_chunks(
            corpus_texts,
            mode=ch_cfg.get("mode", "sliding"),
            max_tokens=ch_cfg.get("max_tokens", 128),
            stride=ch_cfg.get("stride", 64),
            min_tokens=ch_cfg.get("min_tokens", 48),
            tokenizer_name=ch_cfg.get("tokenizer_name", cfg["embedding_model"]["name"]),
            index_out=ch_cfg.get("index_out"),
            store_chunk_text=ch_cfg.get("store_chunk_text", True),
        )
        corpus_texts = chunks
        corpus_doc_ids = chunk_index["doc_id"].astype(int).tolist()
    else:
        corpus_doc_ids = list(range(len(corpus_texts)))
    # Base compressor (no AE) for reuse
    base_compressor = EmbeddingCompressor(
        base_model_name=cfg["embedding_model"]["name"],
        autoencoder=None,
        device=resolve_device(cfg.get("training", {}).get("device")),
    )
    with torch.inference_mode():
        base_corpus_emb = base_compressor.encode_text(list(corpus_texts), compress=False)
        base_query_emb = base_compressor.encode_text(list(queries), compress=False)
    # Per-variant loop (apply AE encode on base embeddings)
    for ae in ae_variants:
        rprint(f"\n[bold cyan]==== PIPELINE ({ae.upper()}) ====\n[/]")
        _print_run_card(cfg, ae, generate=args.generate)
        runner = PipelineRunner(
            cfg, ae, log,
            pre_corpus_texts=corpus_texts,
            pre_corpus_doc_ids=corpus_doc_ids,
            base_corpus_embeddings=base_corpus_emb,
            base_query_embeddings=base_query_emb,
        )
        result = runner.process(queries, corpus_texts, relevant_docs=relevant, generate=args.generate)
        row = build_metrics_row(cfg, args, ae, result)
        _append_csv_row(args.metrics_csv, row)
                


if __name__ == "__main__":
    main()

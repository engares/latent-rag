# /utils/data_utils.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional, Dict, Sequence

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import random
from torch.utils.data import Subset

from pathlib import Path
from hashlib import sha1

import pandas as pd

from utils.chunk_utils import (
    sliding_window_chunker,
    semantic_window_chunker,
)


import pandas as pd
from utils.chunk_utils import build_inference_corpus, save_chunk_index

###############################################################################
# SBERT caching                                                               #
###############################################################################
def text_hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _compute_embeddings(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> torch.Tensor:
    """Return CLS embeddings as a `[N × D]` *float32* CPU tensor."""
    acc: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            acc.append(torch.from_numpy(emb))
    return torch.cat(acc).float()


def _texts_fingerprint(texts: Sequence[str]) -> str:
    h = sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:10]


def ensure_sbert_cache(
    texts: Sequence[str],
    *,
    model_name: str,
    cache_dir: str = "./data/SBERT",
    batch_size: int = 64,
    force: bool = False,
) -> torch.Tensor:
    os.makedirs(cache_dir, exist_ok=True)
    fp = _texts_fingerprint(texts)
    tag = model_name.split("/")[-1]
    path = os.path.join(cache_dir, f"sbert_{fp}_{tag}.pt")

    if not force and os.path.exists(path):
        return torch.load(path, map_location="cpu")

    print(f"[INFO] SBERT cache miss → encoding {len(texts):,} texts …")
    model = SentenceTransformer(model_name)
    emb = _compute_embeddings(texts, model, batch_size=batch_size)
    torch.save(emb, path)
    print(f"[OK]  SBERT embeddings saved → {path}")
    return emb

def _jaccard_sim(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    inter = a_set & b_set
    union = a_set | b_set
    return len(inter) / len(union) if union else 0.0

def _texts_fingerprint(texts: List[str]) -> str:
    """
    Devuelve un hash abreviado (10 hex) de la secuencia de textos.
    El orden de los textos importa, así garantizamos reproducibilidad.
    """
    h = sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:10]                # 40 bits bastan para colisiones muy raras

def prepare_inference_chunks(
    docs: Sequence[str],
    *,
    mode: str = "sliding",                     # "sliding" | "semantic"
    max_tokens: int = 128,
    stride: int = 64,
    min_tokens: int = 48,                      # solo usado en semantic
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_out: str | None = None,              # parquet opcional
    store_chunk_text: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """Chunk the given documents for test/inference and (optionally) persist an index.

    Returns:
        chunks: list[str] with chunk texts in corpus order.
        index : DataFrame indexed by chunk_id with:
                ['doc_id','tok_start','tok_end','char_start','char_end', 'chunk_text?'].
    """
    chunks: List[str] = []
    records: List[Dict] = []

    chunker = sliding_window_chunker if mode == "sliding" else semantic_window_chunker

    for doc_id, text in enumerate(docs):
        if not text:
            continue
        recs = (
            chunker(
                text,
                max_tokens=max_tokens,
                stride=stride,
                tokenizer_name=tokenizer_name,
            )
            if mode == "sliding"
            else semantic_window_chunker(
                text,
                max_tokens=max_tokens,
                stride=stride,
                min_tokens=min_tokens,
                tokenizer_name=tokenizer_name,
            )
        )
        for r in recs:
            cid = len(chunks)
            chunks.append(r.text)
            row = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "tok_start": r.tok_start,
                "tok_end": r.tok_end,
                "char_start": r.char_start,
                "char_end": r.char_end,
            }
            if store_chunk_text:
                row["chunk_text"] = r.text
            records.append(row)

    index = pd.DataFrame.from_records(records).set_index("chunk_id")

    if index_out:
        outp = Path(index_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        index.to_parquet(outp)

    return chunks, index


def ensure_sbert_cache(
    texts: List[str],
    *,
    model_name: str,
    cache_dir: str = "./data/SBERT",
    batch_size: int = 64,
    force: bool = False,
) -> torch.Tensor:
    """
    Calcula (o reutiliza) los embeddings de SBERT y los persiste en disco.

    Args
    ----
    texts       : Lista de cadenas a codificar.
    model_name  : Identificador HuggingFace / Sentence-Transformers del modelo.
    cache_dir   : Carpeta donde almacenar los .pt (creada si no existe).
    batch_size  : Tamaño de lote para _compute_embeddings.
    force       : Si True, rehace el cálculo aunque exista el fichero.

    Returns
    -------
    Tensor CPU float32 de dimensión [N × D].
    """
    os.makedirs(cache_dir, exist_ok=True)

    fp        = _texts_fingerprint(texts)
    model_tag = model_name.split("/")[-1]
    fname     = f"sbert_{fp}_{model_tag}.pt"
    path      = os.path.join(cache_dir, fname)

    if not force and os.path.exists(path):
        return torch.load(path, map_location="cpu")

    print(f"[INFO] SBERT cache miss → codificando {len(texts):,} textos …")
    st_model  = SentenceTransformer(model_name)
    emb       = _compute_embeddings(texts, st_model, batch_size=batch_size)
    torch.save(emb, path)
    print(f"[OK]  SBERT embeddings guardados → {path}")
    return emb

def ensure_uda_data( # EN DESUSO
    *,
    output_dir: str = "./data/",
    max_samples: Optional[int] = None,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    noise_std: float = 0.05,
    force: bool = False,
) -> None:
    """Genera (o reutiliza) los ficheros de embeddings para VAE, DAE y contraste."""
    os.makedirs(output_dir, exist_ok=True)

    vae_path = os.path.join(output_dir, "uda_vae_embeddings.pt")
    dae_path = os.path.join(output_dir, "uda_dae_embeddings.pt")
    contrastive_path = os.path.join(output_dir, "uda_contrastive_embeddings.pt")

    if (
        not force
        and os.path.exists(vae_path)
        and os.path.exists(dae_path)
        and os.path.exists(contrastive_path)
    ):
        print("[INFO] UDA embeddings ya preparados — nada que hacer.")
        return

    print("[INFO] Descargando / cargando UDA…")
    uda = load_dataset("qinchuanhui/UDA-QA", "nq")
    if max_samples is not None:
        uda = uda.select(range(min(max_samples, len(uda))))
    print(f"[INFO] UDA listo con {len(uda):,} ejemplos.")

    clean_texts: List[str] = []
    contrastive_triples: List[Tuple[str, str, str]] = []

    for i, ex in enumerate(uda["test"]): # TEMPORAL ******************************************************* CAMBIAR URGENTEMENTE
        q = ex.get("question", "").strip()
        pos = ex.get("long_answer", "").strip()
        if not q or not pos:
            continue

        neg = None
        for _ in range(10):
            j = random.randint(0, len(uda["test"]) - 1)
            if j == i:
                continue
            neg_cand = uda["test"][j].get("long_answer", "").strip()
            if not neg_cand:
                continue
            if _jaccard_sim(q, neg_cand) < 0.1:
                neg = neg_cand
                break

        if neg is None:
            continue

        clean_texts.extend((q, pos))             # query + positive answer
        contrastive_triples.append((q, pos, neg))


    print(f"[INFO] Tripletas contrastivas generadas: {len(contrastive_triples):,}")

    print(f"[INFO] Cargando SentenceTransformer '{base_model_name}' …")
    st_model = SentenceTransformer(base_model_name)

    print("[INFO] Generando embeddings VAE/DAE (positivos)…")
    target_emb = _compute_embeddings(clean_texts, st_model)

    if force or not os.path.exists(vae_path):
        torch.save({"input": target_emb, "target": target_emb.clone()}, vae_path)
        print(f"[OK]  VAE embeddings guardados → {vae_path}")

    if force or not os.path.exists(dae_path):
        input_emb = target_emb + torch.randn_like(target_emb) * noise_std
        torch.save({"input": input_emb, "target": target_emb}, dae_path)
        print(f"[OK]  DAE embeddings guardados → {dae_path}")

    if force or not os.path.exists(contrastive_path):
        print("[INFO] Generando embeddings de triples (query/pos/neg)…")
        qs, ps, ns = zip(*contrastive_triples)
        q_emb = _compute_embeddings(list(qs), st_model)
        p_emb = _compute_embeddings(list(ps), st_model)
        n_emb = _compute_embeddings(list(ns), st_model)
        torch.save({"query": q_emb, "positive": p_emb, "negative": n_emb}, contrastive_path)
        print(f"[OK]  Contrastive embeddings guardados → {contrastive_path}")

    print("[DONE] Preprocesado de UDA completo.")

def split_dataset(dataset: torch.utils.data.Dataset, val_split: float = 0.1, seed: int = 42) -> Tuple[Subset, Subset]:
    n_total = len(dataset)
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    n_val = int(n_total * val_split)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def ensure_squad_data(
    *,
    output_dir: str = "./data/SQUAD",
    max_samples: Optional[int] = None,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    noise_std: float = 0.05,
    include_unanswerable: bool = False,
    force: bool = False,
    chunk_max_tokens: int = 128,
    chunk_stride: int = 64,
    tokens_before: int = 32,
    tokens_after: int = 32,
    tokenizer_name: str | None = None,
    visualize: bool = False  # New parameter to control visualization
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import defaultdict

    os.makedirs(output_dir, exist_ok=True)

    vae_path = Path(output_dir, "squad_vae_embeddings.pt")
    dae_path = Path(output_dir, "squad_dae_embeddings.pt")
    con_path = Path(output_dir, "squad_contrastive_embeddings.pt")
    idx_path = Path(output_dir, "chunk_index.parquet")

    if (
        not force
        and all(p.exists() for p in (vae_path, dae_path, con_path, idx_path))
    ):
        print("[INFO] SQuAD embeddings already prepared — nothing to do.")
        return

    ds_name = "squad_v2" if include_unanswerable else "squad"
    print(f"[INFO] Loading {ds_name} …")
    squad = load_dataset(ds_name, split="train")
    if max_samples is not None:
        squad = squad.select(range(min(max_samples, len(squad))))
    print(f"[INFO] SQuAD loaded with {len(squad):,} examples.")

    print("[INFO] Chunking contexts (answer‑aware) …")
    chunks, chunk_index = build_chunked_corpus(
        squad,
        max_tokens=chunk_max_tokens,
        stride=chunk_stride,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokenizer_name=tokenizer_name or base_model_name,
    )

    save_chunk_index(idx_path, chunk_index)
    print(f"[OK]  Chunk index saved → {idx_path}")

    # Preindexar: {doc_id → [chunk_id con respuesta]}
    doc_chunks: Dict[int, List[int]] = defaultdict(list)
    for cid, row in chunk_index.iterrows():
        if row["contains_answer"]:
            doc_chunks[row["doc_id"]].append(cid)

    clean_texts: List[str] = []
    pos_chunks: List[str] = []
    doc_ids_skipped: List[int] = []

    for doc_id, ex in enumerate(squad):
        q = ex["question"].strip()
        if doc_id in doc_chunks and doc_chunks[doc_id]:
            cid = doc_chunks[doc_id][0]
            pos_chunk = chunks[cid]
            clean_texts.extend((q, pos_chunk))
            pos_chunks.append(pos_chunk)
        else:
            doc_ids_skipped.append(doc_id)

    if doc_ids_skipped and visualize:
        print(f"[WARN] No chunk found with answer for {len(doc_ids_skipped)} documents.")
        sns.set_theme()
        plt.figure(figsize=(8, 4))
        sns.histplot(doc_ids_skipped, bins=50, kde=False)
        plt.title("Distribución de documentos sin chunk con respuesta")
        plt.xlabel("doc_id")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    neg_chunks: List[str] = []
    rng = random.Random(42)
    for doc_id, pos in enumerate(pos_chunks):
        while True:
            cand_id = rng.randrange(len(chunks))
            if chunk_index.loc[cand_id, "doc_id"] != doc_id:
                cand_chunk = chunks[cand_id]
                if _jaccard_sim(pos, cand_chunk) < 0.1:
                    neg_chunks.append(cand_chunk)
                    break

    print("[INFO] Encoding SBERT embeddings …")
    print(f"[INFO] Codificando queries + positivos: {len(clean_texts)//2} pares.")
    cache_dir = Path(output_dir, "sbert_cache")
    target_emb = ensure_sbert_cache(
        clean_texts,
        model_name=base_model_name,
        cache_dir=str(cache_dir),
        batch_size=64,
    )

    q_emb = target_emb[0::2]
    p_emb = target_emb[1::2]

    print(f"[INFO] Codificando negativos: {len(neg_chunks)} chunks.")
    n_emb = ensure_sbert_cache(
        neg_chunks,
        model_name=base_model_name,
        cache_dir=str(cache_dir),
        batch_size=64,
    )

    if force or not vae_path.exists():
        torch.save({"input": target_emb, "target": target_emb.clone()}, vae_path)
        print(f"[OK]  VAE embeddings   → {vae_path}")

    if force or not dae_path.exists():
        noisy = target_emb + torch.randn_like(target_emb) * noise_std
        torch.save({"input": noisy, "target": target_emb}, dae_path)
        print(f"[OK]  DAE embeddings   → {dae_path}")

    if force or not con_path.exists():
        torch.save({"query": q_emb, "positive": p_emb, "negative": n_emb}, con_path)
        print(f"[OK]  Contrastive embeddings → {con_path}")

    print("[DONE] SQuAD preprocessing finished (chunk‑level).")



def _prepare_uda(cfg: dict) -> Dict[str, str]: # NOT IN USE
    common = dict(
        output_dir="./data/UDA",
        max_samples=cfg["data"].get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
        force=False,
    )
    ensure_uda_data(**common)
    output_dir = common["output_dir"]
    return {
        "vae": os.path.join(output_dir, cfg["models"]["vae"]["dataset_file"]),
        "dae": os.path.join(output_dir, cfg["models"]["dae"]["dataset_file"]),
        "cae": os.path.join(output_dir, cfg["models"]["contrastive"]["dataset_file"]),
    }


def _prepare_squad(cfg: dict) -> Dict[str, str]:
    data_cfg = cfg["data"]
    common = dict(
        output_dir=cfg["paths"]["data_dir"],
        max_samples=data_cfg.get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
        noise_std=0.05,
        include_unanswerable=data_cfg.get("include_unanswerable", False),
        force=False,
    )
    ensure_squad_data(**common)
    output_dir = common["output_dir"]
    return {
        "vae": os.path.join(output_dir, cfg["models"]["vae"]["dataset_file"]),
        "dae": os.path.join(output_dir, cfg["models"]["dae"]["dataset_file"]),
        "cae": os.path.join(output_dir, cfg["models"]["contrastive"]["dataset_file"]),
    }


def prepare_datasets(
    cfg: dict,
    *,
    variant: str,
    dataset_override: Optional[str] = None,
) -> str:
    """Ensure dataset tensors exist and return path for requested variant.

    Args:
        cfg: Parsed YAML config dict (must include data and embedding_model).
        variant: One of `{"vae", "dae", "cae"}.
        dataset_override: If provided, forces `"uda" or "squad"

    Returns:
        The filesystem path to the tensor file corresponding to the *variant*.
    """
    variant = variant.lower()
    assert variant in {"vae", "dae", "cae"}, "variant must be vae, dae or cae"

    ds_name = (dataset_override or cfg.get("data", {}).get("dataset", "squad")).lower()
    if ds_name == "squad":
        paths = _prepare_squad(cfg)
    elif ds_name == "uda":
        paths = _prepare_uda(cfg)
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    path = paths[variant]
    if not Path(path).exists():
        raise FileNotFoundError(f"Expected dataset file not found: {path}")
    return path




def load_eval_queries_from_squad(
    version: str = "v1",
    split: str = "validation",
    max_samples: Optional[int] = None,
    dedup: bool = True,
) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    Prepara triples (queries, corpus, relevantes) para evaluación de retrieval.

    Args:
        version: "v1" o "v2"
        split: "train" o "validation"
        max_samples: límite de queries
        dedup: si True, elimina contextos repetidos del corpus

    Returns:
        queries, corpus, relevant_docs (1 a 1 con queries)
    """
    ds_name = "squad_v2" if version == "v2" else "squad"
    ds = load_dataset(ds_name, split=split)

    queries, contexts, relevant = [], [], []

    for ex in ds:
        q = ex["question"].strip()
        c = ex["context"].strip()

        # descartar preguntas sin respuesta si es v2
        if version == "v2":
            has_answer = bool(ex["answers"]["answer_start"])
            if not has_answer:
                continue

        queries.append(q)
        contexts.append(c)
        relevant.append([c])  # relevante = ese contexto

        if max_samples and len(queries) >= max_samples:
            break

    corpus = list(set(contexts)) if dedup else contexts
    return queries, corpus, relevant


def load_evaluation_data(dataset: str, max_samples: int = 200):
    if dataset == "squad":
        return load_eval_queries_from_squad(
            version="v1", split="validation", max_samples=max_samples
        )
    elif dataset == "uda":
        raise NotImplementedError("TODO: soporte UDA")
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")
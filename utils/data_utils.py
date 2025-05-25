 # /utils/data_utils.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import random
from torch.utils.data import Subset

from pathlib import Path
from typing import Dict, Optional

def _compute_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> torch.Tensor:
    """Devuelve un tensor CPU float32 [N × D] con los CLS-embeddings."""
    chunks: List[torch.Tensor] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            chunks.append(torch.from_numpy(emb))
    return torch.cat(chunks, dim=0).float()

def _jaccard_sim(a: str, b: str) -> float:
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    inter = a_set & b_set
    union = a_set | b_set
    return len(inter) / len(union) if union else 0.0

def ensure_uda_data(
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
    output_dir: str = "./data",
    max_samples: Optional[int] = None,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    noise_std: float = 0.05,
    include_unanswerable: bool = False,
    force: bool = False,
) -> None:
    """Generate VAE / DAE / CAE embedding tensors from **SQuAD v1/v2**.

    The tensors are stored on disk using the same structure as the UDA helper,
    so training scripts remain unchanged.  Filenames:

        squad_vae_embeddings.pt
        squad_dae_embeddings.pt
        squad_contrastive_embeddings.pt
    """
    os.makedirs(output_dir, exist_ok=True)

    vae_path         = os.path.join(output_dir, "squad_vae_embeddings.pt")
    dae_path         = os.path.join(output_dir, "squad_dae_embeddings.pt")
    contrastive_path = os.path.join(output_dir, "squad_contrastive_embeddings.pt")

    # --------------------------------------------------------------------- #
    #  Early exit if everything is already cached                           #
    # --------------------------------------------------------------------- #
    if (
        not force
        and all(os.path.exists(p) for p in (vae_path, dae_path, contrastive_path))
    ):
        print("[INFO] SQuAD embeddings already prepared nothing to do.")
        return

    # --------------------------------------------------------------------- #
    #  1. Load SQuAD                                                        #
    # --------------------------------------------------------------------- #
    ds_name = "squad_v2" if include_unanswerable else "squad"
    print(f"[INFO] Loading {ds_name} …")
    squad = load_dataset(ds_name, split="train")
    if max_samples is not None:
        squad = squad.select(range(min(max_samples, len(squad))))
    print(f"[INFO] SQuAD loaded with {len(squad):,} examples.")

    # --------------------------------------------------------------------- #
    #  2. Build positive contexts and contrastive triples                   #
    # --------------------------------------------------------------------- #
    clean_texts: List[str] = []
    contrastive_triples: List[Tuple[str, str, str]] = []

    for i, ex in enumerate(squad):
        q   = ex["question"].strip()
        ctx = ex["context"].strip()
        if not q or not ctx:
            continue

        # ----------------- simple negative mining ------------------------ #
        neg = None
        for _ in range(10):
            j = random.randint(0, len(squad) - 1)
            if j == i:
                continue
            neg_ctx = squad[j]["context"].strip()
            if neg_ctx and _jaccard_sim(q, neg_ctx) < 0.1:
                neg = neg_ctx
                break
        if neg is None:
            continue

        clean_texts.extend([q, ctx])              # query + positive context
        contrastive_triples.append((q, ctx, neg))

    print(f"[INFO] Contrastive triples generated: {len(contrastive_triples):,}")

    # --------------------------------------------------------------------- #
    #  3. Encode with SBERT                                                 #
    # --------------------------------------------------------------------- #
    print(f"[INFO] Loading SBERT '{base_model_name}' …")
    st_model = SentenceTransformer(base_model_name)

    print("[INFO] Encoding queries + positive contexts …")
    target_emb = _compute_embeddings(clean_texts, st_model)   # shape [N × D]

    # --------------------------------------------------------------------- #
    #  4. Save VAE and DAE variants                                         #
    # --------------------------------------------------------------------- #
    if force or not os.path.exists(vae_path):
        torch.save({"input": target_emb, "target": target_emb.clone()}, vae_path)
        print(f"[OK]  VAE embeddings   → {vae_path}")

    if force or not os.path.exists(dae_path):
        input_emb = target_emb + torch.randn_like(target_emb) * noise_std
        torch.save({"input": input_emb, "target": target_emb}, dae_path)
        print(f"[OK]  DAE embeddings   → {dae_path}")

    # --------------------------------------------------------------------- #
    #  5. Save contrastive triplets                                         #
    # --------------------------------------------------------------------- #
    if force or not os.path.exists(contrastive_path):
        print("[INFO] Encoding triplets …")
        qs, ps, ns = zip(*contrastive_triples)
        q_emb = _compute_embeddings(list(qs), st_model)
        p_emb = _compute_embeddings(list(ps), st_model)
        n_emb = _compute_embeddings(list(ns), st_model)
        torch.save(
            {"query": q_emb, "positive": p_emb, "negative": n_emb},
            contrastive_path,
        )
        print(f"[OK]  Contrastive embeddings → {contrastive_path}")

    print("[DONE] SQuAD preprocessing finished.")



def _prepare_uda(cfg: dict) -> Dict[str, str]:
    common = dict(
        output_dir="./data/UDA",
        max_samples=cfg["data"].get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
        force=False,
    )
    ensure_uda_data(**common)
    return {
        "vae": "./data/UDA/uda_vae_embeddings.pt",
        "dae": "./data/UDA/uda_dae_embeddings.pt",
        "cae": "./data/UDA/uda_contrastive_embeddings.pt",
    }


def _prepare_squad(cfg: dict) -> Dict[str, str]:
    data_cfg = cfg["data"]
    common = dict(
        output_dir="./data/SQUAD/",
        max_samples=data_cfg.get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
        noise_std=0.05,
        include_unanswerable=data_cfg.get("include_unanswerable", False),
        force=False,
    )
    ensure_squad_data(**common)
    return {
        "vae": "./data/SQUAD/squad_vae_embeddings.pt",
        "dae": "./data/SQUAD/squad_dae_embeddings.pt",
        "cae": "./data/SQUAD/squad_contrastive_embeddings.pt",
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
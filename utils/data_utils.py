import os
from typing import List, Tuple, Optional

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import random
from torch.utils.data import Subset

###############################################################################
#  UDA → Sentence-Transformer embeddings (*.pt)                               #
###############################################################################

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
    output_dir: str = "./data",
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

        clean_texts.append(pos)
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

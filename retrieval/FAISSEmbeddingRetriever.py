# retrieval/FAISSEmbeddingRetriever.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import json
import numpy as np
import faiss
import torch
import torch.nn.functional as F

class FAISSEmbeddingRetriever:
    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[str | Path] = None,
        index_type: str = "hnsw",      # "flatip" | "hnsw" | "ivfpq"
        use_gpu: bool = False,
    ):
        self.d = int(embedding_dim)
        self.index_type = index_type
        self.path = Path(index_path) if index_path else None
        self.use_gpu = use_gpu

        self._texts: List[str] = []
        self._doc_ids: List[int] = []

        # 1) construir índice vacío con métrica IP (coseno ≈ IP con normalización L2)
        self.index = self._build_index(self.d, self.index_type)

        # 2) si hay índice en disco, cargarlo (y sidecar)
        if self.path and self.path.exists():
            cpu_index = faiss.read_index(str(self.path), faiss.IO_FLAG_MMAP)
            # si la dimensión no cuadra, ignora el índice en disco
            if cpu_index.d == self.d:
                self.index = cpu_index
                self._load_metadata()
            else:
                # índice incompatible: lo descartamos
                self._texts, self._doc_ids = [], []

        # 3) opcional: mover a GPU
        self.gpu_enabled = False
        if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
            try:
                if faiss.get_num_gpus() > 0:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    self.gpu_enabled = True
                else:
                    print("[WARN] FAISS GPU no disponible. Uso CPU.")
            except Exception:
                print("[WARN] FAISS GPU no disponible. Uso CPU.")

    def _build_index(self, d: int, kind: str) -> faiss.Index:
        if kind == "flatip":
            return faiss.IndexFlatIP(d)
        if kind == "hnsw":
            return faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        if kind == "ivfpq":
            quant = faiss.IndexFlatIP(d)
            return faiss.IndexIVFPQ(quant, d, 4096, 16, 8)
        raise ValueError(f"Index type not supported: {kind}")

    def _meta_path(self) -> Path:
        assert self.path is not None
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def _save_metadata(self) -> None:
        if not self.path:
            return
        meta = {"texts": self._texts, "doc_ids": self._doc_ids}
        self._meta_path().parent.mkdir(parents=True, exist_ok=True)
        with self._meta_path().open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def _load_metadata(self) -> None:
        if not self.path:
            return
        mp = self._meta_path()
        if not mp.exists():
            self._texts, self._doc_ids = [], []
            return
        with mp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self._texts  = list(meta.get("texts", []))
        self._doc_ids = list(meta.get("doc_ids", []))

    def build(
        self,
        embeddings: torch.Tensor,           # [N, D] float32 (CPU/GPU)
        texts: Sequence[str],
        doc_ids: Sequence[int] | None = None,
        train: bool = True,
    ) -> None:
        assert len(embeddings) == len(texts), "len mismatch (embeddings vs texts)"
        if doc_ids is not None:
            assert len(texts) == len(doc_ids), "len mismatch (texts vs doc_ids)"

        # Asegurar CPU, float32, C-contiguous
        x = embeddings.detach().cpu().numpy().astype("float32", copy=False)
        # Normalizar L2 para IP≈coseno
        faiss.normalize_L2(x)

        # Si el índice cargado no es compatible, reconstruir con la nueva D
        if getattr(self.index, "d", None) != x.shape[1]:
            self.d = int(x.shape[1])
            self.index = self._build_index(self.d, self.index_type)
            self._texts, self._doc_ids = [], []  # limpiamos metadatos si cambiamos de D

        # Entrenar si aplica
        if train and hasattr(self.index, "train") and not self.index.is_trained:
            self.index.train(x)

        # Añadir vectores
        self.index.add(x)

        # Metadatos
        self._texts.extend(list(texts))
        self._doc_ids.extend(list(doc_ids) if doc_ids is not None else [-1] * len(texts))

        # Guardar SIEMPRE en CPU
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cpu_index = self.index
            if getattr(self, "gpu_enabled", False) and hasattr(faiss, "index_gpu_to_cpu"):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(self.path))
            self._save_metadata()

    def retrieve(self, query_emb: torch.Tensor, top_k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        q = query_emb.detach().cpu().numpy().astype("float32", copy=False)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)  # para IP, D = similitudes (mayor=mejor)
        idxs = I[0].tolist()

        if not self._texts or not self._doc_ids:
            self._load_metadata()

        return (
            [self._texts[i] for i in idxs],
            D[0].tolist(),
            [self._doc_ids[i] for i in idxs],
        )

# retrieval/FAISSEmbeddingRetriever.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, List, Optional, Dict, Any
import json

import numpy as np
import faiss
import torch


class FAISSEmbeddingRetriever:
    """
    Indexador y recuperador FAISS para embeddings densos.

    Características:
      - Métrica INNER_PRODUCT (IP) con normalización L2: IP ≈ coseno.
      - Persistencia opcional de índice + metadatos (textos, doc_ids y fingerprint).
      - Reconstrucción automática si el índice en disco no es compatible
        (dimensión, modelo de embedding, AE, chunking, etc.).
      - Comprobación de sanidad tras indexar (self-search de un vector).
    """

    # ------------------------------- Init ---------------------------------- #
    def __init__(
        self,
        embedding_dim: int,
        index_path: Optional[str | Path] = None,
        index_type: str = "hnsw",   # "flatip" | "hnsw" | "ivfpq"
        use_gpu: bool = False,
        *,
        hnsw_M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ):
        self.d = int(embedding_dim)
        self.index_type = index_type
        self.path = Path(index_path) if index_path else None
        self.use_gpu = use_gpu

        self.hnsw_M = int(hnsw_M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

        # Memoria de metadatos
        self._texts: List[str] = []
        self._doc_ids: List[int] = []
        self.meta_fp: Dict[str, Any] = {}

        # Índice (CPU por defecto)
        self.index = self._build_index(self.d, self.index_type)

        # Cargar índice si existe (lo validaremos en build())
        if self.path and self.path.exists():
            try:
                cpu_index = faiss.read_index(str(self.path), faiss.IO_FLAG_MMAP)
                # No cambiamos aún: la compatibilidad se decide en build()
                self.index = cpu_index
                self._load_metadata()
            except Exception:
                # Archivo corrupto o incompatible → empezamos limpio
                self.index = self._build_index(self.d, self.index_type)
                self._texts, self._doc_ids, self.meta_fp = [], [], {}

        # Mover a GPU si procede
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

        # Ajuste de parámetros HNSW
        self._maybe_set_hnsw_params(self.index)

    # ------------------------------ Helpers -------------------------------- #
    def _build_index(self, d: int, kind: str) -> faiss.Index:
        if kind == "flatip":                      # Exacto, IP
            return faiss.IndexFlatIP(d)
        if kind == "hnsw":                        # Aproximado, IP
            idx = faiss.IndexHNSWFlat(d, self.hnsw_M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.ef_construction
            idx.hnsw.efSearch = self.ef_search
            return idx
        if kind == "ivfpq":                       # Cuantización (no recomend. para N pequeño)
            quant = faiss.IndexFlatIP(d)
            return faiss.IndexIVFPQ(quant, d, 4096, 16, 8)
        raise ValueError(f"Index type not supported: {kind}")

    def _maybe_set_hnsw_params(self, index: faiss.Index) -> None:
        # Ajusta efSearch si es HNSW
        if isinstance(index, faiss.IndexHNSW):
            index.hnsw.efSearch = self.ef_search
        # Si está en GPU y es HNSW, FAISS gestiona internamente los equivalentes.

    def _meta_path(self) -> Path:
        assert self.path is not None
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def _save_metadata(self) -> None:
        if not self.path:
            return
        meta = {
            "texts": self._texts,
            "doc_ids": self._doc_ids,
            "fingerprint": self.meta_fp,
        }
        self._meta_path().parent.mkdir(parents=True, exist_ok=True)
        with self._meta_path().open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def _load_metadata(self) -> None:
        if not self.path:
            return
        mp = self._meta_path()
        if not mp.exists():
            self._texts, self._doc_ids, self.meta_fp = [], [], {}
            return
        with mp.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self._texts = list(meta.get("texts", []))
        self._doc_ids = list(meta.get("doc_ids", []))
        self.meta_fp = dict(meta.get("fingerprint", {}))

    @staticmethod
    def _normalize_l2_inplace(x: np.ndarray) -> None:
        # IP ≈ coseno si normalizamos L2
        faiss.normalize_L2(x)

    @staticmethod
    def _fingerprint(
        *,
        d: int,
        embedding_model: Optional[str],
        ae_type: Optional[str],
        latent_dim: Optional[int],
        chunking_cfg: Optional[Dict[str, Any]],
        metric: str = "ip",
        normalize_l2: bool = True,
        version: int = 1,
    ) -> Dict[str, Any]:
        ch = chunking_cfg or {}
        return {
            "d": int(d),
            "embedding_model": embedding_model,
            "ae_type": ae_type,
            "latent_dim": int(latent_dim) if latent_dim is not None else None,
            "chunking": {
                "enabled": bool(ch.get("enabled", False)),
                "mode": ch.get("mode", "sliding"),
                "max_tokens": int(ch.get("max_tokens", 128)) if ch.get("max_tokens") is not None else None,
                "stride": int(ch.get("stride", 64)) if ch.get("stride") is not None else None,
                "min_tokens": int(ch.get("min_tokens", 48)) if ch.get("min_tokens") is not None else None,
            },
            "metric": metric,
            "normalize_l2": bool(normalize_l2),
            "version": int(version),
        }

    def _compatible(self, current_fp: Dict[str, Any]) -> bool:
        m = self.meta_fp or {}
        keys = ["d", "embedding_model", "ae_type", "latent_dim", "metric", "normalize_l2", "version"]
        if any(m.get(k) != current_fp.get(k) for k in keys):
            return False
        mch = (m.get("chunking") or {})
        cch = (current_fp.get("chunking") or {})
        for k in ["enabled", "mode", "max_tokens", "stride", "min_tokens"]:
            if mch.get(k) != cch.get(k):
                return False
        return True

    # ------------------------------- Build --------------------------------- #
    def build(
        self,
        embeddings: torch.Tensor,           # [N, D] (CPU/GPU), float32 preferido
        texts: Sequence[str],
        doc_ids: Sequence[int] | None = None,
        train: bool = True,
        *,
        embedding_model_name: Optional[str] = None,
        ae_type: Optional[str] = None,
        latent_dim: Optional[int] = None,
        chunking_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert len(embeddings) == len(texts), "len mismatch (embeddings vs texts)"
        if doc_ids is not None:
            assert len(texts) == len(doc_ids), "len mismatch (texts vs doc_ids)"

        # 1) Preparar X (CPU, float32, C-contiguous) + normalización L2
        x = embeddings.detach().cpu().numpy().astype("float32", copy=False)
        self._normalize_l2_inplace(x)

        # 2) Fingerprint actual
        cur_fp = self._fingerprint(
            d=int(x.shape[1]),
            embedding_model=embedding_model_name,
            ae_type=ae_type,
            latent_dim=latent_dim,
            chunking_cfg=chunking_cfg,
            metric="ip",
            normalize_l2=True,
            version=1,
        )

        # 3) Validar índice en disco; si incompatible → reconstruir limpio
        rebuild = False
        if hasattr(self.index, "d") and int(getattr(self.index, "d")) != cur_fp["d"]:
            rebuild = True
        if (self.path and self.path.exists()) and (not self._compatible(cur_fp)):
            rebuild = True

        if rebuild:
            # Volver a CPU si estamos en GPU
            if getattr(self, "gpu_enabled", False) and hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    self.index = faiss.index_gpu_to_cpu(self.index)
                except Exception:
                    pass
            # Reconstruir
            self.index = self._build_index(cur_fp["d"], self.index_type)
            self._texts, self._doc_ids, self.meta_fp = [], [], {}
            self._maybe_set_hnsw_params(self.index)
            # Volver a GPU si aplica
            if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
                try:
                    if faiss.get_num_gpus() > 0:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                        self.gpu_enabled = True
                except Exception:
                    self.gpu_enabled = False

        # 4) Entrenar si aplica (IVF/IVFPQ) → con X normalizado
        if train and hasattr(self.index, "train") and not self.index.is_trained:
            self.index.train(x)

        # 5) Añadir vectores
        self.index.add(x)

        # 6) Comprobación de sanidad mínima: self-search
        try:
            D_chk, I_chk = self.index.search(x[:1], 1)
            if I_chk.shape[0] == 0 or I_chk[0, 0] != 0:
                print("[ERROR] FAISS sanity check failed; rebuilding index")
                # Reconstruir y re-add
                if getattr(self, "gpu_enabled", False) and hasattr(faiss, "index_gpu_to_cpu"):
                    try:
                        self.index = faiss.index_gpu_to_cpu(self.index)
                    except Exception:
                        pass
                self.index = self._build_index(cur_fp["d"], self.index_type)
                self._maybe_set_hnsw_params(self.index)
                if train and hasattr(self.index, "train") and not self.index.is_trained:
                    self.index.train(x)
                self.index.add(x)
        except Exception:
            # Si algo falla en el sanity, reconstruimos igualmente
            self.index = self._build_index(cur_fp["d"], self.index_type)
            self._maybe_set_hnsw_params(self.index)
            if train and hasattr(self.index, "train") and not self.index.is_trained:
                self.index.train(x)
            self.index.add(x)

        # 7) Metadatos en memoria
        self._texts.extend(list(texts))
        self._doc_ids.extend(list(doc_ids) if doc_ids is not None else [-1] * len(texts))
        self.meta_fp = cur_fp

        # 8) Persistencia (si se configuró path). Siempre escribir en CPU.
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cpu_index = self.index
            if getattr(self, "gpu_enabled", False) and hasattr(faiss, "index_gpu_to_cpu"):
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                except Exception:
                    cpu_index = self.index  # fallback
            faiss.write_index(cpu_index, str(self.path))
            self._save_metadata()

        # 9) Log breve (útil para auditoría)
        try:
            ntotal = getattr(self.index, "ntotal", -1)
            print(f"[FAISS] type={type(self.index).__name__} d={cur_fp['d']} ntotal={ntotal} metric=IP normL2=True")
        except Exception:
            pass

    # ------------------------------ Retrieve ------------------------------- #
    def retrieve(self, query_emb: torch.Tensor, top_k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
        q = query_emb.detach().cpu().numpy().astype("float32", copy=False)
        self._normalize_l2_inplace(q)

        D, I = self.index.search(q, top_k)  # IP → D = similitudes (mayor=mejor)
        idxs = I[0].tolist()

        # Protección si faltan metadatos (compatibilidad)
        if not self._texts or not self._doc_ids:
            self._load_metadata()

        texts = [self._texts[i] for i in idxs]
        scores = D[0].tolist()
        docids = [self._doc_ids[i] for i in idxs]
        return texts, scores, docids

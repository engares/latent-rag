# /retrieval/retriever.py

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from typing import List, Tuple, Literal

SimilarityMetric = Literal["cosine", "euclidean", "mahalanobis"]

###############################################################################
#  MATRICES DE SIMILITUD                                                      #
###############################################################################

def cosine_similarity_matrix(
    query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
) -> np.ndarray:
    """Matriz [Q × N] con similitud coseno."""
    q_norm = F.normalize(query_embeddings, p=2, dim=1)
    d_norm = F.normalize(doc_embeddings, p=2, dim=1)
    sim = torch.mm(q_norm, d_norm.T)  # [Q, N]
    return sim.cpu().numpy()


def euclidean_similarity_matrix(
    query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor
) -> np.ndarray:
    """Matriz [Q × N] con "+similitud" inversa a la distancia euclídea."""
    q = query_embeddings.unsqueeze(1)  # [Q, 1, D]
    d = doc_embeddings.unsqueeze(0)  # [1, N, D]
    dist = torch.norm(q - d, dim=2)  # [Q, N]
    return (-dist).cpu().numpy()  # valores altos = más similares


def mahalanobis_similarity_matrix(
    query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, eps: float = 1e-6
) -> np.ndarray:
    """Matriz [Q × N] con similitud inversa de la distancia de Mahalanobis.

    *Se ajusta la matriz de covarianza exclusivamente sobre los documentos* para
    preservar la simetría deseada en el espacio de recuperación.
    """
    # -- Datos a NumPy -------------------------------------------------------
    d_np: np.ndarray = doc_embeddings.cpu().numpy()
    q_np: np.ndarray = query_embeddings.cpu().numpy()

    # -- Precisión (inversa de la covarianza) -------------------------------
    # EmpiricalCovariance añade regularización de Ledoit‑Wolf si el sistema lo
    # necesita, pero incluimos un término eps para garantizar invertibilidad.
    emp = EmpiricalCovariance(assume_centered=False).fit(d_np)
    VI: np.ndarray = emp.precision_ + eps * np.eye(d_np.shape[1], dtype=np.float64)

    # -- Distancias ----------------------------------------------------------
    diff = q_np[:, None, :] - d_np[None, :, :]  # [Q, N, D]
    # einsum: (q n d, d d, q n d) → (q n)
    dist = np.einsum("qnd,dd,qnd->qn", diff, VI, diff, optimize=True)

    # -- Convertir a "similitud" (negativo de la distancia) -----------------
    sim = -dist  # altos valores ⇒ mayor similitud
    return sim.astype(np.float32)


###############################################################################
#  FRONT‑END DE RECUPERACIÓN                                                  #
###############################################################################

def compute_similarity(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    metric: SimilarityMetric = "cosine",
) -> np.ndarray:
    """Devuelve matriz [Q × N] según la métrica solicitada y valida formas."""

    if query_embeddings.dim() == 1:
        query_embeddings = query_embeddings.unsqueeze(0)
    if doc_embeddings.dim() == 1:
        doc_embeddings = doc_embeddings.unsqueeze(0)

    funcs = {
        "cosine": cosine_similarity_matrix,
        "euclidean": euclidean_similarity_matrix,
        "mahalanobis": mahalanobis_similarity_matrix,
    }

    if metric not in funcs:
        raise ValueError(f"Métrica de similitud '{metric}' no soportada.")

    sim = funcs[metric](query_embeddings, doc_embeddings)

    # -------- Validación de forma -----------------------------------------
    q, d = query_embeddings.shape[0], doc_embeddings.shape[0]
    if sim.shape != (q, d):
        raise RuntimeError(
            f"Shape mismatch: expected ({q}, {d}) got {sim.shape} for metric '{metric}'."
        )
    return sim


def retrieve_top_k(
    query_embedding: torch.Tensor,
    doc_embeddings: torch.Tensor,
    doc_texts: List[str],
    k: int = 5,
    metric: SimilarityMetric = "cosine",
) -> List[Tuple[str, float]]:
    """Recupera los *k* documentos con mayor similitud."""

    sim_scores = compute_similarity(query_embedding, doc_embeddings, metric)  # [Q, N]
    # Suponemos única consulta (Q = 1) para esta utilidad.
    if sim_scores.shape[0] != 1:
        raise ValueError("Esta función está pensada para una única consulta.")

    top_idx = sim_scores[0].argsort()[::-1][:k]
    return [(doc_texts[i], float(sim_scores[0, i])) for i in top_idx]

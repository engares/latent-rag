import torch
import torch.nn.functional as F
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from typing import List, Tuple, Literal

SimilarityMetric = Literal["cosine", "euclidean", "mahalanobis"]

def cosine_similarity_matrix(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> np.ndarray:
    """Calcula la matriz de similitud coseno entre embeddings de consulta y documentos."""
    q_norm = F.normalize(query_embeddings, p=2, dim=1)
    d_norm = F.normalize(doc_embeddings, p=2, dim=1)
    return torch.mm(q_norm, d_norm.T).cpu().numpy()

def euclidean_similarity_matrix(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> np.ndarray:
    """Calcula la matriz de similitud inversa de distancias euclídeas."""
    q = query_embeddings.unsqueeze(1)  # [Q, 1, D]
    d = doc_embeddings.unsqueeze(0)    # [1, N, D]
    distances = torch.norm(q - d, dim=2)  # [Q, N]
    return -distances.cpu().numpy()  # Negative so greater values, greater similarity

def mahalanobis_similarity_matrix(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> np.ndarray:
    """Calcula la matriz de similitud inversa de distancias de Mahalanobis."""
    doc_np = doc_embeddings.cpu().numpy()
    cov = EmpiricalCovariance().fit(doc_np)
    mahal_dist = cov.mahalanobis(query_embeddings.cpu().numpy(), doc_np)
    return -mahal_dist  # NSame as eucliden


def compute_similarity( query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, metric: SimilarityMetric = "cosine") -> np.ndarray:
    """Calcula la matriz de similitud entre embeddings de consulta y documentos."""
    if metric == "cosine":
        return cosine_similarity_matrix(query_embeddings, doc_embeddings)
    elif metric == "euclidean":
        return euclidean_similarity_matrix(query_embeddings, doc_embeddings)
    elif metric == "mahalanobis":
        return mahalanobis_similarity_matrix(query_embeddings, doc_embeddings)
    else:
        raise ValueError(f"Métrica de similitud '{metric}' no soportada.")

def retrieve_top_k( query_embedding: torch.Tensor, doc_embeddings: torch.Tensor, doc_texts: List[str], k: int = 5, metric: SimilarityMetric = "cosine" ) -> List[Tuple[str, float]]:
    """Recupera los Top-k documentos más similares."""
    sim_scores = compute_similarity(query_embedding, doc_embeddings, metric)
    top_indices = sim_scores[0].argsort()[::-1][:k]  # Top-k mayores valores
    return [(doc_texts[i], sim_scores[0][i]) for i in top_indices]

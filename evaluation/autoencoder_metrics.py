import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Tuple


def evaluate_reconstruction_loss(x: torch.Tensor, x_reconstructed: torch.Tensor, reduction: str = "mean") -> float:
    """Calcula el error de reconstrucción (MSE)."""
    loss_fn = torch.nn.MSELoss(reduction=reduction)
    return loss_fn(x_reconstructed, x).item()

def visualize_embeddings(embeddings: torch.Tensor, labels: torch.Tensor = None, title: str = "Embeddings Visualization"):
    """Proyección 2D de los embeddings usando t-SNE."""
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels.cpu().numpy(), palette="tab10")
    else:
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1])

    plt.title(title)
    plt.show()
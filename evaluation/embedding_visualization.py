import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Tuple



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
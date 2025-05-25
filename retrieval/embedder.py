# retrieval/embedder.py

from sentence_transformers import SentenceTransformer
import torch


class EmbeddingCompressor:
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        autoencoder: torch.nn.Module = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained SBERT model (includes pooling + normalization)
        self.model = SentenceTransformer(base_model_name, device=self.device)

        # Optional autoencoder for compression (VAE, DAE, etc.)
        self.autoencoder = autoencoder.to(self.device) if autoencoder else None
        if self.autoencoder:
            self.autoencoder.eval()

    def encode_text(self, texts: list[str], compress: bool = True) -> torch.Tensor:
        """Returns SBERT embeddings, optionally compressed with an autoencoder.

        Args:
            texts: List of input strings to encode.
            compress: Whether to apply the autoencoder (if available).

        Returns:
            A float32 tensor [N × D] on CPU.
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).to(self.device)

            if self.autoencoder and compress:
                encoded = self.autoencoder.encode(embeddings)
                if isinstance(encoded, tuple):  # VAE returns (mu, logvar)
                    encoded = encoded[0]        # use mean as latent code
                return encoded.cpu()

            return embeddings.cpu()

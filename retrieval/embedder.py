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

        # 1) SBERT
        self.model = SentenceTransformer(base_model_name, device=self.device)
        self.input_dim = int(getattr(self.model, "get_sentence_embedding_dimension", lambda: 0)())
        if self.input_dim <= 0:
            raise ValueError("Cannot infer SBERT embedding dimension")

        # 2) Autoencoder (opcional)
        self.autoencoder = autoencoder.to(self.device) if autoencoder else None
        if self.autoencoder:
            self.autoencoder.eval()
            self.latent_dim = int(getattr(self.autoencoder, "latent_dim", self.input_dim))
        else:
            self.latent_dim = self.input_dim  # sin compresión

    def encode_text(self, texts: list[str], compress: bool = True) -> torch.Tensor:
        """Devuelve embeddings en CPU: [N, D_out] con D_out=latent_dim si compress=True, si no D_out=input_dim."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True  # unit-norm
            ).to(self.device).float()      # asegura fp32

            if self.autoencoder and compress:
                encoded = self.autoencoder.encode(embeddings)
                if isinstance(encoded, tuple):  # VAE: (mu, logvar)
                    encoded = encoded[0]
                return encoded.detach().cpu().contiguous()

            return embeddings.detach().cpu().contiguous()

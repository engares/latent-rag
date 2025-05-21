# retrieval/embedder.py

import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingCompressor:
    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        autoencoder: torch.nn.Module = None,
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo base de embeddings (ej. BERT, SBERT, DPR)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModel.from_pretrained(base_model_name).to(self.device)
        self.model.eval()

        # Autoencoder (inyectado externamente, puede ser VAE, DAE, etc.)
        self.autoencoder = autoencoder.to(self.device) if autoencoder else None
        if self.autoencoder:
            self.autoencoder.eval()

    def encode_text(self, texts: list[str], compress: bool = True) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Representación [CLS]

            if self.autoencoder and compress:
                if hasattr(self.autoencoder, "encode"):
                    encoded = self.autoencoder.encode(cls_embeddings)
                    if isinstance(encoded, tuple):  # VAE (mu, logvar)
                        return encoded[0].cpu()     # usamos mu
                    return encoded.cpu()
                else:
                    raise ValueError("El autoencoder debe implementar el método 'encode'")
            return cls_embeddings.cpu()

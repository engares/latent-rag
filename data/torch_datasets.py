# /data/torch_datasets.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


# ---------- UTILIDADES COMUNES ------------------------------------------------
def _load_pt(path: str) -> Dict[str, torch.Tensor]:
    """
    Carga un fichero .pt con tensores y asegura dtype = float32 en CPU.
    El fichero se espera como un dict { name: Tensor }.
    """
    data = torch.load(path, map_location="cpu")
    return {k: v.float() for k, v in data.items()}


# ---------- DATASETS ---------------------------------------------------------


class EmbeddingVAEDataset(Dataset):
    """
    Carga el fichero .pt generado por `ensure_uda_data`.
    Estructura esperada:
        {"input": <tensor [N×D]>, "target": <tensor [N×D]>}
    """
    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.input  = data["input"].float()
        self.target = data["target"].float()
        assert self.input.shape == self.target.shape, "input/target tamaño desigual"

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, idx):
        return {
            "input":  self.input[idx],
            "target": self.target[idx],
        }


class EmbeddingDAEDataset(Dataset):
    """
    Carga 'uda_dae_embeddings.pt' producido por `ensure_uda_data`.

    Estructura:
        {
            "input":  Tensor [N × D]  (embeddings con ruido)
            "target": Tensor [N × D]  (embeddings limpios)
        }
    """
    def __init__(self, path: str):
        d = torch.load(path, map_location="cpu")
        self.x  = d["input" ].float()
        self.y  = d["target"].float()
        assert self.x.shape == self.y.shape, "Input / target mismatch"

    def __len__(self):          return self.x.size(0)
    def __getitem__(self, idx): return {"x": self.x[idx], "y": self.y[idx]}

    

class EmbeddingTripletDataset(Dataset):
    """
    Carga 'uda_contrastive_embeddings.pt' generado por `ensure_uda_data`.

    Estructura esperada:
        {
            "query":     Tensor [N × D],
            "positive":  Tensor [N × D],
            "negative":  Tensor [N × D]
        }
    """
    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.q  = data["query"].float()
        self.p  = data["positive"].float()
        self.n  = data["negative"].float()
        assert self.q.shape == self.p.shape == self.n.shape, "Dimensiones incompatibles"

    def __len__(self) -> int:          return self.q.size(0)

    def __getitem__(self, idx):        # devuelvo tensores individuales
        return {"q": self.q[idx],
                "p": self.p[idx],
                "n": self.n[idx]}


# ---------- PRUEBA RÁPIDA -----------------------------------------------------
if __name__ == "__main__":
    dae_ds = EmbeddingDAEDataset("./data/squad_dae_embeddings.pt")
    vae_ds = EmbeddingDAEDataset("./data/squad_vae_embeddings.pt")
    con_ds = EmbeddingTripletDataset("./data/squad_contrastive_embeddings.pt")

    print("DAE sample ⇒", {k: v.shape for k, v in dae_ds[0].items()})
    print("Contrastive sample ⇒", {k: v.shape for k, v in con_ds[0].items()})
    print("VAE sample ⇒", {k: v.shape for k, v in vae_ds[0].items()})

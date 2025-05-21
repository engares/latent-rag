# training/train_cae.py – Contrastive Auto‑Encoder con hard‑negative mining

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.torch_datasets import EmbeddingTripletDataset
from models.contrastive_autoencoder import ContrastiveAutoencoder
from training.loss_functions import contrastive_loss
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data
from dotenv import load_dotenv

###############################################################################
#  ENTRENAMIENTO                                                               #
###############################################################################

def train_cae(
    dataset_path: str,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    hard_negatives: bool = True,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Contrastive AE on {device}  |  hard_negatives={hard_negatives}")

    # ---------------- Dataset & Dataloader ----------------
    ds = EmbeddingTripletDataset(dataset_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---------------- Model & Optimizer -------------------
    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # ---------------- Training Loop -----------------------
    model.train()
    for ep in range(epochs):
        running = 0.0
        for batch in dl:
            q = batch["q"].to(device)
            pos = batch["p"].to(device)

            optim.zero_grad()
            z_q = model.encode(q)
            z_pos = model.encode(pos)
            loss = contrastive_loss(z_q, z_pos, hard_negatives=hard_negatives)
            loss.backward()
            optim.step()

            running += loss.item() * q.size(0)

        epoch_loss = running / len(ds)
        print(f"[Epoch {ep+1:02d}/{epochs}]  Loss: {epoch_loss:.4f}")

    # ---------------- Save -------------------------------
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[OK] Modelo guardado → {model_save_path}")

###############################################################################
#  CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    load_dotenv()

    ap = argparse.ArgumentParser(description="Train Contrastive Auto‑Encoder (CAE)")
    ap.add_argument("--config", default="./config/config.yaml", help="Ruta YAML de configuración")
    ap.add_argument("--epochs", type=int, help="Número de épocas (override)")
    ap.add_argument("--lr", type=float, help="Learning rate (override)")
    ap.add_argument("--save_path", help="Ruta para guardar el checkpoint")
    ap.add_argument(
        "--batch_size",
        type=int,
        help="Tamaño de batch (override, por defecto el del YAML o 256)",
    )
    ap.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="Desactiva el hard‑negative mining in‑batch",
    )
    args = ap.parse_args()

    # ---------------- Config ------------------------------
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {}).get("contrastive", {})

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    # ---------------- Embeddings UDA ----------------------
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg.get("embedding_model", {})["name"],
    )

    # ---------------- Entrenamiento ----------------------
    train_cae(
        dataset_path=model_cfg.get("dataset_path", "./data/uda_contrastive_embeddings.pt"),
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr=args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path=args.save_path or model_cfg.get(
            "checkpoint", "./models/checkpoints/contrastive_ae.pth"
        ),
        hard_negatives=not args.no_hard_negatives,
        device=device,
    )

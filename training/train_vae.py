# training/train_vae.py – Variational Auto‑Encoder con validación y early‑stopping

from __future__ import annotations

import argparse
import os
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data.torch_datasets import EmbeddingVAEDataset
from models.variational_autoencoder import VariationalAutoencoder
from training.loss_functions import vae_loss
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data, split_dataset  # split util añadido
from dotenv import load_dotenv

###############################################################################
#  ENTRENAMIENTO                                                              #
###############################################################################

def train_vae(
    dataset_path: str,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    val_split: float = 0.1,
    patience: Optional[int] = 5,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training VAE on {device} | val_split={val_split}")

    # ---------------- Dataset ---------------------------
    full_ds = EmbeddingVAEDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------------- Model & Optimizer -----------------
    model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0

    # ---------------- Training Loop ---------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in dl_train:
            x_in = batch["input"].to(device)
            x_tar = batch["target"].to(device)

            optim.zero_grad()
            x_rec, mu, logvar = model(x_in)
            loss = vae_loss(x_rec, x_tar, mu, logvar, reduction="mean")
            loss.backward()
            optim.step()
            running += loss.item() * x_in.size(0)

        train_loss = running / len(train_ds)

        # ---------------- Validation --------------------
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in dl_val:
                x_in = batch["input"].to(device)
                x_tar = batch["target"].to(device)
                x_rec, mu, logvar = model(x_in)
                vloss = vae_loss(x_rec, x_tar, mu, logvar, reduction="mean")
                val_running += vloss.item() * x_in.size(0)
            val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ---------------- Early stopping ---------------
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Nuevo mejor val_loss. Checkpoint guardado en {model_save_path}")
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] Sin mejora en validación.")
                break

    print(f"[DONE] Mejor val_loss = {best_val:.6f}")

###############################################################################
#  CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Variational Auto‑Encoder (VAE)")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # ---------------- Config ---------------------------
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {}).get("vae", {})

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    # ---------------- Embeddings UDA -------------------
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg.get("embedding_model", {})["name"],
    )

    # ---------------- Entrenamiento -------------------
    train_vae(
        dataset_path=model_cfg.get("dataset_path", "./data/uda_vae_embeddings.pt"),
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr=args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path=args.save_path or model_cfg.get("checkpoint", "./models/checkpoints/vae_text.pth"),
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        device=device,
    )

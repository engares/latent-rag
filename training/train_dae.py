# training/train_dae.py – Denoising Auto‑Encoder con validación y early‑stopping

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from data.torch_datasets import EmbeddingDAEDataset
from models.denoising_autoencoder import DenoisingAutoencoder
from training.loss_functions import dae_loss
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data, split_dataset
from utils.load_config import init_logger               # NUEVO
from dotenv import load_dotenv

###############################################################################
#  ENTRENAMIENTO                                                              #
###############################################################################

def train_dae(
    dataset_path: str,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    logger,
    val_split: float = 0.1,
    patience: Optional[int] = 5,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training DAE on {device} | val_split={val_split}")
    logger.main.info("")
    logger.main.info("Training DAE | device=%s | deterministic=%s", device, train_cfg["deterministic"])

    # ---------------- Dataset --------------------------
    full_ds = EmbeddingDAEDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------------- Model & Optimizer ----------------
    model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0

    # ---------------- Training Loop -------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in dl_train:
            x_noisy = batch["x"].to(device)
            x_clean = batch["y"].to(device)

            optim.zero_grad()
            x_rec = model(x_noisy)
            loss = dae_loss(x_rec, x_clean, reduction="mean")
            loss.backward()
            optim.step()
            running += loss.item() * x_noisy.size(0)

        train_loss = running / len(train_ds)

        # ---------------- Validation ------------------
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in dl_val:
                x_noisy = batch["x"].to(device)
                x_clean = batch["y"].to(device)
                x_rec = model(x_noisy)
                vloss = dae_loss(x_rec, x_clean, reduction="mean")
                val_running += vloss.item() * x_noisy.size(0)
            val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        logger.train.info("[Epoch %02d/%d] train=%.6f | val=%.6f", epoch, epochs, train_loss, val_loss)

        # ---------------- Early Stopping --------------
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Checkpoint saved in {model_save_path}")
            logger.train.info("New best val_loss: %.6f → checkpoint %s", best_val, model_save_path)
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] No improvement in validation.")
                logger.train.info("[EARLY STOP] No improvement in validation.")
                break

    print(f"[DONE] Mejor val_loss = {best_val:.6f}")
    logger.main.info("[DONE] Best val_loss = %.6f", best_val)
    logger.main.info("")

###############################################################################
#  CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Denoising Auto‑Encoder (DAE)")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # ---------------- Config & logging ----------------
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {}).get("dae", {})
    log = init_logger(cfg["logging"])

    # ---------------- Reproducibilidad ----------------
    set_seed(train_cfg.get("seed", 42), train_cfg.get("deterministic", False), logger=log.train)
    device = resolve_device(train_cfg.get("device"))

    # ---------------- Embeddings -----------------------
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg.get("embedding_model", {})["name"],
    )

    # ---------------- Entrenamiento -------------------
    train_dae(
        dataset_path=model_cfg.get("dataset_path", "./data/uda_dae_embeddings.pt"),
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr = args.lr if args.lr is not None else float(train_cfg.get("learning_rate", 1e-3)),
        model_save_path=args.save_path or model_cfg.get("checkpoint", "./models/checkpoints/dae_text.pth"),
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        device=device,
        logger=log
    )

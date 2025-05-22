# training/train_cae.py – Contrastive Auto‑Encoder con hard‑negative mining y validación

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
from utils.data_utils import ensure_uda_data, split_dataset
from utils.load_config import init_logger                 # NUEVO
from dotenv import load_dotenv

###############################################################################
#  ENTRENAMIENTO                                                              #
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
    logger,
    hard_negatives: bool = True,
    val_split: float = 0.1,
    patience: Optional[int] = 5,
    margin: float = 0.2,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Contrastive AE on {device} | hard_negatives={hard_negatives} | val_split={val_split}")
    logger.main.info("")
    logger.main.info("Training CAE | device=%s | deterministic=%s | hard_negatives=%s",
                     device, train_cfg["deterministic"], hard_negatives)

    # ---------------- Dataset ---------------------------
    full_ds = EmbeddingTripletDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------------- Model & Optimizer -----------------
    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    no_improve = 0

    # ---------------- Training Loop ---------------------
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in dl_train:
            q = batch["q"].to(device)
            p = batch["p"].to(device)

            optim.zero_grad()
            z_q = model.encode(q)
            z_pos = model.encode(p)
            loss = contrastive_loss(z_q, z_pos, margin=margin, hard_negatives=hard_negatives)
            loss.backward()
            optim.step()
            running += loss.item() * q.size(0)

        train_loss = running / len(train_ds)

        # ---------------- Validation --------------------
        model.eval()
        with torch.no_grad():
            val_running = 0.0
            for batch in dl_val:
                q = batch["q"].to(device)
                p = batch["p"].to(device)
                z_q = model.encode(q)
                z_pos = model.encode(p)
                vloss = contrastive_loss(z_q, z_pos, margin=margin, hard_negatives=hard_negatives)
                val_running += vloss.item() * q.size(0)
            val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        logger.train.info("[Epoch %02d/%d] train=%.6f | val=%.6f", epoch, epochs, train_loss, val_loss)

        # ---------------- Early stopping ---------------
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Nueva mejor val_loss. Checkpoint guardado en {model_save_path}")
            logger.train.info("New best val_loss: %.6f → checkpoint %s", best_val, model_save_path)
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] Sin mejora en validación.")
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

    parser = argparse.ArgumentParser(description="Train Contrastive Auto‑Encoder (CAE)")
    parser.add_argument("--config", default="./config/config.yaml", help="Ruta YAML de configuración")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)    
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_split", type=float, default=0.1, help="Proporción para validación")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia early‑stopping; 0 = off")
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    # ---------------- Config & logging -----------------
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {}).get("contrastive", {})
    log = init_logger(cfg["logging"])

    # ---------------- Reproducibilidad -----------------
    set_seed(train_cfg.get("seed", 42), train_cfg.get("deterministic", False), logger=log.train)
    device = resolve_device(train_cfg.get("device"))

    # ---------------- Embeddings UDA -------------------
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg.get("embedding_model", {})["name"],
    )

    # ---------------- Entrenamiento -------------------
    train_cae(
        dataset_path=model_cfg.get("dataset_path", "./data/uda_contrastive_embeddings.pt"),
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr=args.lr if args.lr is not None else float(train_cfg.get("learning_rate", 1e-3)),
        model_save_path=args.save_path or model_cfg.get("checkpoint", "./models/checkpoints/contrastive_ae.pth"),
        hard_negatives=not args.no_hard_negatives,
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        margin=args.margin,
        device=device,
        logger=log
    )

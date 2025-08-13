# training/train_dae.py – Denoising Auto‑Encoder con validación y early‑stopping

from __future__ import annotations

import argparse
import os
from typing import Optional
import json
import csv

import torch
from torch.utils.data import DataLoader

from data.torch_datasets import EmbeddingDAEDataset
from models.denoising_autoencoder import DenoisingAutoencoder
from training.loss_functions import dae_loss
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import split_dataset, prepare_datasets
from utils.load_config import init_logger
from dotenv import load_dotenv

###############################################################################
#  TRAINING FUNCTION                                                          #
###############################################################################

def train_dae(
    *,
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
    """Run DAE training/validation loop saving only the final best checkpoint
    with train/val losses embedded in the filename plus a JSON sidecar."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training DAE on {device} | val_split={val_split}")
    logger.main.info("")
    logger.main.info("Training DAE | device=%s", device)

    # ---------------- Dataset --------------------------
    full_ds = EmbeddingDAEDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------------- Model & Optimizer ----------------
    model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Base prefix (strip .pth if present)
    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path

    best_val = float("inf")
    best_train = float("inf")
    best_state = None
    no_improve = 0
    history = []  # per-epoch metrics

    # ---------------- Training Loop -------------------
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0
        for batch in dl_train:
            x_noisy = batch["x"].to(device)
            x_clean = batch["y"].to(device)
            optim.zero_grad()
            x_rec = model(x_noisy)
            loss = dae_loss(x_rec, x_clean, reduction="mean")
            loss.backward(); optim.step()
            running += loss.item() * x_noisy.size(0)
        train_loss = running / len(train_ds)

        # ---------------- Validation ------------------
        model.eval(); val_running = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x_noisy = batch["x"].to(device)
                x_clean = batch["y"].to(device)
                x_rec = model(x_noisy)
                vloss = dae_loss(x_rec, x_clean, reduction="mean")
                val_running += vloss.item() * x_noisy.size(0)
        val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        logger.train.info("[Epoch %02d/%d] train=%.6f | val=%.6f", epoch, epochs, train_loss, val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optim.param_groups[0]['lr']),
        })

        # Track best (do not save yet)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_train = train_loss
            no_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"  -> Improved val_loss={best_val:.6f} (train={best_train:.6f}) [pending save]")
            logger.train.info("Improved val=%.6f (train=%.6f) pending final save", best_val, best_train)
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] No improvement in validation.")
                logger.train.info("[EARLY STOP] No improvement in validation.")
                break

    # ---------------- Final Save ----------------------
    history_dir = os.path.join('models', 'history')
    os.makedirs(history_dir, exist_ok=True)

    if best_state is not None:
        final_ckpt = f"{base_ckpt}_tr{best_train:.4f}_val{best_val:.4f}.pth"
        os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
        torch.save(best_state, final_ckpt)
        stem = os.path.splitext(os.path.basename(final_ckpt))[0]
        model_save_path = final_ckpt  # update reference
    else:
        stem = os.path.splitext(os.path.basename(base_ckpt))[0] + '_noimp'
        print("[WARN] No improvement registered; nothing saved.")
        logger.train.warning("No best state captured; nothing saved.")

    meta_path = os.path.join(history_dir, stem + '.json')
    csv_path  = os.path.join(history_dir, stem + '.csv')

    meta_payload = {
        'best_train_loss': None if best_train == float('inf') else best_train,
        'best_val_loss': None if best_val == float('inf') else best_val,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': lr,
        'val_split': val_split,
        'patience': patience,
        'epochs_ran': len(history),
    }
    with open(meta_path, 'w') as jf:
        json.dump(meta_payload, jf, indent=2)

    if history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        print(f"[HISTORY] Saved history CSV -> {csv_path}")
        logger.train.info("Saved history CSV %s", csv_path)
    print(f"[META] Saved meta JSON -> {meta_path}")
    logger.train.info("Saved meta JSON %s", meta_path)

    print(f"[DONE] Best val_loss = {best_val:.6f}")
    logger.main.info("[DONE] Best val_loss = %.6f", best_val)
    logger.main.info("")
    return model_save_path

###############################################################################
#  CLI                                                                        #
###############################################################################

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Denoising Auto‑Encoder (DAE)")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--dataset", choices=["uda", "squad"], help="Override YAML dataset")
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

    # ---------------- Reproducibility ----------------
    set_seed(train_cfg.get("seed", 42), train_cfg.get("deterministic", False), logger=log.train)
    device = resolve_device(train_cfg.get("device"))

    # ---------------- Dataset prep -------------------
    dataset_path = prepare_datasets(cfg, variant="dae", dataset_override=args.dataset)

    # ------------- model save path --------------
    ckpt_name = (
        f"dae"
        f"_in{model_cfg.get('input_dim', 384)}"
        f"_lat{model_cfg.get('latent_dim', 64)}"
        f"_hid{model_cfg.get('hidden_dim', 512)}"
        f"_bs{args.batch_size or train_cfg.get('batch_size', 256)}"
        f"_lr{args.lr or float(train_cfg.get('learning_rate', 1e-3))}"
        f"_ep{args.epochs or train_cfg.get('epochs', 20)}"
        f".pth"  # final losses appended after training
    )

    checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    model_save_path = args.save_path or os.path.join(checkpoints_dir, ckpt_name)

    # ---------------- Training -----------------------
    best_ckpt_path = train_dae(
        dataset_path=dataset_path,
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr=args.lr if args.lr is not None else float(train_cfg.get("learning_rate", 1e-3)),
        model_save_path=model_save_path,
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        device=device,
        logger=log,
    )
    if best_ckpt_path:
        print(f"[RESULT] Final best checkpoint: {best_ckpt_path}")

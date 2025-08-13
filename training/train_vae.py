# training/train_vae.py – Variational Auto‑Encoder con validación y early‑stopping

import argparse
import os
from typing import Optional
import json
import csv

import torch
from torch.utils.data import DataLoader

from data.torch_datasets import EmbeddingVAEDataset
from models.variational_autoencoder import VariationalAutoencoder
from training.loss_functions import vae_loss
from utils.load_config import load_config, init_logger
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import prepare_datasets
from dotenv import load_dotenv

###############################################################################
#  TRAINING LOOP                                                              #
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

    full_ds = EmbeddingVAEDataset(dataset_path)
    from utils.data_utils import split_dataset  # local import to avoid circular

    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Base (prefix) for checkpoint naming: remove .pth if user provided it
    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path

    best_val, best_train, no_improve = float("inf"), float("inf"), 0
    best_model_path = None
    best_state = None  # will hold best model parameters
    history = []  # per-epoch metrics
    for epoch in range(1, epochs + 1):
        # ---------------- train ------------------
        model.train(); running = 0.0
        for batch in dl_train:
            x_in  = batch["input"].to(device)
            x_tar = batch["target"].to(device)
            optim.zero_grad()
            x_rec, mu, logvar = model(x_in)
            loss = vae_loss(x_rec, x_tar, mu, logvar, mse_reduction="mean")
            loss.backward(); optim.step()
            running += loss.item() * x_in.size(0)
        train_loss = running / len(train_ds)

        # ---------------- validation -------------
        model.eval(); val_running = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x_in  = batch["input"].to(device)
                x_tar = batch["target"].to(device)
                x_rec, mu, logvar = model(x_in)
                vloss = vae_loss(x_rec, x_tar, mu, logvar, mse_reduction="mean")
                val_running += vloss.item() * x_in.size(0)
        val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train={train_loss:.6f} | val={val_loss:.6f}")
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optim.param_groups[0]['lr']),
        })

        # Track best (do NOT save yet) ---------------------------------------
        if val_loss < best_val - 1e-4:
            best_val, best_train, no_improve = val_loss, train_loss, 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"[BEST] Improved validation -> val={best_val:.6f} (train={best_train:.6f})")
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] No improvement in validation."); break

    # ---------------- final save ------------------------------------------------
    history_dir = os.path.join('models', 'history')
    os.makedirs(history_dir, exist_ok=True)
    if best_state is not None:
        final_ckpt_path = f"{base_ckpt}_tr{best_train:.4f}_val{best_val:.4f}.pth"
        os.makedirs(os.path.dirname(final_ckpt_path), exist_ok=True)
        torch.save(best_state, final_ckpt_path)
        stem = os.path.splitext(os.path.basename(final_ckpt_path))[0]
        best_model_path = final_ckpt_path
    else:
        stem = os.path.splitext(os.path.basename(base_ckpt))[0] + '_noimp'
        print("[WARN] No best state captured; nothing saved.")

    meta_path = os.path.join(history_dir, stem + '.json')
    csv_path  = os.path.join(history_dir, stem + '.csv')
    with open(meta_path, "w") as jf:
        json.dump({
            "best_train_loss": None if best_train == float('inf') else best_train,
            "best_val_loss": None if best_val == float('inf') else best_val,
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "batch_size": batch_size,
            "learning_rate": lr,
            "val_split": val_split,
            "patience": patience,
            "epochs_ran": len(history),
        }, jf, indent=2)
    if history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        print(f"[HISTORY] Saved history CSV -> {csv_path}")
    print(f"[META] Saved meta JSON -> {meta_path}")

    print(f"[DONE] best_val_loss = {best_val:.6f} | best_train_loss = {best_train:.6f}")

    return best_model_path

###############################################################################
#  CLI                                                                       #
###############################################################################

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Train Variational Auto‑Encoder (VAE)")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--dataset", choices=["uda", "squad"], help="Override dataset in config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    # ------------- config & logging -------------
    cfg       = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("models", {}).get("vae", {})
    log       = init_logger(cfg["logging"])

    set_seed(train_cfg.get("seed", 42), train_cfg.get("deterministic", False))
    device = resolve_device(train_cfg.get("device"))

    # ------------- dataset paths -----------------
    dataset_path = prepare_datasets(cfg, variant="vae", dataset_override=args.dataset)

    # ------------- model save path --------------
    ckpt_name = (
        f"vae"
        f"_in{model_cfg.get('input_dim', 384)}"
        f"_lat{model_cfg.get('latent_dim', 64)}"
        f"_hid{model_cfg.get('hidden_dim', 512)}"
        f"_bs{args.batch_size or train_cfg.get('batch_size', 256)}"
        f"_lr{args.lr or float(train_cfg.get('learning_rate', 1e-3))}"
        f"_ep{args.epochs or train_cfg.get('epochs', 20)}"
        f".pth"  # metrics will be appended inside training loop
    )

    checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    model_save_path = args.save_path or os.path.join(checkpoints_dir, ckpt_name)

    # ------------- training ----------------------
    best_ckpt_path = train_vae(
        dataset_path=dataset_path,
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=args.batch_size or train_cfg.get("batch_size", 256),
        epochs=args.epochs or train_cfg.get("epochs", 20),
        lr=float(args.lr) if args.lr is not None else float(train_cfg.get("learning_rate", 1e-3)),
        model_save_path=model_save_path,
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        device=device,
    )
    if best_ckpt_path:
        print(f"[RESULT] Final best checkpoint: {best_ckpt_path}")

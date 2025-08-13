# training/train_cae.py ― Contrastive Auto-Encoder with negative mining and validation

from __future__ import annotations
import argparse, os, math
from typing import Optional
import json
import csv

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from data.torch_datasets import EmbeddingTripletDataset
from models.contrastive_autoencoder import ContrastiveAutoencoder
from training.loss_functions import contrastive_loss        # in-batch mining
from utils.load_config import load_config, init_logger
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import prepare_datasets, split_dataset
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
#  AUX                                                                       #
# --------------------------------------------------------------------------- #

def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def _build_scheduler(optim: torch.optim.Optimizer, patience: int, factor: float = 0.5):
    # Reduce LR if val_loss does not improve for `patience` consecutive epochs
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=factor, patience=max(1, patience // 2)
    )

# --------------------------------------------------------------------------- #
#  TRAINING LOOP                                                             #
# --------------------------------------------------------------------------- #

def train_cae(
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
    hard_negatives: bool = True,
    margin: float = 0.2,
    val_split: float = 0.1,
    patience: Optional[int] = 5,
    min_delta: float = 0.003,                   # 0.3% relative improvement
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,                # 0 = disable
    device: Optional[str] = None,
) -> str:

    device = device or resolve_device()
    log = logger.train if hasattr(logger, "train") else logger

    log.info("CAE | device=%s | hard_negatives=%s | margin=%.3f", device, hard_negatives, margin)

    # ---------------- Dataset ---------------------------
    full_ds = EmbeddingTripletDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ---------------- Model & Opt -----------------------
    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = _build_optimizer(model, lr, weight_decay)
    scheduler = _build_scheduler(optim, patience or 4)

    best_val, epochs_no_improve = math.inf, 0
    best_train = math.inf
    best_state = None
    history = []  # track per-epoch metrics

    # Triplet loss native
    triplet_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(1, epochs + 1):
        # ---------------- Train -------------------------
        model.train(); running = 0.0
        for batch in dl_train:
            z_q  = model.encode(batch["q"].to(device))
            z_p  = model.encode(batch["p"].to(device))
            z_n  = model.encode(batch["n"].to(device))

            if hard_negatives:
                loss = contrastive_loss(z_q, z_p, margin=margin, hard_negatives=True)
            else:
                loss = triplet_fn(z_q, z_p, z_n)

            optim.zero_grad()
            loss.backward()
            if clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), clip_grad_norm)
            optim.step()
            running += loss.item() * z_q.size(0)

        train_loss = running / len(train_ds)

        # ---------------- Validation --------------------
        model.eval(); val_running = 0.0
        with torch.no_grad():
            for batch in dl_val:
                z_q  = model.encode(batch["q"].to(device))
                z_p  = model.encode(batch["p"].to(device))
                z_n  = model.encode(batch["n"].to(device))

                if hard_negatives:
                    vloss = contrastive_loss(z_q, z_p, margin=margin, hard_negatives=True)
                else:
                    vloss = triplet_fn(z_q, z_p, z_n)

                val_running += vloss.item() * z_q.size(0)
        val_loss = val_running / len(val_ds)

        log.info("[Epoch %02d/%d] train=%.6f | val=%.6f", epoch, epochs, train_loss, val_loss)
        scheduler.step(val_loss)

        # record epoch metrics
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optim.param_groups[0]['lr']),
        })

        # ---------------- Early stop --------------------
        rel_improve = (best_val - val_loss) / best_val if best_val < math.inf else 1.0
        if rel_improve > min_delta:
            best_val, epochs_no_improve = val_loss, 0
            best_train = train_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"  -> Improved val_loss={best_val:.6f} (train={best_train:.6f}) [pending save]")
            logger.train.info("Improved val=%.6f (train=%.6f) pending final save", best_val, best_train)
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print("[EARLY STOP] No improvement in validation.")
                logger.train.info("[EARLY STOP] No improvement in validation.")
                break

    # ------------- Final single save -------------------------------------
    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path
    history_dir = os.path.join('models', 'history')  # central history directory
    os.makedirs(history_dir, exist_ok=True)

    if best_state is not None:
        final_ckpt = f"{base_ckpt}_tr{best_train:.4f}_val{best_val:.4f}.pth"
        os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
        torch.save(best_state, final_ckpt)
        stem = os.path.splitext(os.path.basename(final_ckpt))[0]
        model_save_path = final_ckpt
    else:
        stem = os.path.splitext(os.path.basename(base_ckpt))[0] + '_noimp'
        print("[WARN] No improvement captured; nothing saved.")
        logger.train.warning("No improvement captured; nothing saved.")

    # ---- Save meta JSON & history CSV in /models/history -----------------
    meta_path = os.path.join(history_dir, stem + '.json')
    csv_path  = os.path.join(history_dir, stem + '.csv')

    meta_payload = {
        'best_train_loss': None if best_train == math.inf else best_train,
        'best_val_loss': None if best_val == math.inf else best_val,
        'margin': margin,
        'hard_negatives': hard_negatives,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'patience': patience,
        'val_split': val_split,
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

# --------------------------------------------------------------------------- #
#  CLI                                                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    load_dotenv()

    p = argparse.ArgumentParser(description="Train Contrastive Auto-Encoder (CAE)")
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--dataset", choices=["uda", "squad"], help="Override YAML dataset")
    p.add_argument("--epochs",  type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr",      type=float)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--margin",  type=float, default=0.2)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--no-hard-negatives", action="store_true")
    p.add_argument("--save_path")
    args = p.parse_args()

    # ---------- Config & logging -----------------------------------------
    cfg       = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg["models"]["contrastive"]
    log       = init_logger(cfg["logging"])

    set_seed(train_cfg.get("seed", 42), train_cfg.get("deterministic", False), logger=log.main)

    # ---------- Dataset ---------------------------------------------------
    ds_path = prepare_datasets(cfg, variant="cae", dataset_override=args.dataset)

    # ------------- model save path --------------
    ckpt_name = (
        f"cae"
        f"_in{model_cfg.get('input_dim', 384)}"
        f"_lat{model_cfg.get('latent_dim', 64)}"
        f"_hid{model_cfg.get('hidden_dim', 512)}"
        f"_margin{args.margin:.2f}"
        f"_hn{int(not args.no_hard_negatives)}"
        f"_bs{args.batch_size or train_cfg.get('batch_size', 256)}"
        f"_lr{args.lr or float(train_cfg.get('learning_rate', 1e-3))}"
        f"_ep{args.epochs or train_cfg.get('epochs', 20)}"
        f".pth"
    )

    checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    model_save_path = args.save_path or os.path.join(checkpoints_dir, ckpt_name)


    # ---------- Hparams final ---------------------------------------------
    hparams = dict(
        dataset_path= ds_path,
        input_dim = model_cfg.get("input_dim", 384),
        latent_dim = model_cfg.get("latent_dim", 64),
        hidden_dim = model_cfg.get("hidden_dim", 512),
        batch_size = args.batch_size or train_cfg.get("batch_size", 256),
        epochs = args.epochs or train_cfg.get("epochs", 20),
        lr = args.lr or float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay = args.weight_decay,
        clip_grad_norm = args.clip_grad,
        margin = args.margin,
        hard_negatives = not args.no_hard_negatives,
        val_split = args.val_split,
        patience = None if args.patience == 0 else args.patience,
        model_save_path = model_save_path,
        logger       = log,
    )

    train_cae_return = train_cae(**hparams)
    if train_cae_return:
        print(f"[RESULT] Final best checkpoint: {train_cae_return}")

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
    beta: float = 1.0,  # Target β (final)
    beta_start: float = 0.0,  # Initial β at epoch 1 for warmup
    beta_warmup_epochs: int = 0,  # Linear warmup epochs to reach β
    weight_decay: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    early_stop_delta: float = 1e-4,
    max_grad_norm: float | None = None,
    scheduler_factor: float = 0.5,
    scheduler_patience: Optional[int] = None,  # if None -> (patience//2)
    num_workers: int = 0,
    report_cb: Optional[callable] = None,
    trial_suffix: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith('cuda') and torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] Using GPU: {dev_name} (total {torch.cuda.device_count()} GPU(s))")
        except Exception:
            print("[DEVICE] Using GPU")
    else:
        print("[DEVICE] Using CPU")
    print(f"[INFO] Training VAE on {device} | val_split={val_split} | β={beta} (warmup {beta_warmup_epochs} ep)")

    full_ds = EmbeddingVAEDataset(dataset_path)
    from utils.data_utils import split_dataset  # local import to avoid circular

    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                           pin_memory=(device.startswith('cuda')), num_workers=num_workers)
    dl_val   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False,
                          pin_memory=(device.startswith('cuda')), num_workers=num_workers)

    model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                             betas=(adam_beta1, adam_beta2))

    # Scheduler for consistency with other trainers
    sched_pat = scheduler_patience if scheduler_patience is not None else max(1, (patience or 4)//2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=scheduler_factor, patience=sched_pat
    )

    # Base (prefix) for checkpoint naming: remove .pth if user provided it
    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path
    if trial_suffix:
        base_ckpt = base_ckpt + f"_{trial_suffix}"

    best_val, best_train, no_improve = float("inf"), float("inf"), 0
    best_model_path = None
    best_state = None  # will hold best model parameters
    history = []  # per-epoch metrics
    for epoch in range(1, epochs + 1):
        # β schedule (linear warmup)
        if beta_warmup_epochs > 0 and epoch <= beta_warmup_epochs:
            # progress in [0,1]
            prog = (epoch - 1) / max(1, beta_warmup_epochs - 1)
            current_beta = beta_start + (beta - beta_start) * prog
        else:
            current_beta = beta

        # ---------------- train ------------------
        model.train(); running = 0.0
        for batch in dl_train:
            x_in  = batch["input"].to(device)
            x_tar = batch["target"].to(device)
            optim.zero_grad()
            x_rec, mu, logvar = model(x_in)
            loss = vae_loss(x_rec, x_tar, mu, logvar, beta=current_beta)
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()
            running += loss.item() * x_in.size(0)
        train_loss = running / len(train_ds)

        # ---------------- validation -------------
        model.eval(); val_running = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x_in  = batch["input"].to(device)
                x_tar = batch["target"].to(device)
                x_rec, mu, logvar = model(x_in)
                vloss = vae_loss(x_rec, x_tar, mu, logvar, beta=current_beta)
                val_running += vloss.item() * x_in.size(0)
        val_loss = val_running / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] β={current_beta:.4f} train={train_loss:.6f} | val={val_loss:.6f}")
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'beta': float(current_beta),
            'lr': float(optim.param_groups[0]['lr']),
        })

        # callback early so pruning can stop immediately
        if report_cb is not None:
            try:
                report_cb(epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss), lr=float(optim.param_groups[0]['lr']))
            except Exception as cb_err:
                print(f"[WARN] report_cb failed: {cb_err}")

        # Track best (do NOT save yet) ---------------------------------------
        if val_loss < best_val - early_stop_delta:
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
        final_ckpt_path = f"{base_ckpt}_beta{beta}_tr{best_train:.4f}_val{best_val:.4f}.pth"
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
            "weight_decay": weight_decay,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "val_split": val_split,
            "patience": patience,
            "early_stop_delta": early_stop_delta,
            "epochs_ran": len(history),
            "beta_target": beta,
            "beta_start": beta_start,
            "beta_warmup_epochs": beta_warmup_epochs,
            "max_grad_norm": max_grad_norm,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": sched_pat,
            "num_workers": num_workers,
            "trial_suffix": trial_suffix,
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
    parser.add_argument("--beta", type=float, default=1.0, help="Final weight of KL term (β)")
    parser.add_argument("--beta_start", type=float, default=0.0, help="Initial β for warmup")
    parser.add_argument("--beta_warmup_epochs", type=int, default=0, help="Linear warmup epochs to reach final β")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Clip gradient norm (0=disabled)")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int)
    parser.add_argument("--num_workers", type=int, default=0)
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
    seed_val = train_cfg.get("seed", 42)
    lr_effective = float(args.lr) if args.lr is not None else float(train_cfg.get('learning_rate', 1e-3))
    bs_effective = args.batch_size or train_cfg.get('batch_size', 256)
    ep_effective = args.epochs or train_cfg.get('epochs', 20)

    parts = [
        "vae",
        f"in{model_cfg.get('input_dim', 384)}",
        f"lat{model_cfg.get('latent_dim', 64)}",
        f"hid{model_cfg.get('hidden_dim', 512)}",
        f"bs{bs_effective}",
        f"lr{lr_effective:g}",
        f"beta{args.beta:g}",
    ]
    if args.beta_warmup_epochs > 0:
        parts.append(f"bw{args.beta_warmup_epochs}")
    if args.beta_start > 0:
        parts.append(f"bstart{args.beta_start:g}")
    if args.weight_decay > 0:
        parts.append(f"wd{args.weight_decay:g}")
    if args.max_grad_norm and args.max_grad_norm > 0:
        parts.append(f"gn{args.max_grad_norm:g}")
    if args.scheduler_patience:
        parts.append(f"sp{args.scheduler_patience}")
    if args.scheduler_factor != 0.5:
        parts.append(f"sf{args.scheduler_factor:g}")
    parts.append(f"ep{ep_effective}")
    ckpt_name = "_".join(parts) + ".pth"

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
        beta=args.beta,
        beta_start=args.beta_start,
        beta_warmup_epochs=args.beta_warmup_epochs,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        early_stop_delta=args.early_stop_delta,
        max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        num_workers=args.num_workers,
        report_cb=None,
        trial_suffix=None,
    )
    if best_ckpt_path:
        print(f"[RESULT] Final best checkpoint: {best_ckpt_path}")

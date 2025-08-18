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

def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, beta1: float, beta2: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

def _build_scheduler(optim: torch.optim.Optimizer, patience: int, factor: float):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=factor, patience=max(1, patience)
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
    early_stop_delta: float = 0.003,          # relative improvement threshold
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,               # 0 = disable
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    scheduler_factor: float = 0.5,
    scheduler_patience: Optional[int] = None,  # if None -> patience//2 default
    num_workers: int = 0,
    log_interval: int = 0,                    # 0 = only epoch logs
    seed: int = 42,
    device: Optional[str] = None,
    report_cb: Optional[callable] = None,
    trial_suffix: str | None = None,
) -> str:

    set_seed(seed, False, logger=logger.main if hasattr(logger, "main") else None)
    device = device or resolve_device()
    if device.startswith('cuda') and torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] Using GPU: {dev_name} (total {torch.cuda.device_count()} GPU(s))")
        except Exception:
            print("[DEVICE] Using GPU")
    else:
        print("[DEVICE] Using CPU")
    log = logger.train if hasattr(logger, "train") else logger

    log.info(
        "CAE | device=%s | hard_negatives=%s | margin=%.3f | wd=%.1e | b1=%.2f b2=%.3f",
        device, hard_negatives, margin, weight_decay, adam_beta1, adam_beta2,
    )

    # ---------------- Dataset ---------------------------
    full_ds = EmbeddingTripletDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)
    dl_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=device.startswith("cuda")
    )
    dl_val   = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.startswith("cuda")
    )

    # ---------------- Model & Opt -----------------------
    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = _build_optimizer(model, lr, weight_decay, adam_beta1, adam_beta2)
    sched_pat = scheduler_patience if scheduler_patience is not None else ((patience or 4)//2)
    scheduler = _build_scheduler(optim, sched_pat, scheduler_factor)

    best_val, epochs_no_improve = math.inf, 0
    best_train = math.inf
    best_state = None
    history = []

    triplet_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    for epoch in range(1, epochs + 1):
        # ---------------- Train -------------------------
        model.train(); running = 0.0
        for b_idx, batch in enumerate(dl_train, 1):
            z_q  = model.encode(batch["q"].to(device))
            z_p  = model.encode(batch["p"].to(device))
            z_n  = model.encode(batch["n"].to(device))

            if hard_negatives:
                loss = contrastive_loss(z_q, z_p, margin=margin, hard_negatives=True)
            else:
                loss = triplet_fn(z_q, z_p, z_n)

            optim.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()
            running += loss.item() * z_q.size(0)

            if log_interval and (b_idx % log_interval == 0):
                log.info("  [Epoch %02d] batch %d/%d loss=%.6f", epoch, b_idx, len(dl_train), loss.item())

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

        log.info(
            "[Epoch %02d/%d] train=%.6f | val=%.6f | lr=%.3g",
            epoch, epochs, train_loss, val_loss, optim.param_groups[0]['lr']
        )
        scheduler.step(val_loss)

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optim.param_groups[0]['lr']),
        })

        # callback report BEFORE early-stop break so pruning sees this epoch
        if report_cb is not None:
            try:
                report_cb(epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss), lr=float(optim.param_groups[0]['lr']))
            except Exception as _cb_err:
                log.warning("report_cb failed: %s", _cb_err)

        # ---------------- Early stop --------------------
        rel_improve = (best_val - val_loss) / best_val if best_val < math.inf else 1.0
        if rel_improve > early_stop_delta:
            best_val, epochs_no_improve = val_loss, 0
            best_train = train_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            log.info("  -> Improved val=%.6f (train=%.6f)", best_val, best_train)
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                log.info("[EARLY STOP] No improvement in validation.")
                break

    # ------------- Final single save -------------------------------------
    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path
    if trial_suffix:
        base_ckpt = base_ckpt + f"_{trial_suffix}"
    history_dir = os.path.join('models', 'history')
    os.makedirs(history_dir, exist_ok=True)

    if best_state is not None:
        final_ckpt = f"{base_ckpt}_tr{best_train:.4f}_val{best_val:.4f}.pth"
        os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
        torch.save(best_state, final_ckpt)
        stem = os.path.splitext(os.path.basename(final_ckpt))[0]
        model_save_path = final_ckpt
    else:
        stem = os.path.splitext(os.path.basename(base_ckpt))[0] + '_noimp'
        log.warning("No improvement captured; saving last state snapshot for reference.")
        # Save last model state anyway for reproducibility
        torch.save(model.state_dict(), base_ckpt + '_last.pth')

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
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'patience': patience,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': sched_pat,
        'val_split': val_split,
        'early_stop_delta': early_stop_delta,
        'max_grad_norm': max_grad_norm,
        'num_workers': num_workers,
        'log_interval': log_interval,
        'epochs_ran': len(history),
        'trial_suffix': trial_suffix,
    }
    with open(meta_path, 'w') as jf:
        json.dump(meta_payload, jf, indent=2)

    if history:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        log.info("Saved history CSV %s", csv_path)
    log.info("Saved meta JSON %s", meta_path)
    log.info("[DONE] Best val_loss = %.6f", best_val)
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
    p.add_argument("--clip_grad", type=float, default=1.0, help="Max grad norm (0=disabled)")
    p.add_argument("--margin",  type=float, default=0.2)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--early_stop_delta", type=float, default=0.003)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-hard-negatives", action="store_true")
    p.add_argument("--save_path")
    args = p.parse_args()

    cfg       = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg["models"]["cae"]
    log       = init_logger(cfg["logging"])

    set_seed(args.seed, train_cfg.get("deterministic", False), logger=log.main)

    ds_path = prepare_datasets(cfg, variant="cae", dataset_override=args.dataset)

    # checkpoint naming enriched
    seed_val = args.seed
    lr_eff = args.lr or float(train_cfg.get('learning_rate', 1e-3))
    bs_eff = args.batch_size or train_cfg.get('batch_size', 256)
    ep_eff = args.epochs or train_cfg.get('epochs', 20)
    parts = [
        "cae",
        f"in{model_cfg.get('input_dim', 384)}",
        f"lat{model_cfg.get('latent_dim', 64)}",
        f"hid{model_cfg.get('hidden_dim', 512)}",
        f"m{args.margin:.2f}",
        f"hn{int(not args.no_hard_negatives)}",
        f"bs{bs_eff}",
        f"lr{lr_eff:g}",
    ]
    if args.weight_decay > 0: parts.append(f"wd{args.weight_decay:g}")
    if args.clip_grad > 0: parts.append(f"gn{args.clip_grad:g}")
    parts += [f"b1{args.adam_beta1:.2f}", f"b2{args.adam_beta2:.3f}"]
    if args.scheduler_factor != 0.5: parts.append(f"sf{args.scheduler_factor:g}")
    if args.scheduler_patience: parts.append(f"sp{args.scheduler_patience}")
    parts.append(f"ep{ep_eff}")
    ckpt_name = "_".join(parts) + ".pth"

    checkpoints_dir = cfg["paths"]["checkpoints_dir"]
    model_save_path = args.save_path or os.path.join(checkpoints_dir, ckpt_name)

    hparams = dict(
        dataset_path= ds_path,
        input_dim = model_cfg.get("input_dim", 384),
        latent_dim = model_cfg.get("latent_dim", 64),
        hidden_dim = model_cfg.get("hidden_dim", 512),
        batch_size = bs_eff,
        epochs = ep_eff,
        lr = lr_eff,
        weight_decay = args.weight_decay,
        max_grad_norm = args.clip_grad,
        adam_beta1 = args.adam_beta1,
        adam_beta2 = args.adam_beta2,
        scheduler_factor = args.scheduler_factor,
        scheduler_patience = args.scheduler_patience,
        early_stop_delta = args.early_stop_delta,
        num_workers = args.num_workers,
        log_interval = args.log_interval,
        seed = args.seed,
        margin = args.margin,
        hard_negatives = not args.no_hard_negatives,
        val_split = args.val_split,
        patience = None if args.patience == 0 else args.patience,
        model_save_path = model_save_path,
        logger = log,
    )

    train_cae_return = train_cae(**hparams)
    if train_cae_return:
        print(f"[RESULT] Final best checkpoint: {train_cae_return}")

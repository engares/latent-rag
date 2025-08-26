# training/train_base.py – Plain Auto‑Encoder (AE) with validation and early‑stopping

from __future__ import annotations

import argparse
import os
from typing import Optional
import json
import csv

import torch
from torch.utils.data import DataLoader

from data.torch_datasets import EmbeddingVAEDataset
from models.simple_autoencoder import SimpleAutoencoder
from utils.load_config import load_config, init_logger
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import prepare_datasets, split_dataset
from dotenv import load_dotenv


def train_base(
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
    weight_decay: float = 0.0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    early_stop_delta: float = 1e-4,
    max_grad_norm: float = 0.0,
    scheduler_factor: float = 0.5,
    scheduler_patience: Optional[int] = None,
    num_workers: int = 0,
    report_cb: Optional[callable] = None,
    trial_suffix: str | None = None,
):
    """Train a plain Auto‑Encoder (reconstruction with MSE).

    Mirrors train_dae/train_vae conventions so it plugs into hparam_search.
    Saves a single final best checkpoint and writes JSON/CSV history under
    models/history with key 'best_val_loss'.
    """
    device = device or resolve_device()
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
            print(f"[DEVICE] Using GPU: {dev_name} (total {torch.cuda.device_count()} GPU(s))")
        except Exception:
            print("[DEVICE] Using GPU")
    else:
        print("[DEVICE] Using CPU")

    full_ds = EmbeddingVAEDataset(dataset_path)
    train_ds, val_ds = split_dataset(full_ds, val_split=val_split)

    dl_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=device.startswith('cuda')
    )
    dl_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=device.startswith('cuda')
    )

    model = SimpleAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2)
    )

    sched_pat = scheduler_patience if scheduler_patience is not None else max(1, (patience or 4)//2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=scheduler_factor, patience=sched_pat
    )

    base_ckpt = model_save_path[:-4] if model_save_path.endswith('.pth') else model_save_path
    if trial_suffix:
        base_ckpt = base_ckpt + f"_{trial_suffix}"

    best_val = float('inf')
    best_train = float('inf')
    best_state = None
    no_improve = 0
    history = []

    mse = torch.nn.MSELoss(reduction='mean')

    for epoch in range(1, epochs + 1):
        # ---------------- Train -----------------
        model.train(); run = 0.0
        for batch in dl_train:
            x_in = batch['input'].to(device)
            x_tar = batch['target'].to(device)
            optim.zero_grad(set_to_none=True)
            x_rec = model(x_in)
            loss = mse(x_rec, x_tar)
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optim.step()
            run += loss.item() * x_in.size(0)
        train_loss = run / len(train_ds)

        # ---------------- Val -------------------
        model.eval(); vrun = 0.0
        with torch.no_grad():
            for batch in dl_val:
                x_in = batch['input'].to(device)
                x_tar = batch['target'].to(device)
                x_rec = model(x_in)
                vloss = mse(x_rec, x_tar)
                vrun += vloss.item() * x_in.size(0)
        val_loss = vrun / len(val_ds)

        print(f"[Epoch {epoch:02d}/{epochs}] train={train_loss:.6f} | val={val_loss:.6f}")
        if hasattr(logger, 'train'):
            logger.train.info(
                "[Epoch %02d/%d] train=%.6f | val=%.6f | lr=%.3g",
                epoch, epochs, train_loss, val_loss, optim.param_groups[0]['lr']
            )

        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'lr': float(optim.param_groups[0]['lr']),
        })

        if report_cb is not None:
            try:
                report_cb(epoch=epoch, train_loss=float(train_loss), val_loss=float(val_loss), lr=float(optim.param_groups[0]['lr']))
            except Exception:
                pass

        scheduler.step(val_loss)

        if val_loss < best_val - early_stop_delta:
            best_val = val_loss
            best_train = train_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                print("[EARLY STOP] No improvement in validation.")
                break

    # ---------------- Save ---------------------
    history_dir = os.path.join('models', 'history'); os.makedirs(history_dir, exist_ok=True)
    if best_state is not None:
        final_ckpt = f"{base_ckpt}_tr{best_train:.4f}_val{best_val:.4f}.pth"
        os.makedirs(os.path.dirname(final_ckpt), exist_ok=True)
        torch.save(best_state, final_ckpt)
        stem = os.path.splitext(os.path.basename(final_ckpt))[0]
        model_save_path = final_ckpt
    else:
        stem = os.path.splitext(os.path.basename(base_ckpt))[0] + '_noimp'
        torch.save(model.state_dict(), base_ckpt + '_last.pth')

    meta = {
        'best_train_loss': None if best_train == float('inf') else best_train,
        'best_val_loss': None if best_val == float('inf') else best_val,
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'batch_size': batch_size,
        'learning_rate': lr,
        'weight_decay': weight_decay,
        'adam_beta1': adam_beta1,
        'adam_beta2': adam_beta2,
        'val_split': val_split,
        'patience': patience,
        'early_stop_delta': early_stop_delta,
        'max_grad_norm': max_grad_norm,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': sched_pat,
        'num_workers': num_workers,
        'epochs_ran': len(history),
        'trial_suffix': trial_suffix,
    }
    meta_path = os.path.join(history_dir, stem + '.json')
    with open(meta_path, 'w') as jf:
        json.dump(meta, jf, indent=2)

    if history:
        csv_path = os.path.join(history_dir, stem + '.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader(); writer.writerows(history)
        if hasattr(logger, 'train'):
            logger.train.info("Saved history CSV %s", csv_path)

    if hasattr(logger, 'main'):
        logger.main.info("[DONE] Best val_loss = %.6f", best_val)
    return model_save_path


if __name__ == "__main__":
    load_dotenv()

    p = argparse.ArgumentParser(description="Train plain Auto‑Encoder (AE)")
    p.add_argument("--config", default="./config/config.yaml")
    p.add_argument("--dataset", choices=["uda", "squad"], help="Override YAML dataset")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--early_stop_delta", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--scheduler_factor", type=float, default=0.5)
    p.add_argument("--scheduler_patience", type=int)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_path")
    args = p.parse_args()

    cfg = load_config(args.config)
    log = init_logger(cfg['logging'])

    train_cfg = cfg.get('training', {})
    model_cfg = cfg.get('models', {}).get('base', cfg.get('models', {}).get('vae', {}))

    set_seed(train_cfg.get('seed', 42), train_cfg.get('deterministic', False))
    device = resolve_device(train_cfg.get('device'))

    ds_path = prepare_datasets(cfg, variant="base", dataset_override=args.dataset)

    # checkpoint name
    bs_eff = args.batch_size or train_cfg.get('batch_size', 256)
    lr_eff = float(args.lr) if args.lr is not None else float(train_cfg.get('learning_rate', 1e-3))
    ep_eff = args.epochs or train_cfg.get('epochs', 20)
    parts = [
        "base",
        f"in{model_cfg.get('input_dim', 384)}",
        f"lat{model_cfg.get('latent_dim', 64)}",
        f"hid{model_cfg.get('hidden_dim', 512)}",
        f"bs{bs_eff}", f"lr{lr_eff:g}", f"ep{ep_eff}",
    ]
    if args.weight_decay > 0: parts.append(f"wd{args.weight_decay:g}")
    if args.max_grad_norm and args.max_grad_norm > 0: parts.append(f"gn{args.max_grad_norm:g}")
    if args.scheduler_patience: parts.append(f"sp{args.scheduler_patience}")
    if args.scheduler_factor != 0.5: parts.append(f"sf{args.scheduler_factor:g}")
    ckpt_name = "_".join(parts) + ".pth"

    checkpoints_dir = cfg['paths']['checkpoints_dir']
    model_save_path = args.save_path or os.path.join(checkpoints_dir, ckpt_name)

    best_ckpt = train_base(
        dataset_path=ds_path,
        input_dim=model_cfg.get('input_dim', 384),
        latent_dim=model_cfg.get('latent_dim', 64),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        batch_size=bs_eff,
        epochs=ep_eff,
        lr=lr_eff,
        model_save_path=model_save_path,
        logger=log,
        val_split=args.val_split,
        patience=None if args.patience == 0 else args.patience,
        device=device,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        early_stop_delta=args.early_stop_delta,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else 0.0,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        num_workers=args.num_workers,
        report_cb=None,
        trial_suffix=None,
    )
    if best_ckpt:
        print(f"[RESULT] Final best checkpoint: {best_ckpt}")

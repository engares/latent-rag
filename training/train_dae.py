import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.denoising_autoencoder import DenoisingAutoencoder
from training.loss_functions import dae_loss  # MSE reconstruction loss
from data.torch_datasets import DAEDataset
from dotenv import load_dotenv
from utils.load_config import load_config
from utils.torch_utils import set_seed, resolve_device

def train_dae(
    dataset_path: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    tokenizer_name: str,
    max_length: int = 256,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training DAE on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = DAEDataset(dataset_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # NOTE: order is (input_dim, latent_dim, hidden_dim)
    model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            noisy = batch["input_ids"].to(device).float()
            clean = batch["target_ids"].to(device).float()

            optimizer.zero_grad()
            recon = model(noisy)
            loss = dae_loss(recon, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Model saved to: {model_save_path}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    ae_cfg = cfg.get("autoencoder", {})

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    train_dae(
        dataset_path=train_cfg.get("dae_dataset_path", "./data/uda_dae_train.jsonl"),
        input_dim=ae_cfg["input_dim"],
        hidden_dim=ae_cfg.get("hidden_dim", 512),
        latent_dim=ae_cfg.get("latent_dim", 64),
        batch_size=train_cfg.get("batch_size", 32),
        epochs=args.epochs or train_cfg.get("epochs", 10),
        lr=args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path=args.save_path or ae_cfg.get("checkpoint", "./models/checkpoints/dae_text.pth"),
        tokenizer_name=train_cfg.get("tokenizer", "sentence-transformers/all-MiniLM-L6-v2"),
        max_length=train_cfg.get("max_length", 256),
    )

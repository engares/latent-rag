import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models.contrastive_autoencoder import ContrastiveAutoencoder
from training.loss_functions import contrastive_loss
from data.torch_datasets import ContrastiveDataset
from dotenv import load_dotenv
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device

def train_contrastive(
    dataset_path: str,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    tokenizer_name: str,
    max_length: int = 256,
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset = ContrastiveDataset(dataset_path, tokenizer, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            q = batch["query_ids"].to(device).float()
            pos = batch["pos_ids"].to(device).float()
            neg = batch["neg_ids"].to(device).float()

            optimizer.zero_grad()
            z_q = model.encode(q)
            z_pos = model.encode(pos)
            z_neg = model.encode(neg)
            loss = contrastive_loss(z_q, z_pos, z_neg)
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
    parser.add_argument("--model", type=str, required=True, help="Model key in config.yaml: vae, dae, contrastive")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    model_key = args.model.lower()
    model_cfg = config.get("models", {}).get(model_key, {})

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    train_contrastive(
        dataset_path=model_cfg.get("dataset_path", f"./data/{model_key}_train.jsonl"),
        input_dim=model_cfg.get("input_dim", 384),
        latent_dim=model_cfg.get("latent_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        batch_size=train_cfg.get("batch_size", 32),
        epochs=args.epochs or train_cfg.get("epochs", 10),
        lr=args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path=args.save_path or model_cfg.get("checkpoint", f"./models/checkpoints/{model_key}.pth"),
        tokenizer_name=train_cfg.get("tokenizer", "sentence-transformers/all-MiniLM-L6-v2"),
        max_length=train_cfg.get("max_length", 256),
        device=device
    )

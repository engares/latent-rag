# training/train_cae.py
import argparse, os, torch
from torch.utils.data import DataLoader
from models.contrastive_autoencoder import ContrastiveAutoencoder
from training.loss_functions import contrastive_loss
from data.torch_datasets import EmbeddingTripletDataset
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data
from dotenv import load_dotenv

# ----------------------------------------------------------------------
def train_cae(
    dataset_path: str,
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    model_save_path: str,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training Contrastive AE on {device}")

    dataset    = EmbeddingTripletDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ContrastiveAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dataloader:
            q   = batch["q"].to(device)
            pos = batch["p"].to(device)
            neg = batch["n"].to(device)

            optim.zero_grad()
            z_q   = model.encode(q)
            z_pos = model.encode(pos)
            z_neg = model.encode(neg)
            loss  = contrastive_loss(z_q, z_pos, z_neg)
            loss.backward()
            optim.step()

            tot += loss.item()

        print(f"[Epoch {ep+1}/{epochs}] Loss: {tot/len(dataset):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[OK] Modelo guardado → {model_save_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--model",  required=True, help="contrastive | vae | dae")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr",     type=float)
    ap.add_argument("--save_path")
    args = ap.parse_args()

    cfg        = load_config(args.config)
    train_cfg  = cfg["training"]
    model_cfg  = cfg["models"][args.model.lower()]

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    # genera embeddings si hace falta
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
    )

    train_cae(
        dataset_path   = model_cfg.get("dataset_path", "./data/uda_contrastive_embeddings.pt"),
        input_dim      = model_cfg.get("input_dim", 384),
        latent_dim     = model_cfg.get("latent_dim", 64),
        hidden_dim     = model_cfg.get("hidden_dim", 512),
        batch_size     = train_cfg.get("batch_size", 32),
        epochs         = args.epochs or train_cfg.get("epochs", 10),
        lr             = args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path= args.save_path or model_cfg.get(
                            "checkpoint", "./models/checkpoints/contrastive_ae.pth"),
        device         = device,
    )

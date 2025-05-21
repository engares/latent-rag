# training/train_dae.py
import argparse, os, torch
from torch.utils.data import DataLoader
from models.denoising_autoencoder import DenoisingAutoencoder
from training.loss_functions import dae_loss
from data.torch_datasets import EmbeddingDAEDataset
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data
from dotenv import load_dotenv

# ----------------------------------------------------------------------
def train_dae(
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
    print(f"[INFO] Training Denoising AE on {device}")

    ds  = EmbeddingDAEDataset(dataset_path)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DenoisingAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in dl:
            x_noisy  = batch["x"].to(device)
            x_clean  = batch["y"].to(device)

            optim.zero_grad()
            x_recon  = model(x_noisy)
            loss     = dae_loss(x_recon, x_clean)
            loss.backward()
            optim.step()
            tot += loss.item()

        print(f"[Epoch {ep+1}/{epochs}] Loss: {tot/len(ds):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[OK] Modelo guardado → {model_save_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/config.yaml")
    ap.add_argument("--model",  required=True, help="dae | vae | contrastive")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr",     type=float)
    ap.add_argument("--save_path")
    args = ap.parse_args()

    cfg        = load_config(args.config)
    train_cfg  = cfg["training"]
    model_cfg  = cfg["models"][args.model.lower()]

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    # Asegúrate de tener los ficheros .pt
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
    )

    train_dae(
        dataset_path = model_cfg.get("dataset_path", "./data/uda_dae_embeddings.pt"),
        input_dim = model_cfg.get("input_dim", 384),
        latent_dim = model_cfg.get("latent_dim", 64),
        hidden_dim = model_cfg.get("hidden_dim", 512),
        batch_size = train_cfg.get("batch_size", 32),
        epochs = args.epochs or train_cfg.get("epochs", 10),
        lr  = args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path= args.save_path or model_cfg.get(
                            "checkpoint", "./models/checkpoints/dae_text.pth"),
        device  = device
    )

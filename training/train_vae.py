# /training/train_vae.py
# training/train_vae.py
import argparse, os, torch, yaml
from torch.utils.data import DataLoader
from models.variational_autoencoder import VariationalAutoencoder
from data.torch_datasets import EmbeddingVAEDataset
from training.loss_functions import vae_loss
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device
from utils.data_utils import ensure_uda_data
from dotenv import load_dotenv

# ------------------- Entrenamiento ----------------------------------------
def train_vae(
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
    print(f"[INFO] Training VAE on: {device}")

    dataset   = EmbeddingVAEDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VariationalAutoencoder(input_dim, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in dataloader:
            x_in  = batch["input"].to(device)
            x_tar = batch["target"].to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(x_in)
            loss = vae_loss(x_recon, x_tar, mu, logvar)
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total/len(dataset):.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[OK] Modelo guardado → {model_save_path}")

# ------------------- CLI ---------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--model",  required=True, help="vae | dae | contrastive")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr",     type=float)
    parser.add_argument("--save_path")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    train_cfg  = cfg.get("training", {})
    model_cfg  = cfg.get("models", {}).get(args.model.lower(), {})

    set_seed(train_cfg.get("seed", 42))
    device = resolve_device(train_cfg.get("device"))

    # Garantizar embeddings UDA
    ensure_uda_data(
        output_dir="./data",
        max_samples=train_cfg.get("max_samples"),
        base_model_name=cfg["embedding_model"]["name"],
    )

    train_vae(
        dataset_path = model_cfg.get("dataset_path", "./data/uda_vae_embeddings.pt"),
        input_dim    = model_cfg.get("input_dim", 384),
        latent_dim   = model_cfg.get("latent_dim", 64),
        hidden_dim   = model_cfg.get("hidden_dim", 512),
        batch_size   = train_cfg.get("batch_size", 32),
        epochs       = args.epochs or train_cfg.get("epochs", 10),
        lr           = args.lr or train_cfg.get("learning_rate", 1e-3),
        model_save_path = args.save_path or model_cfg.get("checkpoint", "./models/checkpoints/vae.pth"),
        device       = device,
    )

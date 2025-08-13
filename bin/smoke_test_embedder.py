#!/usr/bin/env python
import argparse, json, os, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

from retrieval.embedder import EmbeddingCompressor

def _read_meta_from_ckpt(ckpt_path: str):
    """Lee hiperparámetros auxiliares desde <ckpt>.meta.json si existe."""
    meta = {}
    if not ckpt_path:
        return meta
    mp = Path(ckpt_path).with_suffix(Path(ckpt_path).suffix + ".meta.json")
    if mp.exists():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return meta or {}

def _strip_module_prefix(sd: dict):
    """Quita 'module.' si el checkpoint procede de DataParallel."""
    if not sd:
        return sd
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def load_autoencoder(ae_type: str, ckpt: str, device: str, input_dim: int):
    """
    Construye el AE con los argumentos requeridos y carga pesos.
    - Intenta deducir latent_dim/hidden_dim/beta desde <ckpt>.meta.json.
    - Fallbacks sensatos si no hay meta.
    """
    if not ae_type:
        return None

    meta = _read_meta_from_ckpt(ckpt)
    latent_dim = int(meta.get("latent_dim", 64))
    hidden_dim = int(meta.get("hidden_dim", 512))
    beta       = float(meta.get("beta", 1.0))

    if ae_type.lower() == "dae":
        # Firma típica: (input_dim, latent_dim, hidden_dim=...)
        from models.denoising_autoencoder import DenoisingAutoencoder as DAE
        ae = DAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    elif ae_type.lower() == "vae":
        # Firma típica: (input_dim, latent_dim, hidden_dim=..., beta=...)
        from models.variational_autoencoder import VariationalAutoencoder as VAE
        try:
            ae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, beta=beta).to(device)
        except TypeError:
            # Si la clase no acepta beta en __init__
            ae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    elif ae_type.lower() == "cae":
        from models.contrastive_autoencoder import ContrastiveAutoencoder as CAE
        ae = CAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown ae_type={ae_type}")

    if ckpt:
        sd = torch.load(ckpt, map_location=device)
        state_dict = sd.get("state_dict", sd)
        state_dict = _strip_module_prefix(state_dict)
        missing, unexpected = ae.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys in AE state_dict: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys in AE state_dict: {len(unexpected)}")
    ae.eval()
    # Garantiza que la clase expone latent_dim para las aserciones
    if not hasattr(ae, "latent_dim"):
        ae.latent_dim = latent_dim
    return ae

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ae_type", default=None, choices=[None, "dae", "vae", "cae"])
    ap.add_argument("--ae_ckpt", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) Dimensión del embedder (SBERT)
    sbert = SentenceTransformer(args.model, device=args.device)
    sbert_dim = int(getattr(sbert, "get_sentence_embedding_dimension")())
    print(f"[SBERT] embedding dimension = {sbert_dim}")
    assert sbert_dim == 384, f"Expected 384 for MiniLM-L6-v2, got {sbert_dim}"

    # 2) Construir compresor con AE real
    ae = load_autoencoder(args.ae_type, args.ae_ckpt, args.device, input_dim=sbert_dim)
    comp = EmbeddingCompressor(base_model_name=args.model, autoencoder=ae, device=args.device)
    print(f"[Compressor] input_dim={comp.input_dim}, latent_dim={comp.latent_dim}")
    assert comp.input_dim == sbert_dim

    # 3) Textos de prueba
    texts = [
        "The Eiffel Tower is in Paris.",
        "Neural networks can compress embeddings.",
        "SQuAD is a QA dataset."
    ]

    # 4) Sin compresión
    z_orig = comp.encode_text(texts, compress=False)
    print(f"[Uncompressed] shape={tuple(z_orig.shape)}, dtype={z_orig.dtype}")
    assert z_orig.shape[1] == sbert_dim

    # 5) Con compresión (si hay AE)
    if ae is not None:
        z_lat = comp.encode_text(texts, compress=True)
        print(f"[Compressed]  shape={tuple(z_lat.shape)}, dtype={z_lat.dtype}")
        assert z_lat.shape[1] == getattr(ae, "latent_dim", None) == comp.latent_dim, \
            f"Latent dim mismatch: got {z_lat.shape[1]}, expected {comp.latent_dim}"
        cr = comp.input_dim / comp.latent_dim
        print(f"[Compression ratio] dim_in/dim_out = {comp.input_dim}/{comp.latent_dim} = {cr:.2f}×")
    else:
        print("[Note] No AE provided; skipping compressed check.")

if __name__ == "__main__":
    main()

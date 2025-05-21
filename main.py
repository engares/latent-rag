import argparse
import os
import torch
from dotenv import load_dotenv

from retrieval.embedder import EmbeddingCompressor
from retrieval.retriever import retrieve_top_k
from models.variational_autoencoder import VariationalAutoencoder
from generation.generator import RAGGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation_torch
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device


def load_autoencoder(models_cfg: dict, ae_type: str = "none", device: str = "cpu"):
    """Construye y carga el auto‑encoder solicitado.  
    Actualmente sólo se implementa la ruta VAE; amplíe aquí para DAE/CAE."""
    if ae_type == "none" or ae_type not in models_cfg:
        return None

    if ae_type == "vae":
        mcfg = models_cfg["vae"]
        model = VariationalAutoencoder(
            input_dim=mcfg["input_dim"],
            latent_dim=mcfg["latent_dim"],
            hidden_dim=mcfg.get("hidden_dim", 512),
        ).to(device)
        model.load_state_dict(torch.load(mcfg["checkpoint"], map_location=device))
        model.eval()
        return model

    raise ValueError(f"Tipo de auto‑encoder '{ae_type}' no soportado todavía.")


def main() -> None:
    # ---------------- CLI ----------------
    parser = argparse.ArgumentParser(
        description="RAG Pipeline with Autoencoders for TFM",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/config.yaml",
        help="Ruta al fichero YAML de configuración",
    )
    parser.add_argument(
        "--ae-type",
        choices=["none", "vae"],
        default="vae",
        help="Tipo de auto‑encoder a emplear (none desactiva la compresión)",
    )
    parser.add_argument(
        "--visualize-embeddings",
        action="store_true",
        help="Visualizar embeddings comprimidos con t‑SNE",
    )
    parser.add_argument(
        "--evaluate-autoencoder",
        action="store_true",
        help="Calcular pérdida de reconstrucción tras la compresión",
    )
    args = parser.parse_args()

    # --------------- Configuración ---------------
    load_dotenv()
    cfg = load_config(args.config)

    # Resolución de dispositivo y semilla
    device = resolve_device(cfg.get("training", {}).get("device"))
    set_seed(cfg.get("training", {}).get("seed", 42))

    # --------------- Embedding model ---------------
    emb_model_name = cfg.get("embedding_model", {}).get(
        "name", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------- Autoencoder ---------------
    ae = load_autoencoder(cfg.get("models", {}), args.ae_type, device)

    # --------------- Compressor ---------------
    compressor = EmbeddingCompressor(
        base_model_name=emb_model_name, autoencoder=ae, device=device
    )

    # --------- Corpus de demostración (reemplazar) ---------
    corpus = [
        "Paris is the capital of France.",
        "The Pythagorean theorem applies to right‑angled triangles.",
        "The Spanish Civil War began in 1936.",
        "GPT is a language model developed by OpenAI.",
        "Autoencoders allow nonlinear compression.",
    ]
    query = ["Which model does OpenAI use for text generation?"]
    

    # --------------- Codificación de textos ---------------
    doc_emb = compressor.encode_text(corpus, compress=True)
    q_emb = compressor.encode_text(query, compress=True)

    # --------------- Diagnósticos opcionales ---------------
    if args.evaluate_autoencoder and ae is not None:
        from evaluation.autoencoder_metrics import evaluate_reconstruction_loss

        print("[INFO] Evaluando reconstrucción…")
        recon = ae(doc_emb.to(device))
        loss = evaluate_reconstruction_loss(doc_emb, recon.detach().cpu())
        print(f"Reconstruction MSE: {loss:.4f}")

    if args.visualize_embeddings:
        from evaluation.autoencoder_metrics import visualize_embeddings

        visualize_embeddings(doc_emb)

    # --------------- Recuperación ---------------
    retr_cfg = cfg.get("retrieval", {})
    top_k = retr_cfg.get("top_k", 5)
    sim_metric = retr_cfg.get("similarity_metric", "cosine")

    retrieved = retrieve_top_k(q_emb, doc_emb, corpus, k=top_k, metric=sim_metric)
    retrieved_docs, _ = zip(*retrieved)

    # Evaluación de recuperación (ejemplo)
    relevant = ["GPT is a language model developed by OpenAI."]
    ret_metrics = evaluate_retrieval(list(retrieved_docs), relevant)
    print("\n[RETRIEVAL RESULTS]")
    for m, v in ret_metrics.items():
        print(f"{m}: {v:.4f}")

    # --------------- Generación ---------------
    gen_cfg = cfg.get("generation", {})
    sys_prompt_path = gen_cfg.get(
        "system_prompt_path", "./config/prompts/system_prompt.txt"
    )
    generator = RAGGenerator(
        config_path=args.config,
        llm={"system_prompt_path": sys_prompt_path},  # override incoherencia YAML
    )

    answer = generator.generate(query[0], list(retrieved_docs))
    print("\n[GENERATED RESPONSE]\n", answer)

    # Evaluación (demo)
    gen_metrics_cfg = cfg.get("evaluation", {}).get(
        "generation_metrics", ["ROUGE-L", "BLEU"]
    )
    gen_scores = evaluate_generation_torch(references=relevant, candidates=[answer], metrics=gen_metrics_cfg)
    print("\n[GENERATION RESULTS]")
    for m, v in gen_scores.items():
        print(f"{m}: {v:.4f}")


if __name__ == "__main__":
    main()

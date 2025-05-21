import argparse
import os
import torch
from dotenv import load_dotenv

from retrieval.embedder import EmbeddingCompressor
from retrieval.retriever import retrieve_top_k
from models.variational_autoencoder import VariationalAutoencoder
from generation.generator import RAGGenerator
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation_bootstrap as eval_gen
from utils.load_config import load_config
from utils.training_utils import set_seed, resolve_device

###############################################################################
#  UTILIDADES                                                                 #
###############################################################################

def load_autoencoder(models_cfg: dict, ae_type: str, device: str):
    if ae_type == "none" or ae_type not in models_cfg:
        return None
    if ae_type == "vae":
        mcfg = models_cfg["vae"]
        model = VariationalAutoencoder(
            mcfg["input_dim"], mcfg["latent_dim"], mcfg.get("hidden_dim", 512)
        ).to(device)
        model.load_state_dict(torch.load(mcfg["checkpoint"], map_location=device))
        model.eval()
        return model
    raise ValueError(f"Auto‑encoder '{ae_type}' no soportado.")

###############################################################################
#  MAIN                                                                       #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Demo RAG + Autoencoders")
    parser.add_argument("--config", default="./config/config.yaml")
    parser.add_argument("--ae-type", choices=["none", "vae"], default="vae")
    args = parser.parse_args()

    # Config & env
    load_dotenv()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("training", {}).get("device"))
    set_seed(cfg.get("training", {}).get("seed", 42))

    # Components ------------------------------------------------------------
    ae = load_autoencoder(cfg["models"], args.ae_type, device)
    compressor = EmbeddingCompressor(
        base_model_name=cfg["embedding_model"]["name"], autoencoder=ae, device=device
    )

    # Tiny toy corpus -------------------------------------------------------
    corpus = [
        "Paris is the capital of France.",
        "The Pythagorean theorem applies to right‑angled triangles.",
        "The Spanish Civil War began in 1936.",
        "GPT is a language model developed by OpenAI.",
        "Autoencoders allow nonlinear compression.",
    ]
    query = ["Which model does OpenAI use for text generation?"]

    doc_emb = compressor.encode_text(corpus, compress=True)
    q_emb = compressor.encode_text(query, compress=True)

    # Retrieval -------------------------------------------------------------
    retr_cfg = cfg["retrieval"]
    retrieved = retrieve_top_k(
        q_emb, doc_emb, corpus, k=retr_cfg.get("top_k", 5), metric=retr_cfg["similarity_metric"]
    )
    retrieved_docs, _ = zip(*retrieved)

    relevant = ["GPT is a language model developed by OpenAI."]
    ret_scores = evaluate_retrieval(list(retrieved_docs), relevant)
    print("\n[RETRIEVAL RESULTS]")
    for k, v in ret_scores.items():
        print(f"{k}: {v:.4f}")

    # Generation ------------------------------------------------------------
    gen = RAGGenerator(config_path=args.config)
    answer = gen.generate(query[0], list(retrieved_docs))
    print("\n[GENERATED RESPONSE]\n", answer)

    # Generation metrics (bootstrap, need ≥30 examples) ---------------------
    refs = relevant * 30  # replicamos para cumplir mínimo bootstrap
    cands = [answer] * 30
    gen_scores = eval_gen(refs, cands)
    print("\n[GENERATION RESULTS] (bootstrap 95% CI)")
    for m, d in gen_scores.items():
        print(f"{m}: {d['mean']:.2f}  (CI: {d['ci_lower']:.2f}–{d['ci_upper']:.2f})")

if __name__ == "__main__":
    main()

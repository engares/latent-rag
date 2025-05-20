import argparse
import os
import yaml
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



def main() -> None:
    # ------------------ CLI Arguments ------------------
    parser = argparse.ArgumentParser(description="RAG Pipeline with Autoencoders for TFM")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--visualize-embeddings", action="store_true", help="Visualize compressed embeddings using t‑SNE")
    parser.add_argument("--evaluate-autoencoder", action="store_true", help="Compute Reconstruction Loss after compression")
    args = parser.parse_args()

    # ------------------ Configuration ------------------
    load_dotenv()
    config = load_config(args.config)

    # ------------------ Embeddings and Autoencoder ------------------
    ae_cfg = config.get("autoencoder", {})
    embedding_model = ae_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    # Load Autoencoder checkpoint if requested
    autoencoder = None
    if ae_cfg.get("type", "none") == "vae":
        autoencoder = VariationalAutoencoder(ae_cfg["input_dim"], ae_cfg["latent_dim"])
        autoencoder.load_state_dict(torch.load(ae_cfg["checkpoint"]))

    # Create embedder (raw or compressed)
    compressor = EmbeddingCompressor(base_model_name=embedding_model, autoencoder=autoencoder)

    # -----------------------------------------------------------------
    # Demo corpus & query — replace with dataset loader in real pipeline
    # -----------------------------------------------------------------
    corpus = [
        "Paris is the capital of France.",
        "The Pythagorean theorem applies to right‑angled triangles.",
        "The Spanish Civil War began in 1936.",
        "GPT is a language model developed by OpenAI.",
        "Autoencoders allow nonlinear compression."
    ]
    query = ["Which model does OpenAI use for text generation?"]

    # -----------------------------------------------------------------
    # Encode texts
    # -----------------------------------------------------------------
    doc_embeddings = compressor.encode_text(corpus, compress=True)
    query_embedding = compressor.encode_text(query, compress=True)

    # -----------------------------------------------------------------
    # Optional: autoencoder diagnostics
    # -----------------------------------------------------------------
    if args.evaluate_autoencoder:
        from evaluation.autoencoder_metrics import evaluate_reconstruction_loss
        print("[INFO] Autoencoder Reconstruction Loss evaluation enabled.")
        # Example: compute loss on a small batch (requires original x)
        # loss = evaluate_reconstruction_loss(x_batch, x_reconstructed)

    if args.visualize_embeddings:
        from evaluation.autoencoder_metrics import visualize_embeddings
        print("[INFO] Embedding visualization enabled.")
        visualize_embeddings(doc_embeddings)

    # ------------------ Retrieval ------------------
    retrieval_cfg = config.get("retrieval", {})  # <‑‑ now actually used
    top_k = retrieval_cfg.get("top_k", 5)
    similarity_metric = retrieval_cfg.get("similarity_metric", "cosine")

    retrieved_docs = retrieve_top_k(query_embedding, doc_embeddings, corpus, k=top_k, metric=similarity_metric)
    retrieved_ids = [doc for doc, _ in retrieved_docs]

    # ------------------ Retrieval Evaluation ------------------
    relevant_docs = ["GPT is a language model developed by OpenAI."]
    retrieval_results = evaluate_retrieval(retrieved_ids, relevant_docs)

    print("\n[RETRIEVAL RESULTS]")
    for metric, value in retrieval_results.items():
        print(f"{metric}: {value:.4f}")

    # ------------------ Generation ------------------
    generator = RAGGenerator(config_path=args.config)
    generated_response = generator.generate(query[0], [doc for doc, _ in retrieved_docs])

    print("\n[GENERATED RESPONSE]")
    print(generated_response)

    # ------------------ Generation Evaluation ------------------
    generation_cfg = config.get("evaluation", {}).get("generation_metrics", ["ROUGE-L", "BLEU"])
    gen_results = evaluate_generation_torch(
        references=relevant_docs,
        candidates=[generated_response],
        metrics=generation_cfg,
    )

    print("\n[GENERATION RESULTS]")
    for metric, value in gen_results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

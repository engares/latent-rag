
# rag\_autoencoder\_tfm

A repository for training and evaluating retrieval-augmented generation (RAG) pipelines enhanced by various autoencoder compression methods. Supported variants include:

* Variational Autoencoder (VAE)
* Denoising Autoencoder (DAE)
* Contrastive Autoencoder (CAE)

This framework covers from data preparation, model training, retrieval to optional generation with LLMs, and comprehensive evaluation metrics.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Data Preparation](#data-preparation)
6. [Training](#training)

   * [VAE](#vae)
   * [DAE](#dae)
   * [CAE](#cae)
7. [Pipeline Execution](#pipeline-execution)
8. [Evaluation](#evaluation)
9. [Code Style](#code-style)
10. [Project Structure](#project-structure)
11. [Testing](#testing)
12. [Scripts](#scripts)

---

## Features

* Encode text embeddings using SBERT and compress with VAE/DAE/CAE.
* Retrieve top‑k relevant documents using cosine, Euclidean or Mahalanobis similarity.
* Generate answers via RAG using OpenAI GPT-4o-mini with custom system prompts.
* Evaluate retrieval (Recall\@k, MRR, nDCG) and generation (BLEU, ROUGE-L, METEOR) with bootstrap CIs.
* Configurable via YAML; extensible for other datasets or LLM providers.

## Prerequisites

* Python ≥ 3.10
* GPU recommended for large-scale embedding and training.
* OpenAI API key (set in `.env` as `OPENAI_API_KEY`).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

```bash

git clone https://github.com/engares/latent-rag.git
cd latent-rag
```

2. Create `.env` with your OpenAI key:
```ini
OPENAI_API_KEY=your_api_key_here
````

3. Adjust paths and hyperparameters in **`config/config.yaml`** as needed.

## Configuration

**`config/config.yaml`** contains:

* Project metadata (name, version)
* Directory paths (`data_dir`, `checkpoints_dir`, `logs_dir`)
* Embedding model settings
* Autoencoder parameters and checkpoints
* Training hyperparameters (batch size, epochs, LR)
* Retrieval & generation options
* Evaluation metrics
* Logging level and file

System prompt for generation is located in **`config/prompts/system_prompt.txt`**.

## Data Preparation

Data tensors for SQuAD are generated automatically when running training or pipeline. To prepare manually:

```bash
python -c "from utils.data_utils import ensure_squad_data; ensure_squad_data(output_dir='./data/SQUAD')"
```

This creates:

* `data/SQUAD/squad_vae_embeddings.pt`
* `data/SQUAD/squad_dae_embeddings.pt`
* `data/SQUAD/squad_contrastive_embeddings.pt`

## Training

### VAE

```bash
python training/train_vae.py \
  --dataset squad \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_path models/checkpoints/vae_text.pth
```

### DAE

```bash
python training/train_dae.py \
  --dataset squad \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_path models/checkpoints/dae_text.pth
```

### CAE

```bash
python training/train_cae.py \
  --dataset squad \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --save_path models/checkpoints/contrastive_ae.pth
```

All training scripts support early stopping, checkpointing and configurable device.

## Pipeline Execution

Run the end-to-end RAG pipeline:

```bash
python main.py --config config/config.yaml --ae_type vae
```

Replace `--ae_type` with `dae`, `contrastive`, `all` or `none`. The pipeline will:

1. Encode corpus and queries
2. Retrieve top‑k documents
3. Generate answers via GPT-4o-mini
4. Evaluate retrieval and generation metrics

## Evaluation

* Retrieval metrics: per-query and aggregated Recall\@k, MRR, nDCG.
* Generation metrics: BLEU, ROUGE-L, METEOR with 95% bootstrap CIs.
* Visualise embeddings via **`evaluation/autoencoder_metrics.py`** (t-SNE plots).

Commands:

```python
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.generation_metrics import evaluate_generation_bootstrap
```


## Project Structure

```text
src/
├── config/           # YAML and prompts
├── data/             # Embedding tensors and loaders
├── evaluation/       # Metrics and visualisations
├── generation/       # RAG generator
├── models/           # AE implementations
├── retrieval/        # Embeddings & retriever
├── training/         # Training scripts and loss functions
├── utils/            # Helpers (config, data, logging)
├── test/             # Unit tests (pytest)
├── main.py           # CLI orchestration
├── requirements.txt
└── style_guide.md
```

## Testing

Run all tests via pytest:

```bash
pytest -q
```

Coverage threshold: 80% (unit tests for data processing, models, retrieval, evaluation, training scripts).


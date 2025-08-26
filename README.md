# latent-rag — RAG with Latent Compression (VAE/DAE/CAE)

A research framework to **train and evaluate Retrieval-Augmented Generation (RAG)** pipelines where text embeddings are **compressed with autoencoders** to reduce storage and accelerate retrieval with minimal accuracy loss.

Supported autoencoders:
- **Variational Autoencoder (VAE)**
- **Denoising Autoencoder (DAE)**
- **Contrastive Autoencoder (CAE)**

The framework covers **data preparation → training → retrieval (FAISS) → optional generation (LLM) → evaluation** with standard IR and NLG metrics.

---

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation & Setup](#installation--setup)  
4. [Configuration](#configuration)  
5. [Data Preparation](#data-preparation)  
6. [Training](#training)  
   - [VAE](#vae)  
   - [DAE](#dae)  
   - [CAE](#cae)  
   - [Hyperparameter Search (Optuna)](#hyperparameter-search-optuna)
7. [Pipeline Execution](#pipeline-execution)  
8. [Evaluation](#evaluation)  
9. [Project Structure](#project-structure)  
10. [Testing](#testing)  
11. [Reproducibility](#reproducibility)

---

## Features

- **SBERT → AE compression**: Encode with Sentence-Transformers (SBERT) and compress embeddings using VAE/DAE/CAE.
- **FAISS retrieval backend**: `flatip`, `hnsw`, `ivfpq`, with **GPU auto-detection** (`use_gpu: auto`) and **index persistence**.
- **Cosine or Euclidean similarity**:  
  - *Cosine* implemented as **Inner Product + L2 normalization** (`normalize_l2: true`).  
  - *Euclidean* (L2) also supported.  
  *(Mahalanobis is not included in this repo to avoid over-promising.)*
- **Chunking for retrieval**: Sliding-window or semantic chunking with **chunk→document score aggregation** and optional storage of chunk text/metadata.
- **Optional RAG generation**: Integrates with OpenAI API (system prompt configurable) to generate answers on top of retrieved contexts.
- **Evaluation**: Retrieval metrics (Recall@k, MRR, nDCG) and generation metrics (BLEU, ROUGE-L, METEOR) with bootstrap confidence intervals.
- **Config-first**: All behavior controlled via YAML. Extensible to new datasets, models and providers.

---

## Prerequisites

- **Python ≥ 3.10**
- **PyTorch** (GPU recommended for training and large-scale embedding)
- **FAISS** (CPU and/or GPU build, depending on your environment)
- **OpenAI API key** (only required if you enable the generation step)

> Reference hardware (dev): **RTX 4060 8 GB VRAM**.

```bash
pip install -r requirements.txt
````

---

## Installation & Setup

```bash
git clone https://github.com/engares/latent-rag.git
cd latent-rag
```

Create a `.env` if you plan to use generation:

```ini
OPENAI_API_KEY=your_api_key_here
```

Adjust paths and hyperparameters in **`config/config.yaml`** as needed.

---

## Configuration

The main configuration lives in **`config/config.yaml`**. Below is a **minimal, self-consistent example** with the key blocks you will likely modify:

```yaml
project:
  name: latent-rag
  version: "0.1.0"

paths:
  data_dir: "./data"
  checkpoints_dir: "./models/checkpoints"
  logs_dir: "./logs"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "auto"        # 'cuda' | 'cpu' | 'auto'

models:
  vae:
    checkpoint: "./models/checkpoints/vae_text.pth"
    dataset_file: "squad_vae_embeddings.pt"
  dae:
    checkpoint: "./models/checkpoints/dae_text.pth"
    dataset_file: "squad_dae_embeddings.pt"
  cae:
    checkpoint: "./models/checkpoints/cae_text.pth"
    dataset_file: "squad_cae_embeddings.pt"   # unified name

training:
  batch_size: 128
  epochs: 50
  lr: 1e-3
  seed: 42
  deterministic: true
  early_stopping:
    enabled: true
    patience: 5

chunking:
  enabled: true
  mode: "sliding"        # 'sliding' | 'semantic'
  max_tokens: 256
  stride: 64
  store_chunk_text: true
  index_out: "./data/index/chunks_meta.parquet"
  aggregate_chunks: true  # aggregate chunk scores back to the document

retrieval:
  backend: "faiss"               # currently 'faiss' is the supported backend
  metric: "cosine"               # 'cosine' | 'euclidean'
  index_type: "flatip"           # 'flatip' (for cosine) | 'hnsw' | 'ivfpq'
  normalize_l2: true             # MUST be true when metric=cosine (IP ≈ cosine)
  use_gpu: "auto"                # 'auto' | true | false
  index_path: "./data/index/faiss.idx"
  top_k: 10
  candidate_k: 100               # for multi-stage retrieval if used

generation:
  enabled: false
  provider: "openai"
  model: "gpt-4o-mini"
  max_tokens: 256
  temperature: 0.0
  system_prompt_path: "./config/prompts/system_prompt.txt"

evaluation:
  k_list: [1, 5, 10]
  bootstrap_iters: 1000
  output_dir: "./reports"

logging:
  level: "INFO"
  file: "./logs/run.log"
```

**Notes**

* If `retrieval.metric = cosine`, set `normalize_l2: true` and prefer `index_type: flatip`.
* `use_gpu: auto` will fall back to CPU if CUDA/FAISS-GPU are unavailable, logging a clear message.
* The dataset filenames under `models.*.dataset_file` are **authoritative**—adjust them if your preprocessing uses different names.

---

## Data Preparation

SQuAD assets and caches are created on demand by the training/pipeline. To run it explicitly:

```bash
python -c "from utils.data_utils import ensure_squad_data; ensure_squad_data(output_dir='./data/SQUAD')"
```

This step will:

* Download/prepare SQuAD (specify v1/v2 inside the function/config).
* Cache SBERT embeddings for corpus/queries (under `data/SQUAD/sbert_cache/...`).
* Prepare any CAE contrastive pairs if configured.
* Write tensors/metadata controlled by `models.*.dataset_file`.

> Ensure the **SQuAD version and policy for unanswerables** are documented in your experiments.

---

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
  --save_path models/checkpoints/cae_text.pth
```

All scripts support early stopping, checkpointing and device selection from config/CLI.

Note: You can run evaluation via the end-to-end pipeline in `main.py`. Omit `--generate` for retrieval-only metrics, or add `--generate` to include generation metrics. See the Pipeline Execution and Evaluation sections below for details.

#### Hyperparameter Search (Optuna)

```bash
python hparam_search.py --model cae --sweep sweeps/cae.yaml --n_trials 25
```

The search ranks trials using the **validation objective** recorded in each trial’s metadata, keeping checkpoint↔meta filenames consistent.

---

## Pipeline Execution

Run end-to-end RAG:

```bash
python main.py --config config/config.yaml --ae_type vae
```

`--ae_type` can be `vae | dae | cae | none | all`. The pipeline:

1. Encodes corpus & queries (SBERT → optional AE compression).
2. Retrieves top-k documents:

   * **FAISS** backend (builds/loads `index_path` with the selected `index_type`).
   * **Chunking** if enabled, with chunk→document aggregation.
   * **Cosine** (IP + L2) or **Euclidean** distance per config.
3. *(Optional)* Generates answers via OpenAI if `generation.enabled=true` or CLI `--generate`.
4. Evaluates retrieval/generation and writes reports to `evaluation.output_dir`.

To enable generation via CLI:

```bash
python main.py --config config/config.yaml --ae_type vae --generate
```

---

## Evaluation

Quick evaluation via `main.py`:

```bash
# Retrieval-only metrics (Recall@k, MRR, nDCG)
python main.py --config config/config.yaml --ae_type all 

# Include generation metrics (BLEU, ROUGE-L, METEOR)
python main.py --config config/config.yaml --ae_type vae --generate
```

Results are written to `evaluation.output_dir` 

* **Retrieval**: per-query and aggregate **Recall\@k, MRR, nDCG** (configurable `k_list`).
* **Generation**: **BLEU, ROUGE-L, METEOR** with **95% bootstrap CIs**.
* **Embedding diagnostics** (optional): t-SNE/PCA visualisations comparing original vs. compressed embeddings.

Example visualisation:

```bash
python -m utils.visualization_exp \
  --sbert-cache data/SQUAD/sbert_cache/sbert_2254a38d6b_all-MiniLM-L6-v2.pt \
  --checkpoint  models/checkpoints/cae_text.pth \
  --projection  tsne \
  --components  2 \
  --sample-size 1200 \
  --k-near 10
```

Outputs are saved under `fig/` (e.g., 2D t-SNE with distance distributions and recall overlays).

---

## Project Structure

```text
.
├── config/           # YAML and prompts
├── data/             # Datasets, caches, and tensors
├── evaluation/       # Metrics and visualizations
├── generation/       # RAG generator (OpenAI, prompts)
├── models/           # AE implementations (VAE/DAE/CAE)
├── retrieval/        # FAISS indexes and retrievers
├── training/         # Trainers and loss functions
├── utils/            # Config, data, logging, helpers
├── tests/            # Unit/integration tests (pytest)
├── main.py           # Orchestration CLI
├── requirements.txt
└── style_guide.md
```

---

## Testing

Run the test suite:

```bash
PYTHONPATH=. pytest -q
```

## Reproducibility

* Set `training.deterministic: true` and a fixed `training.seed`.
* Log hardware, CUDA, FAISS build, and key package versions.
* When reporting results, specify:

  * Similarity metric (cosine = IP + L2 normalization, or Euclidean).
  * SQuAD version and unanswerable policy.
  * Chunking parameters (`mode`, `max_tokens`, `stride`) and whether chunk→doc aggregation was enabled.
  * FAISS index type, GPU/CPU, and index size.







├── models/
│   ├── base_autoencoder.py        # Clase abstracta
│   ├── vae.py                     # Variational Autoencoder
│   ├── dae.py                     # Denoising Autoencoder
│   ├── contrastive_ae.py          # Autoencoder con contraste
│
├── training/
│   ├── train_ae.py                # Rutinas de entrenamiento
│   ├── loss_functions.py          # Pérdidas personalizadas
│
├── retrieval/
│   ├── embedder.py                # Codificadores tradicionales y refinados
│   ├── retriever.py               # Recuperación (BM25, DPR, embeddings comprimidos)
│
├── generation/
│   ├── generator.py               # Integración con LLM vía API
│
├── evaluation/
│   ├── retrieval_metrics.py                 # Métricas de recuperación y generación
|   |_ autoencoder_metrics.py
|   |_ generation_metrics.py
│
├── data/
│   └── ...                        # Datasets y preprocessors
│
├── config/
│   └── config.yaml                # Parámetros experimentales
│
├── main.py                        # Script principal para orquestar el flujo
└── requirements.txt               # Dependencias
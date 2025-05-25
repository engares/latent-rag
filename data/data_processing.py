#/data_processing.py

import random
import json
import re
from typing import List, Dict
from datasets import load_dataset

# ---------------------------------------------------------
# UDA Dataset Preprocessing for Autoencoder Training
# ---------------------------------------------------------
# Supports: Denoising AE (with artificial noise), VAE, Contrastive AE
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def add_noise(text: str, removal_prob=0.1, swap_prob=0.05) -> str:
    words = text.split()
    # Remove tokens
    words = [w for w in words if random.random() > removal_prob]
    # Swap nearby tokens
    for i in range(len(words)-1):
        if random.random() < swap_prob:
            words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)

def build_dae_dataset(samples: List[str]) -> List[Dict[str, str]]:
    dataset = []
    for original in samples:
        noisy = add_noise(original)
        dataset.append({"input": noisy, "target": original})
    return dataset

def build_contrastive_pairs(dataset, max_negatives=1) -> List[Dict]:
    pairs = []
    for example in dataset:
        q = example["query"]
        pos = example["positive_passages"][0]["text"]
        negs = [n["text"] for n in example["negative_passages"][:max_negatives]]
        for neg in negs:
            pairs.append({"query": q, "positive": pos, "negative": neg})
    return pairs

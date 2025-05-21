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

def load_uda(split="train", max_samples=5000):
    print("[INFO] Loading UDA benchmark dataset...")
    uda = load_dataset("osunlp/uda", split=split)
    return uda.select(range(min(max_samples, len(uda))))

if __name__ == "__main__":
    # Load base data
    data = load_uda(split="train", max_samples=1000)
    texts = [clean_text(p["text"]) for row in data for p in row["positive_passages"][:1]]

    # Generate DAE data
    dae_data = build_dae_dataset(texts)
    with open("./data/uda_dae_train.jsonl", "w", encoding="utf-8") as f:
        for item in dae_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Generate contrastive pairs
    contrastive = build_contrastive_pairs(data, max_negatives=1)
    with open("./data/uda_contrastive_train.jsonl", "w", encoding="utf-8") as f:
        for item in contrastive:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("[INFO] Dataset files written to ./data/")

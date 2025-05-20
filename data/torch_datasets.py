import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional

# ---------------------------------------------------------
# PyTorch Datasets for Autoencoder Training (DAE & Contrastive)
# ---------------------------------------------------------

class DAEDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256):
        with open(path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        noisy_input = item["input"]
        target = item["target"]

        encoded = self.tokenizer(
            noisy_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoded = self.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_ids": target_encoded["input_ids"].squeeze(0)
        }

class ContrastiveDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 256):
        with open(path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        query = item["query"]
        pos = item["positive"]
        neg = item["negative"]

        q_enc = self.tokenizer(query, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        p_enc = self.tokenizer(pos, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        n_enc = self.tokenizer(neg, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "query_ids": q_enc["input_ids"].squeeze(0),
            "pos_ids": p_enc["input_ids"].squeeze(0),
            "neg_ids": n_enc["input_ids"].squeeze(0)
        }

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    dae_ds = DAEDataset("./data/uda_dae_train.jsonl", tokenizer)
    contrastive_ds = ContrastiveDataset("./data/uda_contrastive_train.jsonl", tokenizer)

    print("[DAE]", dae_ds[0])
    print("[Contrastive]", contrastive_ds[0])

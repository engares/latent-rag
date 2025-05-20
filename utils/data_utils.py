import os
import json
from data.data_processing import load_uda, clean_text, build_dae_dataset, build_contrastive_pairs

def ensure_uda_data(output_dir: str = "./data", max_samples: int | None = None):
    os.makedirs(output_dir, exist_ok=True)
    print("[INFO] Loading UDA dataset...")

    uda = load_uda(split="train", max_samples=max_samples or 9999999)
    texts = [clean_text(p["text"]) for row in uda for p in row["positive_passages"][:1]]

    dae_path = os.path.join(output_dir, "uda_dae_train.jsonl")
    if not os.path.exists(dae_path):
        print("[INFO] Building DAE version...")
        dae_data = build_dae_dataset(texts)
        with open(dae_path, "w", encoding="utf-8") as f:
            for item in dae_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] DAE dataset written to: {dae_path}")

    contrastive_path = os.path.join(output_dir, "uda_contrastive_train.jsonl")
    if not os.path.exists(contrastive_path):
        print("[INFO] Building Contrastive version...")
        contrastive = build_contrastive_pairs(uda, max_negatives=1)
        with open(contrastive_path, "w", encoding="utf-8") as f:
            for item in contrastive:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[INFO] Contrastive dataset written to: {contrastive_path}")

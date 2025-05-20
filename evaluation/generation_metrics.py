import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from typing import List, Dict

def compute_bleu_torch(candidates: List[str], references: List[str]) -> float:
    metric = BLEUScore(n_gram=4)
    return metric(candidates, [[ref] for ref in references]).item()

def compute_rouge_l_torch(candidates: List[str], references: List[str]) -> float:
    metric = ROUGEScore(rouge_keys=["rougeL"])
    scores = metric(candidates, references)
    return scores["rougeL_fmeasure"].item()

def evaluate_generation_torch(
    references: List[str], 
    candidates: List[str], 
    metrics: List[str] = ["ROUGE-L", "BLEU"]
) -> Dict[str, float]:
    assert len(references) == len(candidates), "El número de referencias y candidatos debe coincidir."
    results = {}
    
    if "BLEU" in metrics:
        results["BLEU"] = compute_bleu_torch(candidates, references)
    if "ROUGE-L" in metrics:
        results["ROUGE-L"] = compute_rouge_l_torch(candidates, references)
    
    return results

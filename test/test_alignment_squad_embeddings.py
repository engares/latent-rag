# test/test_alignment_squad_embeddings.py
# -*- coding: utf-8 -*-
"""
Test de alineamiento SQuAD ↔ embeddings
---------------------------------------
• Solo usa los 10 primeros ejemplos para acelerar.
• Regenera en memoria el índice con chunk_text.
"""
import torch
import unittest
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from utils.chunk_utils import build_chunked_corpus

class TestSquadEmbeddingAlignment(unittest.TestCase):
    MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
    TOKENIZER    = "bert-base-uncased"
    EMB_PATH     = Path("data/SQUAD/squad_contrastive_embeddings.pt")
    MAX_EXAMPLES = 5

    def setUp(self):
        # 1) Cargamos solo los primeros 10 ejemplos
        self.squad = load_dataset("squad", split=f"train[:{self.MAX_EXAMPLES}]")
        # 2) Regeneramos chunks+índice con texto incluido
        chunks, idx = build_chunked_corpus(
            self.squad,
            tokenizer_name=self.TOKENIZER,
            store_chunk_text=True,
        )
        self.chunk_index = idx
        # 3) Cargamos embeddings completos y forzamos CPU
        self.embeddings = torch.load(self.EMB_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        self.device = device

        # 4) Preparamos lista intercalada [q1, ch1, q2, ch2, ...]
        self.clean_texts: List[str] = []
        doc_groups = self.chunk_index[self.chunk_index["contains_answer"]]\
                          .groupby("doc_id").groups
        for doc_id_np, cids in doc_groups.items():
            doc_id = int(doc_id_np)
            q  = self.squad[doc_id]["question"].strip()
            ch = self.chunk_index.loc[cids[0], "chunk_text"]
            self.clean_texts.extend([q, ch])

    def test_embedding_exact_alignment(self):
        """Los embeddings guardados coinciden con los recalculados (solo 10 ej.)."""
        queries   = self.clean_texts[0::2]
        positives = self.clean_texts[1::2]

        # Recalcular y forzar a CPU
        q_ref = self.model.encode(queries, convert_to_tensor=True).to(self.device)
        p_ref = self.model.encode(positives, convert_to_tensor=True).to(self.device)

        # Tomamos sólo los primeros len(queries) de disco
        q_disk = self.embeddings["query"][: len(queries)].to(self.device)
        p_disk = self.embeddings["positive"][: len(positives)].to(self.device)

        for i in range(len(queries)):
            self.assertTrue(
                torch.allclose(q_ref[i], q_disk[i], atol=1e-5),
                f"Query #{i} difiere del recalculado."
            )
            self.assertTrue(
                torch.allclose(p_ref[i], p_disk[i], atol=1e-5),
                f"Positive #{i} difiere del recalculado."
            )

    def test_chunk_contains_answer(self):
        """Cada chunk positivo contiene el texto de la respuesta (case-insensitive)."""
        checked = 0
        for doc_id_np, rows in self.chunk_index.groupby("doc_id"):
            doc_id = int(doc_id_np)
            answer = self.squad[doc_id]["answers"]["text"][0].lower()
            candidates = (
                rows[rows["contains_answer"]]["chunk_text"]
                .dropna()
                .map(str.lower)
            )
            # debe encontrarse en al menos un fragmento
            self.assertTrue(
                any(answer in ch for ch in candidates),
                f"Doc {doc_id}: ‘{answer}’ no aparece en ningún chunk."
            )
            checked += 1

if __name__ == "__main__":
    unittest.main()

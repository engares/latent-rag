

# utils/chunk_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

import pandas as pd
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)

ChunkMode = Literal["sliding", "semantic"]
from tqdm import tqdm 

###############################################################################
# Main chunker                                                                #
###############################################################################

def chunk_context_with_alignment(
    context: str,
    answer_start: int,
    answer_end: int,
    *,
    max_tokens: int = 128,
    stride: int = 64,
    tokens_before: int = 32,
    tokens_after: int = 32,
    tokenizer_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
) -> List[str]:
    """Chunk context ensuring answer span tokens appear in at least one chunk."""
    tok = _get_tokenizer(tokenizer_name)

    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    try:
        t_start, t_end = _char_to_token_span(offsets, answer_start, answer_end)
    except ValueError:
        LOGGER.warning("[chunk_utils] Span alignment failed; using full context.")
        return [context.strip()]

    answer_ids = input_ids[t_start : t_end + 1]
    n_tokens = len(input_ids)

    # 1) Centred window ----------------------------------------------------
    win_start = max(0, t_start - tokens_before)
    win_end = min(n_tokens, t_end + tokens_after + 1)
    cur_len = win_end - win_start
    if cur_len < max_tokens:
        pad = max_tokens - cur_len
        pre = min(pad // 2, win_start)
        post = min(pad - pre, n_tokens - win_end)
        win_start -= pre
        win_end += post

    centred_ids = input_ids[win_start:win_end]
    centred_chunk = tok.decode(
        centred_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    # 2) Sliding windows ---------------------------------------------------
    sliding_chunks: List[Tuple[str, List[int]]] = []
    idx = 0
    while idx < n_tokens:
        sw_end = min(idx + max_tokens, n_tokens)
        ids_slice = input_ids[idx:sw_end]
        ch_str = tok.decode(
            ids_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        sliding_chunks.append((ch_str, ids_slice))
        if sw_end == n_tokens:
            break
        idx += stride

    # 3) Combine + deduplicate -------------------------------------------
    out: List[str] = []
    out_ids: List[List[int]] = []
    seen = set()

    # first centred chunk
    if centred_chunk:
        out.append(centred_chunk)
        out_ids.append(centred_ids)
        seen.add(centred_chunk)

    for ch_str, ids_slice in sliding_chunks:
        if ch_str and ch_str not in seen:
            out.append(ch_str)
            out_ids.append(ids_slice)
            seen.add(ch_str)

    # 4) Integrity check on original ids ----------------------------------
    if not any(_has_subseq(ids, answer_ids) for ids in out_ids):
        LOGGER.warning(
            "[chunk_utils] Answer span missing after chunking; adding full context."
        )
        out.append(context.strip())

    return out


# ============================== Tokenizer cache ==============================

@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tok.is_fast:
        raise ValueError(
            f"Tokenizer '{model_name}' must be a *fast* tokenizer to provide offset mappings."
        )
    return tok

# ============================== Data structures ==============================

@dataclass(frozen=True)
class ChunkRecord:
    """Lightweight record for a chunk with token/char spans."""
    doc_id: int
    tok_start: int
    tok_end: int           # inclusive
    char_start: int
    char_end: int          # exclusive
    text: str

# ============================== Helpers =====================================

def _has_subseq(hay: Sequence[int], needle: Sequence[int]) -> bool:
    n = len(needle)
    if n == 0 or n > len(hay):
        return False
    for i in range(len(hay) - n + 1):
        if hay[i : i + n] == list(needle):
            return True
    return False


def _char_to_token_span(
    offsets: Sequence[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> Tuple[int, int]:
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if tok_start is None and s <= char_start < e:
            tok_start = i
        if s < char_end <= e:
            tok_end = i
            break
    if tok_start is None or tok_end is None:
        raise ValueError("Answer span could not be aligned to token offsets.")
    return tok_start, tok_end

# ============================== Inference chunkers ===========================

# utils/chunk_utils.py

def sliding_window_chunker(
    text: str,
    *,
    max_tokens: int = 128,
    stride: int = 64,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[ChunkRecord]:
    """Return fixed-size overlapping chunks (token-based).

    The chunk text is built from character slices aligned to token boundaries
    to avoid WordPiece artifacts (e.g., '##suffix') and ensure that re-tokenising
    the chunk never exceeds `max_tokens`.
    """
    tok = _get_tokenizer(tokenizer_name)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]
    n = len(input_ids)

    out: List[ChunkRecord] = []
    i = 0
    while i < n:
        end = min(i + max_tokens, n)
        # character-aligned slice covering tokens [i, end)
        char_start = offsets[i][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end].strip()
        if chunk_text:
            out.append(
                ChunkRecord(
                    doc_id=-1,
                    tok_start=i,
                    tok_end=end - 1,
                    char_start=char_start,
                    char_end=char_end,
                    text=chunk_text,
                )
            )
        if end == n:
            break
        i += stride
    return out


def semantic_window_chunker(
    text: str,
    *,
    max_tokens: int = 128,
    stride: int = 64,
    min_tokens: int = 48,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    boundary_chars: str = ".!?;:\n",
) -> List[ChunkRecord]:
    """Return chunks that try to end at punctuation boundaries near max_tokens.

    Heuristic:
      - Start a window at `start`.
      - Prefer ending at the nearest token (within [start+min_tokens, start+max_tokens])
        whose *last character* is in `boundary_chars`.
      - If none is found, end at `start+max_tokens`.
    Text is reconstructed via character slicing aligned to token boundaries.
    """
    tok = _get_tokenizer(tokenizer_name)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]
    n = len(input_ids)

    out: List[ChunkRecord] = []
    seen_spans: set[Tuple[int, int]] = set()
    start = 0

    while start < n:
        hard_end = min(start + max_tokens, n)
        soft_floor = min(hard_end - 1, max(start + min_tokens, start + 1))

        # search backward for a punctuation boundary
        best_end = None
        j = hard_end - 1
        while j >= soft_floor:
            _, ce = offsets[j]
            if ce > 0:
                last_char = text[ce - 1]
                if last_char in boundary_chars:
                    best_end = j + 1  # make `end` exclusive
                    break
            j -= 1
        end = best_end or hard_end

        span = (start, end - 1)
        if span not in seen_spans:
            seen_spans.add(span)
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunk_text = text[char_start:char_end].strip()
            if chunk_text:
                out.append(
                    ChunkRecord(
                        doc_id=-1,
                        tok_start=start,
                        tok_end=end - 1,
                        char_start=char_start,
                        char_end=char_end,
                        text=chunk_text,
                    )
                )

        if end == n:
            break
        start += stride

    return out

# ============================== Builders ====================================
def build_chunked_corpus(
    squad_split,
    *,
    max_tokens: int = 128,
    stride: int = 64,
    tokens_before: int = 32,
    tokens_after: int = 32,
    tokenizer_name: str = "bert-base-uncased",
    store_chunk_text: bool = True,        # <-- NUEVO
) -> Tuple[List[str], pd.DataFrame]:
    """
    Devuelve (chunks, index).  
    Si *store_chunk_text* es False la columna `chunk_text NO se guarda,
    reduciendo drásticamente el peso en disco.
    """
    tok = _get_tokenizer(tokenizer_name)
    chunks: List[str] = []
    records: List[Dict] = []

    for doc_id, ex in tqdm(
        enumerate(squad_split),
        total=len(squad_split),
        desc="Chunking SQuAD",
    ):
        ctx = ex["context"].rstrip()
        if not ctx:
            continue
        answers = ex["answers"]
        if not answers["text"]:
            continue
        ans_text = answers["text"][0]
        a_start = answers["answer_start"][0]
        a_end = a_start + len(ans_text)

        doc_chunks = chunk_context_with_alignment(
            ctx,
            a_start,
            a_end,
            max_tokens=max_tokens,
            stride=stride,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokenizer_name=tokenizer_name,
        )

        answer_ids = tok(ans_text, add_special_tokens=False)["input_ids"]
        for ch in doc_chunks:
            cid = len(chunks)
            chunks.append(ch)
            ch_ids = tok(ch, add_special_tokens=False)["input_ids"]
            contains = _has_subseq(ch_ids, answer_ids)

            rec = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "contains_answer": contains,
            }
            if store_chunk_text:            
                rec["chunk_text"] = ch
            records.append(rec)

    index = pd.DataFrame.from_records(records).set_index("chunk_id")
    return chunks, index

def build_inference_corpus(
    docs: Sequence[str],
    *,
    mode: ChunkMode = "sliding",
    max_tokens: int = 128,
    stride: int = 64,
    min_tokens: int = 48,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    store_chunk_text: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """Materialise a chunk-level corpus for test/inference.

    Returns:
        chunks: list of chunk texts (for embedding/FAISS).
        index:  DataFrame indexed by chunk_id with columns:
                ['doc_id','tok_start','tok_end','char_start','char_end', 'chunk_text?'].
    """
    chunks: List[str] = []
    records: List[Dict] = []
    chunker = sliding_window_chunker if mode == "sliding" else semantic_window_chunker

    for doc_id, text in enumerate(docs):
        if not text:
            continue
        recs = chunker(
            text,
            max_tokens=max_tokens,
            stride=stride,
            tokenizer_name=tokenizer_name,
            **({} if mode == "sliding" else {"min_tokens": min_tokens}),
        )
        for r in recs:
            cid = len(chunks)
            chunks.append(r.text)
            row = {
                "chunk_id": cid,
                "doc_id": doc_id,
                "tok_start": r.tok_start,
                "tok_end": r.tok_end,
                "char_start": r.char_start,
                "char_end": r.char_end,
            }
            if store_chunk_text:
                row["chunk_text"] = r.text
            records.append(row)

    df = pd.DataFrame.from_records(records).set_index("chunk_id")
    return chunks, df

# ============================== Persistence =================================

def save_chunk_index(path, df: pd.DataFrame):
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_chunk_index(path):
    return pd.read_parquet(path)

__all__ = [
    "ChunkRecord",
    "sliding_window_chunker",
    "semantic_window_chunker",
    "build_inference_corpus",
    "chunk_context_with_alignment",
    "build_chunked_corpus",
    "save_chunk_index",
    "load_chunk_index",
]

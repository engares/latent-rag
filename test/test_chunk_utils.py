# test/test_chunk_utils.py
"""
Robust tests for tokenizer-based chunkers in utils.chunk_utils.

Run:
    PYTHONPATH=. pytest test/test_chunk_utils.py -q
"""
from __future__ import annotations

import string
from typing import List

import pytest
from transformers import AutoTokenizer

from utils.chunk_utils import (
    chunk_context_with_alignment,
    build_chunked_corpus,
    sliding_window_chunker,
    semantic_window_chunker,
    build_inference_corpus,
)

TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
PUNCT = set(".!?;:\n")


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_doc():
    """Minimal SQuAD-like example that always aligns cleanly."""
    context = (
        "The Alliance for Catholic Education (ACE) was founded at the "
        "University of Notre Dame in 1994. Its aim is to strengthen and "
        "sustain Catholic schools in the United States."
    )
    answer = "Alliance for Catholic Education"
    start = context.index(answer)
    end = start + len(answer)
    return context, answer, start, end


@pytest.fixture(scope="module")
def long_text():
    """A multi-sentence paragraph to exercise sliding/semantic chunkers."""
    return (
        "RAG systems retrieve passages to ground answers in external knowledge. "
        "However, naive retrieval can surface redundant or off-topic text. "
        "Chunking strategies mitigate this by constraining window size and overlap. "
        "Semantic chunking further tries to end on punctuation boundaries. "
        "This improves readability and can help downstream generation."
    )


# --------------------------------------------------------------------------- #
# Answer-aware chunker (training)                                             #
# --------------------------------------------------------------------------- #
def test_chunk_contains_answer(sample_doc):
    """At least one produced chunk must (case-insensitively) include the answer."""
    ctx, ans, s, e = sample_doc
    chunks = chunk_context_with_alignment(
        ctx,
        s,
        e,
        max_tokens=20,
        stride=10,
        tokenizer_name=TOKENIZER,
    )
    assert any(ans.lower() in c.lower() for c in chunks)


def test_central_chunk_is_first_and_within_budget(sample_doc):
    """
    The first returned chunk is the centred one. Even with a tiny token budget
    it must still cover the answer span and respect *max_tokens*.
    """
    ctx, ans, s, e = sample_doc
    max_toks = 8
    chunks = chunk_context_with_alignment(
        ctx,
        s,
        e,
        max_tokens=max_toks,
        stride=4,
        tokens_before=1,
        tokens_after=1,
        tokenizer_name=TOKENIZER,
    )

    # 1) Centred chunk should contain the answer (case-insensitive)
    assert ans.lower().split()[0] in chunks[0].lower()

    # 2) Token length check (no special tokens are added)
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    n_tokens = len(tok(chunks[0], add_special_tokens=False)["input_ids"])
    assert n_tokens <= max_toks


def test_span_alignment_fallback():
    """
    If the char-span cannot be aligned (e.g. bogus indices) the full context
    should be returned as a *single* chunk.
    """
    ctx = "ABC DEF GHI"
    # Force failure: use an out-of-range character span
    answer_start = 100
    answer_end = 110

    chunks = chunk_context_with_alignment(
        ctx,
        answer_start,
        answer_end,
        max_tokens=4,
        stride=2,
        tokenizer_name=TOKENIZER,
    )
    assert chunks == [ctx]


def test_build_chunked_corpus_flag(sample_doc):
    """`contains_answer` flag in the returned DataFrame must be correct."""
    ctx, ans, s, e = sample_doc
    squad_like = [
        {
            "context": ctx,
            "answers": {"text": [ans], "answer_start": [s]},
        }
    ]

    chunks, index = build_chunked_corpus(
        squad_like,
        max_tokens=20,
        stride=10,
        tokenizer_name=TOKENIZER,
    )

    # At least one chunk for doc_id=0 must have contains_answer == True
    assert index.loc[index["doc_id"] == 0, "contains_answer"].any()


# --------------------------------------------------------------------------- #
# Inference chunkers (sliding / semantic)                                     #
# --------------------------------------------------------------------------- #
def _count_tokens(text: str, tok) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def test_sliding_window_chunker_respects_budget_and_stride(long_text):
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    max_tokens, stride = 24, 12

    recs = sliding_window_chunker(
        long_text,
        max_tokens=max_tokens,
        stride=stride,
        tokenizer_name=TOKENIZER,
    )

    assert len(recs) >= 2, "Should produce multiple chunks for a long paragraph."

    # 1) Each chunk within token budget
    for r in recs:
        assert _count_tokens(r.text, tok) <= max_tokens
        # token span length consistent
        assert (r.tok_end - r.tok_start + 1) <= max_tokens

    # 2) Successive starts advance by 'stride' (except possibly the last window)
    for a, b in zip(recs, recs[1:]):
        assert b.tok_start - a.tok_start in {stride, max_tokens}, "Unexpected stride step"


def test_semantic_chunker_prefers_punctuation_boundaries(long_text):
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    max_tokens, stride, min_tokens = 28, 14, 10

    recs = semantic_window_chunker(
        long_text,
        max_tokens=max_tokens,
        stride=stride,
        min_tokens=min_tokens,
        tokenizer_name=TOKENIZER,
        boundary_chars=".!?;:\n",
    )

    assert len(recs) >= 2

    # 1) Budget respected
    for r in recs:
        assert _count_tokens(r.text, tok) <= max_tokens

    # 2) At least one non-final chunk ends exactly at a punctuation boundary
    ended_on_punct = 0
    for i, r in enumerate(recs[:-1]):
        # character at char_end - 1 is the last character covered by the span
        ch = long_text[r.char_end - 1] if r.char_end - 1 < len(long_text) else ""
        if ch in PUNCT:
            ended_on_punct += 1
    assert ended_on_punct >= 1, "Semantic chunker should snap to punctuation at least once."


def test_build_inference_corpus_metadata_consistency(long_text):
    docs: List[str] = [long_text, long_text + " Extra sentence."]
    max_tokens, stride = 16, 8

    chunks, idx = build_inference_corpus(
        docs,
        mode="sliding",
        max_tokens=max_tokens,
        stride=stride,
        tokenizer_name=TOKENIZER,
        store_chunk_text=True,
    )

    # Shapes match
    assert len(chunks) == len(idx)

    # doc_id is valid, spans are in range, and chunk_text is present
    for cid, row in idx.iterrows():
        doc_id = int(row["doc_id"])
        assert 0 <= doc_id < len(docs)
        assert row["tok_start"] <= row["tok_end"]
        assert 0 <= row["char_start"] < row["char_end"] <= len(docs[doc_id])
        assert isinstance(row.get("chunk_text", ""), str) and row["chunk_text"].strip() != ""

    # Token-length budget for some random chunks
    tok = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    sample_ids = list(idx.index)[: min(5, len(idx))]
    for cid in sample_ids:
        text = idx.loc[cid, "chunk_text"]
        assert _count_tokens(text, tok) <= max_tokens

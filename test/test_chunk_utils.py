# test/test_chunk_utils.py
"""
Robust tests for the tokenizer-based answer-aware chunker in utils.chunk_utils.

Run:
    PYTHONPATH=. pytest test/test_chunk_utils.py -q
"""
from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from utils.chunk_utils import (
    chunk_context_with_alignment,
    build_chunked_corpus,
)

TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"


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


# --------------------------------------------------------------------------- #
# Unit tests                                                                  #
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
    # Forzar fallo: usar un rango de caracteres fuera del contexto
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

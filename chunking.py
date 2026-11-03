"""Sentence-aware sliding-window chunking.

Same hybrid strategy as before (sentence boundaries, token budget, overlap),
but punkt is resolved lazily and falls back to a regex splitter, so importing
this module never triggers a network download.
"""
from __future__ import annotations

import re
from typing import Callable, List

import config

_splitter: Callable[[str], List[str]] | None = None

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def _regex_split(text: str) -> List[str]:
    return [s for s in _SENTENCE_RE.split(text) if s and s.strip()]


def _get_splitter() -> Callable[[str], List[str]]:
    """Return a sentence splitter, preferring NLTK punkt when available."""
    global _splitter
    if _splitter is not None:
        return _splitter
    try:
        import nltk
        from nltk.tokenize import sent_tokenize

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
        sent_tokenize("Probe sentence. Another one.")
        _splitter = sent_tokenize
    except Exception:
        _splitter = _regex_split
    return _splitter


def chunk_text_hybrid(
    text: str,
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """Split text into overlapping, sentence-aligned chunks.

    A single sentence longer than `max_tokens` is hard-split on word
    boundaries rather than emitted oversized, which the previous
    implementation did silently.
    """
    max_tokens = max_tokens or config.CHUNK_MAX_TOKENS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    if not text or not text.strip():
        return []

    split = _get_splitter()
    try:
        sentences = split(text)
    except Exception:
        sentences = _regex_split(text)

    # Pre-split any sentence that alone blows the budget.
    units: List[str] = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_tokens:
            units.append(s)
        else:
            for i in range(0, len(words), max_tokens):
                units.append(" ".join(words[i : i + max_tokens]))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in units:
        token_len = len(sentence.split())
        if current and current_len + token_len > max_tokens:
            chunks.append(" ".join(current))
            # Slide the window back by ~`overlap` tokens of trailing context.
            tail: List[str] = []
            tail_len = 0
            while current and tail_len < overlap:
                last = current.pop()
                tail.insert(0, last)
                tail_len += len(last.split())
            current = tail + [sentence]
            current_len = sum(len(s.split()) for s in current)
        else:
            current.append(sentence)
            current_len += token_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if len(c.strip()) >= config.MIN_CHUNK_CHARS]

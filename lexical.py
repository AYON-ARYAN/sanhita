"""BM25 lexical index over the indexed chunks.

Dense retrieval alone has a specific, reproducible failure on this corpus.
Asked "What is the punishment for theft under the Indian Penal Code?", the
phrase "under the Indian Penal Code" dominates the embedding: every passage
that contains that phrase literally outranks the one that actually answers
the question, which is phrased "What is the punishment for 379 Theft? answer:
Imprisonment for 3 years, or fine, or both." The correct chunk did not appear
in the top 100. Rephrased with "379" in the query it ranks first at 0.895.

BM25 scores on term overlap and rewards rare terms, so it recovers exactly
this case. Fusing the two rankings gives a retriever that is robust to how
the question happens to be worded.

The index is built once from the text already stored in Chroma and cached to
disk; rebuilding is cheap enough to do whenever the chunk count changes.
"""
from __future__ import annotations

import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import config

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on",
    "by", "with", "as", "at", "it", "that", "this", "be", "was", "were", "which",
    "what", "who", "whom", "under", "shall", "any", "such", "from", "if", "not",
    "question", "answer", "instruction", "response",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall((text or "").lower())
            if t not in _STOP and len(t) > 1]


class BM25:
    """Classic Okapi BM25 over an in-memory inverted index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: List[int] = []
        self.ids: List[str] = []
        self.avgdl = 0.0

    def add_many(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        for doc_id, text in zip(ids, texts):
            idx = len(self.ids)
            self.ids.append(doc_id)
            tf = Counter(tokenize(text))
            self.doc_len.append(sum(tf.values()))
            for term, n in tf.items():
                self.postings[term].append((idx, n))
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.ids:
            return []
        N = len(self.ids)
        scores: Dict[int, float] = defaultdict(float)
        for term in set(terms):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            for idx, tf in plist:
                dl = self.doc_len[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.ids[i], s) for i, s in ranked]

    # ------------------------------------------------------------ cache ---
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"k1": self.k1, "b": self.b, "postings": dict(self.postings),
                         "doc_len": self.doc_len, "ids": self.ids,
                         "avgdl": self.avgdl}, fh, protocol=4)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "BM25 | None":
        if not path.is_file():
            return None
        try:
            with path.open("rb") as fh:
                d = pickle.load(fh)
            o = cls(d["k1"], d["b"])
            o.postings = defaultdict(list, d["postings"])
            o.doc_len, o.ids, o.avgdl = d["doc_len"], d["ids"], d["avgdl"]
            return o
        except Exception:
            return None


def build_from_collection(collection, batch: int = 10000, verbose: bool = True) -> BM25:
    """Build BM25 from the documents already stored in Chroma."""
    bm = BM25()
    n = collection.count()
    for off in range(0, n, batch):
        got = collection.get(limit=batch, offset=off, include=["documents"])
        bm.add_many(got["ids"], got["documents"])
        if verbose:
            print(f"  BM25 indexed {min(off+batch, n)}/{n}", flush=True)
    return bm


def get(collection, path: str | None = None, rebuild: bool = False) -> BM25 | None:
    """Load the cached BM25 index, rebuilding if stale or missing."""
    p = Path(path or config.BM25_PATH)
    if not rebuild:
        cached = BM25.load(p)
        # Stale if the collection grew: the index is still being built.
        if cached and abs(len(cached.ids) - collection.count()) <= config.BM25_STALE_TOLERANCE:
            return cached
    bm = build_from_collection(collection)
    bm.save(p)
    return bm

"""Chroma retrieval over Ollama embeddings.

Replaces index_loader.py and removes llama_index from the query path.
Previously the index was *written* with a raw SentenceTransformer and *read*
through llama_index's HuggingFaceEmbedding — two stacks over one collection,
which drift apart the moment either model name changes. Now both ingestion
and retrieval call `ollama_client`, so there is a single embedding path.

Also exposes `EphemeralRetriever` for the upload flow, so uploaded documents
and the persistent corpus share the same retrieval code.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import chromadb
from chromadb.config import Settings

import config
import lexical
import ollama_client as oc


@dataclass
class RetrievedChunk:
    """Mirrors the attribute surface the app previously got from llama_index."""
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)

    # llama_index compatibility
    def get_content(self) -> str:
        return self.text

    @property
    def source(self) -> str:
        return self.metadata.get("source_file", "Unknown Source")


def _to_chunks(res) -> List[RetrievedChunk]:
    """Convert a Chroma query result into RetrievedChunks.

    Chroma reports cosine *distance*; similarity is 1 - distance.
    """
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    out: List[RetrievedChunk] = []
    for i, text in enumerate(docs):
        dist = dists[i] if i < len(dists) else None
        score = (1.0 - dist) if dist is not None else 0.0
        out.append(RetrievedChunk(text=text,
                                  score=score,
                                  metadata=(metas[i] if i < len(metas) else {}) or {}))
    return out


class Retriever:
    """Persistent-corpus retriever."""

    def __init__(self, path: str | None = None, collection: str | None = None,
                 top_k: int | None = None):
        self.path = path or config.CHROMA_PATH
        self.collection_name = collection or config.COLLECTION
        self.top_k = top_k or config.TOP_K
        self._client = chromadb.PersistentClient(path=self.path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embed_model": config.EMBED_MODEL},
        )

    def count(self) -> int:
        return self._collection.count()

    # --- lexical half of the hybrid, built lazily and cached ---------------
    _bm25 = None
    _bm25_tried = False

    def _lexical(self):
        if not config.HYBRID_SEARCH:
            return None
        if not Retriever._bm25_tried:
            Retriever._bm25_tried = True
            try:
                Retriever._bm25 = lexical.get(self._collection)
            except Exception as exc:  # noqa: BLE001
                print(f"[retriever] BM25 unavailable, dense only: {exc}")
                Retriever._bm25 = None
        return Retriever._bm25

    def _dense(self, query: str, n: int) -> List[RetrievedChunk]:
        vec = oc.embed_query(query)
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=min(n, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return _to_chunks(res)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        """Hybrid retrieval: dense + BM25, fused by Reciprocal Rank Fusion.

        Dense search alone has a reproducible failure here. For "What is the
        punishment for theft under the Indian Penal Code?" the phrase "under
        the Indian Penal Code" dominates the embedding, so every chunk
        containing it literally outranks the one that answers the question
        ("What is the punishment for 379 Theft? ... Imprisonment for 3 years")
        -- which did not appear in the top 100. BM25 ranks that chunk highly on
        term overlap, and RRF combines the two orderings without needing the
        two scores to be on a comparable scale.
        """
        k = top_k or self.top_k
        total = self.count()
        if total == 0:
            return []

        dense = self._dense(query, max(config.DENSE_CANDIDATES, k))
        bm = self._lexical()
        if bm is None:
            return dense[:k]

        lex = bm.search(query, top_k=config.BM25_CANDIDATES)
        if not lex:
            return dense[:k]

        RRF = config.RRF_K
        fused: Dict[str, float] = {}
        by_text: Dict[str, RetrievedChunk] = {}

        for rank, ch in enumerate(dense, 1):
            key = ch.text
            by_text[key] = ch
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF + rank)

        # Pull the BM25 winners' text and metadata out of Chroma in one call.
        lex_ids = [i for i, _ in lex]
        try:
            got = self._collection.get(ids=lex_ids, include=["documents", "metadatas"])
        except Exception:
            return dense[:k]
        pos = {cid: i for i, cid in enumerate(got.get("ids") or [])}
        for rank, (cid, _score) in enumerate(lex, 1):
            i = pos.get(cid)
            if i is None:
                continue
            text = got["documents"][i]
            meta = (got["metadatas"][i] or {})
            key = text
            if key not in by_text:
                by_text[key] = RetrievedChunk(text=text, score=0.0, metadata=meta)
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF + rank)

        order = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out: List[RetrievedChunk] = []
        for key, score in order:
            ch = by_text[key]
            # Keep the cosine score when we have one; it is what the UI shows.
            out.append(RetrievedChunk(text=ch.text,
                                      score=ch.score if ch.score else round(score, 4),
                                      metadata=ch.metadata))
        return out


class EphemeralRetriever:
    """In-memory retriever for per-request uploaded documents.

    Nothing is persisted. The collection name is unique per instance because
    chromadb caches clients by Settings: `EphemeralClient()` hands back the
    SAME in-process client on every call, so a fixed name collides with
    "Collection [...] already exists" on the second request. Call `close()`
    (or use it as a context manager) to drop the collection afterwards,
    otherwise collections accumulate in the shared client for the process
    lifetime.
    """

    def __init__(self, name: str | None = None):
        self._client = chromadb.EphemeralClient(
            Settings(anonymized_telemetry=False, allow_reset=True)
        )
        self._name = name or f"upload_{uuid.uuid4().hex}"
        self._collection = self._client.create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )
        self._n = 0

    def close(self) -> None:
        """Drop the scratch collection. Safe to call more than once."""
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass

    def __enter__(self) -> "EphemeralRetriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, texts: Sequence[str], metadatas: Sequence[Dict]) -> None:
        if not texts:
            return
        for i in range(0, len(texts), config.EMBED_BATCH):
            bt = list(texts[i : i + config.EMBED_BATCH])
            bm = list(metadatas[i : i + config.EMBED_BATCH])
            vectors = oc.embed_documents(bt)
            self._collection.add(
                documents=bt,
                embeddings=vectors,
                metadatas=bm,
                ids=[f"up_{self._n + j}" for j in range(len(bt))],
            )
            self._n += len(bt)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        if self._n == 0:
            return []
        k = min(top_k or config.UPLOAD_TOP_K, self._n)
        vec = oc.embed_query(query)
        res = self._collection.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return _to_chunks(res)


# Module-level singleton used by app.py. Construction is cheap: it opens the
# Chroma collection but loads no embedding model into this process.
retriever = Retriever()

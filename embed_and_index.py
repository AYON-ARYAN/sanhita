"""Build the Chroma index using Ollama embeddings.

Replaces the SentenceTransformer pipeline. Key properties:

  * Embeddings come from Ollama (`config.EMBED_MODEL`), the same path the
    retriever uses, so index and query can never drift vector spaces.
  * Chunk IDs are deterministic (sha1 of source path + record + chunk index),
    so re-running is an idempotent upsert rather than a duplicate insert.
  * The run is resumable: IDs already present in the collection are skipped,
    which matters because a full corpus build takes roughly an hour.
  * The embedding dimension is asserted against config before any write, so
    swapping the model cannot silently corrupt a collection.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from typing import Dict, List, Sequence

import chromadb

import config
import loaders
import ollama_client as oc
from chunking import chunk_text_hybrid  # re-exported for callers

__all__ = ["chunk_text_hybrid", "build", "get_collection"]


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
            print(f"Deleted existing collection '{config.COLLECTION}'")
        except Exception:
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": "cosine",
                  "embed_model": config.EMBED_MODEL},
    )


def _chunk_id(source_path: str, record_index: int | None, chunk_index: int) -> str:
    h = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    return f"{h}_r{record_index if record_index is not None else 0}_c{chunk_index}"


def _existing_ids(collection, ids: Sequence[str]) -> set[str]:
    """Which of these IDs are already indexed."""
    found: set[str] = set()
    step = 500
    for i in range(0, len(ids), step):
        batch = list(ids[i : i + step])
        try:
            got = collection.get(ids=batch, include=[])
            found.update(got.get("ids") or [])
        except Exception:
            pass
    return found


def build(
    dirs: Sequence[str] | None = None,
    reset: bool = False,
    resume: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
) -> None:
    t_start = time.time()

    oc.preflight(require_llm=False, require_embed=True)
    print(f"Ollama {oc.server_version()} | embed model: {config.EMBED_MODEL}")

    # --- verify embedding dimension before touching the collection --------
    probe = oc.embed_documents(["dimension probe"])[0]
    if len(probe) != config.EMBED_DIM:
        raise SystemExit(
            f"Embedding dim mismatch: {config.EMBED_MODEL} returns {len(probe)}, "
            f"config.EMBED_DIM is {config.EMBED_DIM}. Set EMBED_DIM={len(probe)} "
            f"and use a fresh COLLECTION name."
        )
    print(f"Embedding dimension verified: {len(probe)}")

    # --- extract ----------------------------------------------------------
    print("\n=== Extracting documents via Apache Tika ===")
    docs, report = loaders.load_documents(dirs=dirs, limit=limit)
    print("\n" + report.summary())

    # --- chunk ------------------------------------------------------------
    print("\n=== Chunking ===")
    texts: List[str] = []
    ids: List[str] = []
    metas: List[Dict] = []
    for d in docs:
        for ci, chunk in enumerate(chunk_text_hybrid(d.text)):
            texts.append(chunk)
            ids.append(_chunk_id(d.source_path, d.record_index, ci))
            metas.append(
                {
                    "source_file": d.source_file,
                    "source_path": d.source_path,
                    "chunk_index": ci,
                    "record_index": d.record_index if d.record_index is not None else -1,
                }
            )
    print(f"Chunks produced: {len(texts)}")

    # Deterministic IDs collide only if two docs share a path+record+chunk,
    # which would mean duplicate input. Drop dupes rather than upserting over
    # ourselves mid-run.
    seen: set[str] = set()
    keep = [i for i, cid in enumerate(ids) if not (cid in seen or seen.add(cid))]
    if len(keep) != len(ids):
        print(f"Dropped {len(ids) - len(keep)} duplicate chunk IDs")
        texts = [texts[i] for i in keep]
        ids = [ids[i] for i in keep]
        metas = [metas[i] for i in keep]

    if dry_run:
        est = len(texts) / 51.0
        print(f"\n[dry-run] would embed {len(texts)} chunks "
              f"(~{est/60:.1f} min at ~51 chunks/s). No writes made.")
        return

    collection = get_collection(reset=reset)

    if resume and not reset:
        already = _existing_ids(collection, ids)
        if already:
            print(f"Resume: {len(already)} chunks already indexed, skipping them")
            keep = [i for i, cid in enumerate(ids) if cid not in already]
            texts = [texts[i] for i in keep]
            ids = [ids[i] for i in keep]
            metas = [metas[i] for i in keep]

    total = len(texts)
    if total == 0:
        print("Nothing to embed. Index is up to date.")
        print(f"Collection '{config.COLLECTION}' count: {collection.count()}")
        return

    print(f"\n=== Embedding + indexing {total} chunks "
          f"(batch={config.EMBED_BATCH}) ===")
    done = 0
    t0 = time.time()
    for start in range(0, total, config.EMBED_BATCH):
        end = min(start + config.EMBED_BATCH, total)
        bt = texts[start:end]
        vectors = oc.embed_documents(bt)
        collection.upsert(
            documents=bt,
            embeddings=vectors,
            metadatas=metas[start:end],
            ids=ids[start:end],
        )
        done += len(bt)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate if rate else 0
        print(
            f"  {done}/{total}  {rate:5.1f} chunks/s  "
            f"elapsed {elapsed/60:5.1f}m  ETA {eta/60:5.1f}m",
            flush=True,
        )

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min")
    print(f"Collection '{config.COLLECTION}' now holds {collection.count()} chunks")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the Ollama-embedded legal index")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="data directories (default: config.DATA_DIRS)")
    ap.add_argument("--reset", action="store_true",
                    help="delete the collection first (full rebuild)")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not skip already-indexed chunk IDs")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N files (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract and chunk, report counts, write nothing")
    a = ap.parse_args(argv)
    build(dirs=a.dirs, reset=a.reset, resume=not a.no_resume,
          limit=a.limit, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

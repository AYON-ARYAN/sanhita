"""Build the knowledge graph over the corpus.

    python build_graph.py                 # deterministic, whole corpus (~minutes)
    python build_graph.py --dirs data/pdfs
    python build_graph.py --llm --llm-chunks 20   # opt-in LLM enrichment
    python build_graph.py --reset         # discard the existing graph first
    python build_graph.py --dry-run       # report counts, write nothing

Why two extractors: LLM triple extraction costs ~100s per 500-token chunk on
this machine, so the full 121k-chunk corpus would take ~141 days. The
deterministic legal extractor runs at ~600 chunks/s, covering the whole corpus
in minutes. `--llm` adds model-extracted triples on the first N chunks of each
document, where the richer relations are worth the wait.
"""
from __future__ import annotations

import argparse
import time
from typing import Sequence

import config
import graph_rag
import legal_entities as le
import loaders
from chunking import chunk_text_hybrid
from graph_store import GraphStore


def build(dirs: Sequence[str] | None = None, reset: bool = False,
          use_llm: bool = False, llm_chunks: int = 4,
          limit: int | None = None, dry_run: bool = False) -> None:
    t0 = time.time()

    store = GraphStore(autoload=not reset)
    if reset:
        store.clear()
        print("Starting from an empty graph")
    else:
        print(f"Loaded graph: {store.graph.number_of_nodes()} nodes, "
              f"{store.graph.number_of_edges()} edges")

    print("\n=== Extracting documents via Apache Tika ===")
    docs, report = loaders.load_documents(dirs=dirs, limit=limit)
    print("\n" + report.summary())

    print(f"\n=== Extracting triples "
          f"({'deterministic + LLM' if use_llm else 'deterministic'}) ===")
    if use_llm:
        print(f"    LLM enrichment on the first {llm_chunks} chunks per document")

    total_chunks = 0
    total_triples = 0
    per_doc_llm = {}
    t_extract = time.time()

    for i, d in enumerate(docs, 1):
        chunks = chunk_text_hybrid(d.text)
        total_chunks += len(chunks)
        triples = []
        for ci, ch in enumerate(chunks):
            triples.extend(le.extract_triples(ch, source=d.source_file))
            if use_llm:
                used = per_doc_llm.get(d.source_path, 0)
                if used < llm_chunks:
                    triples.extend(graph_rag.llm_extract_triples(ch))
                    per_doc_llm[d.source_path] = used + 1
        total_triples += len(triples)
        if not dry_run:
            store.add_triples(triples, source=d.source_file)
            store.note_document(d.source_file)

        if i % 2000 == 0 or i == len(docs):
            el = time.time() - t_extract
            print(f"  [{i}/{len(docs)}] {total_chunks} chunks, "
                  f"{total_triples} triples, {total_chunks/max(el,1e-9):,.0f} chunks/s",
                  flush=True)

    if dry_run:
        print(f"\n[dry-run] {total_chunks} chunks -> {total_triples} triples. "
              "Nothing written.")
        return

    store.save()
    print(f"\nGraph saved to {store.path}")
    print(f"  nodes : {store.graph.number_of_nodes()}")
    print(f"  edges : {store.graph.number_of_edges()}")
    print(f"  types : {store.get_stats()['type_counts']}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build the legal knowledge graph")
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--llm", action="store_true",
                    help="also run LLM triple extraction (slow: ~100s/chunk)")
    ap.add_argument("--llm-chunks", type=int, default=config.GRAPH_LLM_MAX_CHUNKS,
                    help="max chunks per document to send to the LLM")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    build(dirs=a.dirs, reset=a.reset, use_llm=a.llm,
          llm_chunks=a.llm_chunks, limit=a.limit, dry_run=a.dry_run)


if __name__ == "__main__":
    main()

"""End-to-end check of the local RAG stack.

Verifies, in order: Ollama reachable -> models installed -> embeddings work
and have the right dimension -> Tika extracts -> index is populated ->
retrieval returns scored chunks -> Qwen generates grounded text.

Run:  .venv/bin/python smoke_test.py
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import sys
import time

import config
import ollama_client as oc
import tika_extract as tika
from chunking import chunk_text_hybrid
from retriever import Retriever

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    def deco(fn):
        def wrapped():
            t0 = time.time()
            try:
                detail = fn() or ""
                results.append((PASS, name, f"{detail} ({time.time()-t0:.1f}s)"))
                return True
            except Exception as exc:  # noqa: BLE001
                results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
                return False
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


@check("Ollama reachable")
def t_ollama():
    v = oc.server_version()
    assert v, f"no response from {config.OLLAMA_HOST}"
    return f"version {v}"


@check("Models installed")
def t_models():
    oc.preflight()
    return f"{config.LLM_MODEL} + {config.EMBED_MODEL}"


@check("Embeddings + dimension")
def t_embed():
    v = oc.embed_documents(["Section 378 of the Indian Penal Code defines theft."])[0]
    assert len(v) == config.EMBED_DIM, f"dim {len(v)} != {config.EMBED_DIM}"
    return f"dim {len(v)}"


@check("Query/document prefix asymmetry")
def t_prefix():
    # nomic-embed is asymmetric; the two prefixes must produce different vectors.
    d = oc.embed_documents(["theft is defined in section 378"])[0]
    q = oc.embed_query("theft is defined in section 378")
    assert d != q, "doc and query embeddings identical - prefixes not applied"
    return "doc != query vector, prefixes active"


@check("Tika extraction")
def t_tika():
    tika.start_server()
    text = tika.extract_text_from_bytes(
        b"Clause 1. The Tenant shall pay rent monthly.", "probe.txt"
    )
    assert "Tenant" in text, f"unexpected extraction: {text!r}"
    return f"{len(text)} chars"


@check("Chunking")
def t_chunk():
    cs = chunk_text_hybrid("Section 378 defines theft. " * 300)
    assert cs and all(len(c.split()) <= config.CHUNK_MAX_TOKENS for c in cs)
    return f"{len(cs)} chunks, max {max(len(c.split()) for c in cs)} tokens"


@check("Index populated")
def t_index():
    r = Retriever()
    n = r.count()
    assert n > 0, (
        f"collection '{config.COLLECTION}' is empty - "
        "run: python embed_and_index.py --reset"
    )
    return f"{n} chunks in '{config.COLLECTION}'"


@check("Retrieval returns scored chunks")
def t_retrieve():
    r = Retriever()
    hits = r.retrieve("What is the punishment for theft under the Indian Penal Code?")
    assert hits, "no hits returned"
    assert all(-1.0 <= h.score <= 1.0 for h in hits), "scores out of range"
    top = hits[0]
    return (f"{len(hits)} hits, top score {top.score:.3f} "
            f"from {top.metadata.get('source_file')}")


@check("Repeated ephemeral uploads")
def t_ephemeral():
    # Regression: chromadb caches EphemeralClient by Settings, so a fixed
    # collection name failed with "Collection already exists" on the 2nd
    # upload request. Each instance must get a unique name and clean up.
    from retriever import EphemeralRetriever
    for _ in range(3):
        with EphemeralRetriever() as er:
            er.add(["The Tenant shall pay rent monthly."],
                   [{"source_document": "probe.txt"}])
            hits = er.retrieve("when is rent due")
            assert hits, "no hits from ephemeral retriever"
    return "3 consecutive upload cycles OK"


@check("Legal entity extraction")
def t_legal_entities():
    import legal_entities as le
    text = ("In Solomon Ogbe vs Narcotics Control Bureau the High Court of Delhi "
            "considered Section 37 of the NDPS Act and Article 21 of the "
            "Constitution of India.")
    tr = le.extract_triples(text)
    assert tr, "no triples extracted"
    rels = {t["relation"] for t in tr}
    assert "party_to" in rels, f"missing party_to, got {rels}"
    assert any("Article 21" in t["head"] for t in tr), "Article 21 not captured"
    # Regression: articles must not be force-attributed to the Constitution.
    civil = le.extract_triples("Article 2011 of the Portuguese Civil Code provides")
    assert any("Portuguese Civil Code" in t["tail"] for t in civil), \
        f"civil-code article misattributed: {civil}"
    return f"{len(tr)} triples, relations={sorted(rels)}"


@check("Graph store round-trip")
def t_graph_store():
    import tempfile, os
    from graph_store import GraphStore
    path = os.path.join(tempfile.mkdtemp(), "g.json")
    a = GraphStore(path=path, autoload=False)
    a.add_triples([{"head": "Case X", "head_type": "Case", "relation": "invokes",
                    "tail": "Section 1 Act Y", "tail_type": "Statute"}], source="x")
    a.save()
    b = GraphStore(path=path)
    assert b.graph.number_of_nodes() == 2, b.graph.number_of_nodes()
    assert b.traverse(["Case X"]), "traversal returned nothing after reload"
    return "saved and reloaded with traversal intact"


@check("Graph populated + hub-aware traversal")
def t_graph_live():
    from graph_store import GraphStore
    gs = GraphStore()
    n = gs.graph.number_of_nodes()
    assert n > 0, "graph is empty - run: python build_graph.py"
    rels = gs.traverse(["Article 21"])
    assert rels, "no relations for Article 21"
    # Hub blocking: results must not be dominated by the Constitution hub.
    filler = sum(1 for r in rels if r.endswith("--> Constitution of India"))
    assert filler < len(rels), "traversal returned only hub filler"
    return f"{n} nodes, {len(rels)} relations, {len(rels)-filler} non-filler"


@check("Connected viz subgraph")
def t_graph_viz():
    from graph_store import GraphStore
    d = GraphStore().get_graph_data(max_nodes=120)
    assert d["nodes"], "no nodes returned"
    # Regression: top-N-by-degree alone yielded 120 nodes / 9 edges.
    assert len(d["edges"]) >= len(d["nodes"]), \
        f"subgraph too sparse to render: {len(d['nodes'])} nodes, {len(d['edges'])} edges"
    return f"{len(d['nodes'])} nodes, {len(d['edges'])} edges"


@check("Hybrid graph RAG retrieval")
def t_hybrid():
    import graph_rag
    g = graph_rag.GraphRAG()
    ctx = g.retrieve("Does Article 21 apply to prolonged incarceration under the NDPS Act?")
    assert ctx["query_entities"], "no query entities extracted"
    assert ctx["graph_relations"], "no graph relations"
    assert len(ctx["pipeline_steps"]) == 3, ctx["pipeline_steps"]
    return (f"{len(ctx['vector_results'])} vector hits, "
            f"{len(ctx['graph_relations'])} relations, "
            f"{len(ctx['highlighted_nodes'])} highlighted")


@check("Qwen generates grounded text")
def t_generate():
    r = Retriever()
    hits = r.retrieve("What is theft under the Indian Penal Code?")
    context = "\n\n".join(h.text for h in hits[:3])
    prompt = (
        "Answer using only the context.\n\n"
        f"Context:\n{context}\n\nQuestion: What is theft under the IPC?\nAnswer:"
    )
    out = "".join(oc.run_llm_query(prompt))
    assert out.strip(), "empty generation"
    assert "[LLM ERROR]" not in out, out[:200]
    return f"{len(out)} chars: {' '.join(out.split())[:90]}..."


def main() -> int:
    for fn in (t_ollama, t_models, t_embed, t_prefix, t_tika,
               t_chunk, t_index, t_retrieve, t_ephemeral,
               t_legal_entities, t_graph_store, t_graph_live, t_graph_viz,
               t_hybrid, t_generate):
        fn()

    print("\n" + "=" * 78)
    for status, name, detail in results:
        mark = "✅" if status == PASS else "❌"
        print(f"{mark} {status}  {name:38s} {detail}")
    print("=" * 78)
    failed = [r for r in results if r[0] == FAIL]
    print(f"{len(results)-len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

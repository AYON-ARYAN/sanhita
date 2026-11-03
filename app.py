"""Flask app for the legal RAG assistant.

Stack: Qwen (via Ollama) for generation, nomic-embed-text (via Ollama) for
embeddings, Apache Tika for document extraction, Chroma for vector search.

The HTTP contract is unchanged from the previous version so the existing
frontend keeps working:
  POST /query              form: query                     -> streamed text/plain
  POST /upload_and_query   form: query, document[]         -> streamed text/plain
  guard rejections         -> 422 JSON {error, is_guard_error}
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   send_file, stream_with_context, url_for)
from werkzeug.utils import secure_filename

import config
import drafting
import exporters
import graph_rag
import ollama_client as oc
import tika_extract as tika
from chunking import chunk_text_hybrid
from graph_store import GraphStore
from retriever import EphemeralRetriever, retriever

app = Flask(__name__)

ENABLE_GUARD = os.getenv("ENABLE_GUARD", "0") == "1"

# Hybrid engine: Chroma vectors + persistent knowledge graph + Qwen.
graphrag = graph_rag.GraphRAG(retriever=retriever)
ALLOWED_UPLOAD_EXT = sorted(tika.TIKA_EXTENSIONS)


# ------------------------------------------------------------------ guard ---
# The previous implementation tested `"LEGAL" in classification`, which is
# True for BOTH "LEGAL" and "NON-LEGAL" because the latter contains the
# former as a substring -- so the guard never actually blocked anything.
# The labels here are deliberately "LEGAL" / "NONLEGAL" (no hyphen) so that
# substring trap cannot come back.
#
# A bare instruction also misclassifies at 3B: qwen2.5:3b-instruct labelled
# "What does the Constitution say about the territory of the Union of India?"
# as non-legal. The few-shot form below scores 9/9 on that probe set.
GUARD_SYSTEM = (
    "You classify whether text concerns law. Answer with exactly one token: "
    "LEGAL or NONLEGAL. Treat constitutions, statutes, contracts, courts, "
    "rights, crimes, regulations, property, taxation and civic governance "
    "as LEGAL."
)

GUARD_PROMPT = """Classify each text as LEGAL or NONLEGAL.

Text: What is the punishment for theft?
Answer: LEGAL
Text: How do I bake sourdough bread?
Answer: NONLEGAL
Text: Which article of the Constitution defines the territory of the Union?
Answer: LEGAL
Text: What is the best football team?
Answer: NONLEGAL
Text: Explain the termination clause in this lease agreement.
Answer: LEGAL
Text: How do I fix a leaking tap?
Answer: NONLEGAL
Text: {snippet}
Answer:"""


def is_legal_query(query_text: str, document_texts: list[str] | None = None) -> bool:
    """Classify whether the request is legal in nature, using the LLM.

    Fails open: if classification errors, the request is allowed through.
    """
    if not ENABLE_GUARD:
        return True

    snippet = query_text
    if document_texts:
        snippet += "\n\n" + document_texts[0][:500]

    try:
        verdict = oc.complete(
            GUARD_PROMPT.format(snippet=snippet), system=GUARD_SYSTEM
        ).strip().upper()
        token = verdict.split()[0] if verdict.split() else ""
        allowed = not token.startswith("NON")
        app.logger.info("Guard: %r -> %s", token[:16], "allow" if allowed else "block")
        return allowed
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("Guard classification failed, allowing: %s", exc)
        return True


# ----------------------------------------------------------------- prompts ---
def build_prompt(user_query: str, chunks) -> str:
    """Prompt for corpus-backed queries.

    Each chunk is labelled with its source file so the answer is traceable
    and the model can cite. The old version dropped this metadata.
    """
    context = "\n\n".join(
        f"[{i+1}] (source: {getattr(c, 'source', 'Unknown Source')})\n"
        f"{getattr(c, 'text', str(c))}"
        for i, c in enumerate(chunks)
    )
    return (
        "You are a legal assistant. Use the following context to answer the "
        "question. Use Markdown for formatting, such as bolding key terms "
        "(**text**). Structure the answer with paragraphs or line breaks so it "
        "is readable. If the context does not contain the answer, say so "
        "plainly rather than inventing one.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\nAnswer:"
    )


def build_single_doc_prompt(user_query: str, source: str, chunks: list[str],
                            graph_relations: list[str] | None = None) -> str:
    """Prompt for ONE uploaded document.

    The per-document structure is produced by code (see `upload_and_query`),
    not by asking the model to emit headings. qwen2.5:3b-instruct ignores that
    instruction even when told to write exactly N verbatim headings: it merges
    the documents into a single answer. Generating once per document makes the
    contract deterministic at the cost of one LLM call per file.
    """
    context = "\n\n".join(chunks)
    graph_block = ""
    if graph_relations:
        rels = "\n".join(graph_relations)
        graph_block = (
            "\n--- ENTITY RELATIONSHIPS EXTRACTED FROM THIS DOCUMENT ---\n"
            f"{rels}\n"
        )
    return (
        "You are a legal assistant. Answer the question using ONLY the context "
        f"below, which comes from the single document '{source}'. "
        "Use Markdown for emphasis but do NOT write any heading - the heading "
        "is added for you. If this context does not address the question, say "
        "so in one short sentence and nothing more.\n\n"
        f"---\nCONTEXT:\n{context}\n{graph_block}---\n\n"
        f"Question: {user_query}\nAnswer:"
    )


# ------------------------------------------------------------------ routes ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/explorer", methods=["GET"])
def explorer():
    """Kept as a bookmarkable alias. The graph lives in the main UI now."""
    return redirect(url_for("index") + "#evidence")


@app.route("/health", methods=["GET"])
def health():
    """Report the state of every backing service."""
    ollama_ver = oc.server_version()
    models = oc.installed_models()
    return jsonify(
        {
            "ollama": {
                "host": config.OLLAMA_HOST,
                "version": ollama_ver,
                "reachable": ollama_ver is not None,
                "llm_model": config.LLM_MODEL,
                "llm_installed": oc._model_present(config.LLM_MODEL, models),
                "embed_model": config.EMBED_MODEL,
                "embed_installed": oc._model_present(config.EMBED_MODEL, models),
                "installed": models,
            },
            "tika": {"url": config.TIKA_URL, "reachable": tika.is_up()},
            "chroma": {
                "path": config.CHROMA_PATH,
                "collection": config.COLLECTION,
                "chunks": retriever.count(),
            },
            "graph": {
                "path": config.GRAPH_PATH,
                "enabled": config.GRAPH_ENABLED,
                "nodes": graphrag.store.graph.number_of_nodes(),
                "edges": graphrag.store.graph.number_of_edges(),
                "llm_extraction": config.GRAPH_LLM_EXTRACT,
            },
        }
    )


@app.route("/query", methods=["POST"])
def query():
    """Answer from the indexed corpus."""
    user_query = (request.form.get("query") or "").strip()
    if not user_query:
        return "No query provided.", 400

    if not is_legal_query(user_query):
        return (
            jsonify(
                {
                    "error": "I am a legal assistant and can only process "
                             "queries related to legal matters.",
                    "is_guard_error": True,
                }
            ),
            422,
        )

    if retriever.count() == 0:
        return (
            jsonify(
                {
                    "error": "The document index is empty. Build it with: "
                             "python embed_and_index.py",
                    "is_guard_error": True,
                }
            ),
            422,
        )

    # Hybrid retrieval: vector hits + knowledge-graph relations.
    ctx = graphrag.retrieve(user_query)
    app.logger.info(
        "Query: %d vector hits, %d graph relations, entities=%s",
        len(ctx["vector_results"]), len(ctx["graph_relations"]),
        ctx["query_entities"])

    def generate():
        yield from graphrag.stream(user_query, ctx)

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/upload_and_query", methods=["POST"])
def upload_and_query():
    """Answer from documents uploaded with this request."""
    files = request.files.getlist("document")
    user_query = (request.form.get("query") or "").strip()

    if not files or all(f.filename == "" for f in files):
        return "No files selected.", 400
    if not user_query:
        return "No query provided.", 400

    try:
        tika.start_server()
    except Exception as exc:  # noqa: BLE001
        return f"Document extraction service unavailable: {exc}", 503

    document_texts: list[str] = []
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    skipped: list[str] = []

    for file in files:
        if not file or not file.filename:
            continue
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in tika.TIKA_EXTENSIONS and ext not in tika.STRUCTURED_EXTENSIONS:
            skipped.append(filename)
            app.logger.info("Skipping unsupported upload: %s", filename)
            continue

        data = file.read()
        if not data:
            skipped.append(filename)
            continue

        try:
            text = tika.extract_text_from_bytes(data, filename=filename)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Tika failed on %s: %s", filename, exc)
            skipped.append(filename)
            continue

        if not text.strip():
            skipped.append(filename)
            continue

        document_texts.append(text)
        for chunk in chunk_text_hybrid(text):
            all_chunks.append(chunk)
            all_metadatas.append({"source_document": filename})

    if not is_legal_query(user_query, document_texts):
        return (
            jsonify(
                {
                    "error": "I am a legal assistant and can only process "
                             "documents and queries related to legal matters.",
                    "is_guard_error": True,
                }
            ),
            422,
        )

    if not all_chunks:
        detail = f" Skipped: {', '.join(skipped)}." if skipped else ""
        return f"Failed to extract any text from the provided files.{detail}", 400

    # A throwaway graph over just these uploads. Deterministic extraction is
    # instant (~600 chunks/s), so this costs nothing noticeable; LLM
    # extraction would add ~100s per chunk and is left to GRAPH_LLM_EXTRACT.
    upload_graph = GraphStore(autoload=False)
    chunks_by_source: dict[str, list[str]] = defaultdict(list)
    for chunk, meta in zip(all_chunks, all_metadatas):
        chunks_by_source[meta["source_document"]].append(chunk)
    for src, cs in chunks_by_source.items():
        upload_graph.add_triples(
            [t for c in cs for t in graph_rag.extract_triples(c, use_llm=False)],
            source=src)

    query_entities = graph_rag.extract_query_entities(user_query)
    graph_rels = (upload_graph.traverse(query_entities)
                  if config.GRAPH_ENABLED else [])
    app.logger.info("Upload graph: %d nodes, %d edges, %d relations for query",
                    upload_graph.graph.number_of_nodes(),
                    upload_graph.graph.number_of_edges(), len(graph_rels))

    with EphemeralRetriever() as scratch:
        scratch.add(all_chunks, all_metadatas)
        hits = scratch.retrieve(user_query, top_k=config.UPLOAD_TOP_K)

    grouped: dict[str, list[str]] = defaultdict(list)
    for hit in hits:
        grouped[hit.metadata.get("source_document", "Unknown Source")].append(hit.text)

    # Preserve the order the files arrived in, not retrieval order.
    ordered = [m["source_document"] for m in all_metadatas]
    seen: set[str] = set()
    sources = [x for x in ordered
               if x in grouped and not (x in seen or seen.add(x))]

    def generate():
        for i, source in enumerate(sources):
            if i:
                yield "\n\n"
            yield f"### From {source}\n\n"
            yield from oc.run_llm_query(
                build_single_doc_prompt(user_query, source, grouped[source],
                                        graph_relations=graph_rels)
            )

    return Response(stream_with_context(generate()), mimetype="text/plain")


# ------------------------------------------------------------- chat stream ---
def _ndjson(obj) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.route("/chat", methods=["POST"])
def chat():
    """Single streaming endpoint for the UI: answer plus evidence, one request.

    Streams newline-delimited JSON so the client gets the retrieval evidence
    immediately and the answer token by token:

        {"type":"meta",    ...}            once, before generation
        {"type":"section", "source":"..."} per document, uploads only
        {"type":"token",   "t":"..."}      many
        {"type":"done"}                    once
        {"type":"error",   "message":"..."}

    The older text/plain /query and JSON /graph_query routes still work; this
    exists because the UI needs citations and graph relations *with* the
    stream, and firing two requests would double generation cost.
    """
    user_query = (request.form.get("query") or "").strip()
    files = [f for f in request.files.getlist("document") if f and f.filename]

    if not user_query:
        return Response(_ndjson({"type": "error",
                                 "message": "Ask a question to get started."}),
                        mimetype="application/x-ndjson", status=400)

    # ---------- uploads: build per-document context up front ----------
    upload_payload = None
    if files:
        try:
            tika.start_server()
        except Exception as exc:  # noqa: BLE001
            return Response(_ndjson({"type": "error",
                                     "message": f"Document reader unavailable: {exc}"}),
                            mimetype="application/x-ndjson", status=503)
        upload_payload = _prepare_uploads(files, user_query)
        if upload_payload.get("error"):
            return Response(_ndjson({"type": "error",
                                     "message": upload_payload["error"]}),
                            mimetype="application/x-ndjson", status=400)

    if ENABLE_GUARD and not is_legal_query(
            user_query, upload_payload["texts"] if upload_payload else None):
        return Response(_ndjson({
            "type": "error",
            "message": "This assistant only answers questions about legal "
                       "matters, documents and proceedings.",
            "is_guard_error": True,
        }), mimetype="application/x-ndjson", status=422)

    def generate():
        try:
            if upload_payload:
                meta = {
                    "type": "meta", "mode": "upload",
                    "query_entities": upload_payload["entities"],
                    "graph_relations": upload_payload["relations"],
                    "vector_results": upload_payload["hits"],
                    "highlighted_nodes": upload_payload["highlighted"],
                    "documents": upload_payload["sources"],
                    "skipped": upload_payload["skipped"],
                    "pipeline": [
                        {"step": "Read documents",
                         "detail": f"{len(upload_payload['sources'])} file(s) via Apache Tika"},
                        {"step": "Embed + search",
                         "detail": f"{len(upload_payload['hits'])} passages matched"},
                        {"step": "Build document graph",
                         "detail": f"{upload_payload['graph_nodes']} entities, "
                                   f"{len(upload_payload['relations'])} relations"},
                    ],
                }
                yield _ndjson(meta)
                for i, source in enumerate(upload_payload["sources"]):
                    if source not in upload_payload["grouped"]:
                        continue
                    yield _ndjson({"type": "section", "source": source,
                                   "index": i})
                    prompt = build_single_doc_prompt(
                        user_query, source, upload_payload["grouped"][source],
                        graph_relations=upload_payload["relations"])
                    for tok in oc.run_llm_query(prompt):
                        yield _ndjson({"type": "token", "t": tok})
            else:
                ctx = graphrag.retrieve(user_query)
                app.logger.info("Chat: %d vector hits, %d graph relations, entities=%s",
                                len(ctx["vector_results"]),
                                len(ctx["graph_relations"]), ctx["query_entities"])
                yield _ndjson({
                    "type": "meta", "mode": "corpus",
                    "query_entities": ctx["query_entities"],
                    "graph_relations": ctx["graph_relations"],
                    "vector_results": ctx["vector_results"],
                    "highlighted_nodes": ctx["highlighted_nodes"],
                    "pipeline": [
                        {"step": s["step"], "detail": s["detail"]}
                        for s in ctx["pipeline_steps"]
                    ],
                })
                for tok in graphrag.stream(user_query, ctx):
                    yield _ndjson({"type": "token", "t": tok})
            yield _ndjson({"type": "done"})
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("chat stream failed")
            yield _ndjson({"type": "error", "message": str(exc)})

    return Response(stream_with_context(generate()),
                    mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-store",
                             "X-Accel-Buffering": "no"})


def _prepare_uploads(files, user_query: str) -> dict:
    """Extract, chunk, embed and graph the uploaded documents."""
    texts: list[str] = []
    all_chunks: list[str] = []
    all_metadatas: list[dict] = []
    skipped: list[str] = []
    sources: list[str] = []

    for file in files:
        filename = secure_filename(file.filename) or "upload"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in tika.TIKA_EXTENSIONS and ext not in tika.STRUCTURED_EXTENSIONS:
            skipped.append(filename)
            continue
        data = file.read()
        if not data:
            skipped.append(filename)
            continue
        try:
            text = tika.extract_text_from_bytes(data, filename=filename)
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Tika failed on %s: %s", filename, exc)
            skipped.append(filename)
            continue
        if not text.strip():
            skipped.append(filename)
            continue
        texts.append(text)
        if filename not in sources:
            sources.append(filename)
        for chunk in chunk_text_hybrid(text):
            all_chunks.append(chunk)
            all_metadatas.append({"source_document": filename})

    if not all_chunks:
        detail = f" Could not read: {', '.join(skipped)}." if skipped else ""
        return {"error": f"No readable text in the uploaded file(s).{detail}"}

    upload_graph = GraphStore(autoload=False)
    by_source: dict[str, list[str]] = defaultdict(list)
    for chunk, meta in zip(all_chunks, all_metadatas):
        by_source[meta["source_document"]].append(chunk)
    for src, cs in by_source.items():
        upload_graph.add_triples(
            [t for c in cs for t in graph_rag.extract_triples(c, use_llm=False)],
            source=src)

    entities = graph_rag.extract_query_entities(user_query)
    relations = upload_graph.traverse(entities) if config.GRAPH_ENABLED else []

    # A generic question ("what did the court hold?") names no statute or party,
    # so there is nothing to seed traversal with and the document graph we just
    # built goes unused. Fall back to the document's own best-connected
    # entities, which is what someone asking about their upload wants to see.
    anchors_for_highlight: list[str] = []
    if config.GRAPH_ENABLED and upload_graph.graph.number_of_nodes():
        anchors_for_highlight = sorted(
            upload_graph.graph.nodes,
            key=lambda n: -upload_graph.graph.degree(n))[:6]
    if config.GRAPH_ENABLED and not relations and anchors_for_highlight:
        relations = upload_graph.traverse(anchors_for_highlight)
        if not entities:
            entities = anchors_for_highlight[:4]

    with EphemeralRetriever() as scratch:
        scratch.add(all_chunks, all_metadatas)
        hits = scratch.retrieve(user_query, top_k=config.UPLOAD_TOP_K)

    grouped: dict[str, list[str]] = defaultdict(list)
    for h in hits:
        grouped[h.metadata.get("source_document", "Unknown")].append(h.text)

    return {
        "texts": texts,
        "sources": sources,
        "skipped": skipped,
        "grouped": grouped,
        "entities": entities,
        "relations": relations,
        # Highlights must resolve against the graph the UI actually renders
        # (the persistent corpus graph), not the throwaway upload graph --
        # otherwise nothing lights up. The uploaded documents' entities
        # usually exist in the corpus graph too.
        "highlighted": graphrag.store.subgraph_nodes(
            entities + [a for a in anchors_for_highlight if a not in entities]),
        "graph_nodes": upload_graph.graph.number_of_nodes(),
        "hits": [{"chunk": h.text, "score": round(h.score, 4),
                  "source": h.metadata.get("source_document", "Unknown")}
                 for h in hits],
    }


# ------------------------------------------------------------ drafting ------
@app.route("/doctypes", methods=["GET"])
def doctypes():
    """Document types the drafter can produce, with their section skeletons."""
    return jsonify({"types": drafting.list_types()})


@app.route("/draft", methods=["POST"])
def draft():
    """Draft a document from the clause bank, streamed like /chat.

    The corpus holds 397 clause types of real drafted language, so a draft is
    assembled from wording the corpus actually contains rather than invented.
    """
    req = (request.form.get("request") or request.form.get("query") or "").strip()
    key = (request.form.get("doctype") or "").strip()
    if not req:
        return Response(_ndjson({"type": "error",
                                 "message": "Describe the document you want drafted."}),
                        mimetype="application/x-ndjson", status=400)

    dt = drafting.DOC_TYPES.get(key) or drafting.infer_type(req)
    material = drafting.gather_material(dt, req, retriever)
    prompt = drafting.build_prompt(dt, req, material)
    app.logger.info("Draft: %s | %d clauses, %d statute passages",
                    dt.key, len(material["clauses"]), len(material["statutes"]))

    def generate():
        try:
            yield _ndjson({
                "type": "meta", "mode": "draft",
                "doctype": dt.key, "doctype_name": dt.name,
                "sections": dt.sections,
                "vector_results": [
                    {"chunk": c["text"], "score": 0, "source": c["source"]}
                    for c in material["clauses"] + material["statutes"]],
                "graph_relations": [], "query_entities": [], "highlighted_nodes": [],
                "pipeline": [
                    {"step": "Document type", "detail": dt.name},
                    {"step": "Clause bank", "detail": f"{len(material['clauses'])} clauses retrieved"},
                    {"step": "Statutory material", "detail": f"{len(material['statutes'])} passages"},
                ],
            })
            for tok in oc.run_llm_query(prompt, system=drafting.DRAFT_SYSTEM):
                yield _ndjson({"type": "token", "t": tok})
            yield _ndjson({"type": "done"})
        except Exception as exc:  # noqa: BLE001
            app.logger.exception("draft failed")
            yield _ndjson({"type": "error", "message": str(exc)})

    return Response(stream_with_context(generate()),
                    mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.route("/export", methods=["POST"])
def export():
    """Save an answer or draft as Markdown, PDF or DOCX.

    Rendering happens server side so the PDF and DOCX get real typography and
    carry the sources, rather than being a screenshot of the page.
    """
    data = request.get_json(silent=True) or {}
    fmt = (data.get("format") or "md").lower()
    title = (data.get("title") or "Sanhita document").strip()[:120]
    body = data.get("markdown") or ""
    sources = data.get("sources") or []
    if not body.strip():
        return jsonify({"error": "Nothing to export."}), 400

    try:
        buf, mime, ext = exporters.render(fmt, title, body, sources)
    except exporters.UnsupportedFormat as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("export failed")
        return jsonify({"error": f"Could not build the file: {exc}"}), 500

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_") or "sanhita"
    return send_file(buf, mimetype=mime, as_attachment=True,
                     download_name=f"{safe}.{ext}")


# ------------------------------------------------------------ graph routes ---
@app.route("/graph", methods=["GET"])
def graph_data():
    """Knowledge graph serialised for D3.

    Capped at GRAPH_VIZ_MAX_NODES (highest-degree nodes first): the corpus
    graph has far more nodes than a force simulation can render.
    """
    try:
        limit = int(request.args.get("max_nodes", config.GRAPH_VIZ_MAX_NODES))
    except ValueError:
        limit = config.GRAPH_VIZ_MAX_NODES
    return jsonify(graphrag.store.get_graph_data(max_nodes=limit))


@app.route("/stats", methods=["GET"])
def stats():
    """Graph + vector-index statistics (upstream's /stats, extended)."""
    payload = graphrag.store.get_stats()
    payload["total_chunks"] = retriever.count()
    payload["collection"] = config.COLLECTION
    return jsonify(payload)


@app.route("/graph_query", methods=["POST"])
def graph_query():
    """Non-streaming hybrid query returning the full Graph RAG payload.

    Mirrors the upstream /query response shape (answer, pipeline_steps,
    vector_results, graph_relations, query_entities, highlighted_nodes) so the
    graph UI can animate the pipeline and highlight the active subgraph. The
    streaming /query route is kept for the original chat UI.
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or request.form.get("query") or "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    if not is_legal_query(question):
        return jsonify({
            "error": "I am a legal assistant and can only process queries "
                     "related to legal matters.",
            "is_guard_error": True,
        }), 422

    if retriever.count() == 0 and graphrag.store.graph.number_of_nodes() == 0:
        return jsonify({
            "error": "Nothing indexed yet. Run: python embed_and_index.py "
                     "and python build_graph.py",
        }), 400

    try:
        return jsonify({"success": True, **graphrag.query(question)})
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("graph_query failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/reset", methods=["POST"])
def reset_graph():
    """Clear the knowledge graph only.

    Upstream's /reset rebuilt the whole engine, wiping vectors too. Here the
    Chroma index is a multi-hour build, so this route deliberately never
    touches it -- rebuild vectors with embed_and_index.py if that is what you
    want. Requires {"confirm": true} so a stray POST cannot discard the graph.
    """
    data = request.get_json(silent=True) or {}
    if not data.get("confirm"):
        return jsonify({
            "error": "Refusing to clear the graph without confirmation. "
                     'POST {"confirm": true} to proceed. '
                     "The vector index is never touched by this route.",
            "nodes": graphrag.store.graph.number_of_nodes(),
        }), 400
    before = graphrag.store.graph.number_of_nodes()
    graphrag.store.clear()
    graphrag.store.save()
    return jsonify({"success": True, "message": f"Graph cleared ({before} nodes). "
                                                "Vector index untouched.",
                    "vector_chunks": retriever.count()})


# ------------------------------------------------------------------- main ----
if __name__ == "__main__":
    print("Sanhita — Indian legal research\n")
    print(f"Ollama   : {config.OLLAMA_HOST} (LLM={config.LLM_MODEL}, "
          f"embed={config.EMBED_MODEL})")
    try:
        oc.preflight()
        print("Ollama preflight OK")
    except oc.OllamaError as exc:
        print(f"WARNING: {exc}")
    try:
        tika.start_server()
        print(f"Tika     : {config.TIKA_URL} (up)")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: Tika did not start: {exc}")
    print(f"Chroma   : {config.CHROMA_PATH}/{config.COLLECTION} "
          f"({retriever.count()} chunks)")
    print(f"Graph    : {config.GRAPH_PATH} "
          f"({graphrag.store.graph.number_of_nodes()} nodes, "
          f"{graphrag.store.graph.number_of_edges()} edges)")
    app.run(debug=True, port=int(os.getenv("PORT", "5002")), use_reloader=False)

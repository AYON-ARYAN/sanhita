"""Graph RAG engine: knowledge graph + vector search + Qwen.

Cherry-picked from AYON-ARYAN/graph-rag and adapted to this stack:

  upstream                        here
  ------------------------------  --------------------------------------------
  Groq llama-3.1-8b-instant       Qwen via Ollama (`ollama_client`)
  SentenceTransformer + FAISS     nomic-embed-text via Ollama + Chroma
  PyPDF2                          Apache Tika
  in-memory nx.DiGraph            persistent `GraphStore`
  LLM triple extraction only      deterministic legal patterns + opt-in LLM
  time.sleep(12) for Groq TPM     not needed; local inference is the limit

Kept intact: the triple schema, the JSON-array extraction prompt with retry,
query-entity extraction, BFS graph traversal into "A --[rel]--> B" context,
the combined vector+graph prompt, pipeline-step telemetry, and the
highlighted-subgraph payload for visualisation.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Generator, List, Sequence

import config
import legal_entities as le
import ollama_client as oc
from graph_store import GraphStore

# ---------------------------------------------------------------- prompts ---
EXTRACT_SYSTEM = ("You are a precise information-extraction engine. "
                  "You output only a JSON array and nothing else.")

EXTRACT_PROMPT = """Extract entities and their relationships from the following text.
Return a JSON array of objects, each with:
- "head": the source entity name
- "head_type": one of Person, Organization, Court, Case, Party, Statute, Act, Concept, Location, Date
- "relation": a short verb phrase linking head to tail (e.g. invokes, party_to, decided_by, represented_by)
- "tail": the target entity name
- "tail_type": same vocabulary as head_type

IMPORTANT: Return ONLY a valid JSON array. No prose, no code fences.
If no entities are found, return [].

Text: {chunk}

JSON:"""

QUERY_ENTITY_SYSTEM = ("You extract lookup keys for a knowledge graph. "
                       "You output only a JSON array of strings.")

QUERY_ENTITY_PROMPT = """Extract the key entities or topics from this question that should be looked up in a legal knowledge graph.
Return ONLY a JSON array of strings. No prose.

Question: {query}

JSON:"""


def _parse_json_array(text: str) -> List:
    """Pull the first JSON array out of a model response."""
    if not text:
        return []
    fenced = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                    flags=re.MULTILINE).strip()
    for candidate in (fenced, text):
        m = re.search(r"\[.*\]", candidate, re.DOTALL)
        if not m:
            continue
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            continue
    return []


# ------------------------------------------------------------- extraction ---
def llm_extract_triples(chunk: str, retries: int = 2) -> List[Dict[str, str]]:
    """LLM triple extraction (upstream's approach).

    Costs roughly 100s per 500-token chunk on this machine, so callers must
    cap how many chunks they send. Upstream slept 12s between chunks for
    Groq's rate limit; that is unnecessary locally.
    """
    for _ in range(retries):
        try:
            out = oc.complete(EXTRACT_PROMPT.format(chunk=chunk),
                              system=EXTRACT_SYSTEM)
        except Exception:  # noqa: BLE001
            continue
        triples = [t for t in _parse_json_array(out) if isinstance(t, dict)]
        if triples:
            return triples
    return []


def extract_triples(chunk: str, use_llm: bool | None = None) -> List[Dict[str, str]]:
    """Deterministic legal extraction, optionally enriched by the LLM."""
    use_llm = config.GRAPH_LLM_EXTRACT if use_llm is None else use_llm
    triples = le.extract_triples(chunk)
    if use_llm:
        triples.extend(llm_extract_triples(chunk))
    return triples


def extract_query_entities(query: str, use_llm: bool | None = None) -> List[str]:
    """Lookup keys for the graph.

    Deterministic by default: legal text names its entities in citable forms,
    and this avoids a second LLM round-trip on every query. Set
    GRAPH_LLM_EXTRACT=1 to use the model instead (upstream behaviour).
    """
    use_llm = config.GRAPH_LLM_EXTRACT if use_llm is None else use_llm

    if use_llm:
        try:
            out = oc.complete(QUERY_ENTITY_PROMPT.format(query=query),
                              system=QUERY_ENTITY_SYSTEM)
            ents = [str(e).strip() for e in _parse_json_array(out) if str(e).strip()]
            if ents:
                return ents[:8]
        except Exception:  # noqa: BLE001
            pass

    ents: List[str] = []

    # Statute and article citations are the highest-value graph keys.
    for m in le._ARTICLE.finditer(query):
        ents.append(f"Article {m.group(1)}")
    for m in le._SECTION_OF_ACT.finditer(query):
        if m.group(1):
            ents.append(f"Section {m.group(1).strip()}")
    for canonical, rx in le._ACT_RE.items():
        if rx.search(query):
            ents.append(canonical)
    # Capitalised phrases (party and court names), minus sentence-initial noise.
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})", query):
        phrase = m.group(1).strip()
        if phrase.lower() not in le._NOISE and m.start() != 0:
            ents.append(phrase)
    # Deliberately NO generic-keyword fallback. Probing the graph with bare
    # words like "decision" or "grounds" substring-matches unrelated case
    # names ("Recreation Grounds Co Ltd vs Taylor") and produces highlights
    # and relations that have nothing to do with the question. An empty list
    # is the honest answer; callers with better seeds (e.g. an uploaded
    # document's own entities) supply them instead.

    seen: set[str] = set()
    uniq = [e for e in ents if not (e.lower() in seen or seen.add(e.lower()))]
    return uniq[:8]


# ------------------------------------------------------------------ query ---
# Answer length is a real failure mode here, in both directions. Five numbered
# rules made the 3B model echo the context back verbatim; "be concise" made it
# answer a legal question in a single sentence, which is useless to someone
# trying to understand a provision. Naming the parts an answer should cover
# works far better on a 3B model than an abstract "be thorough".
HYBRID_PROMPT = """You are a legal research assistant. Explain the answer properly, using the passages and entity relations below.

Passages:
{vector_context}

Entity relations:
{graph_context}

Write a full explanation, not a one-line answer. Cover these, in this order,
and only as far as the material supports them:

- **The answer** - state it directly in the first sentence or two.
- **The provision** - name the section or article that governs and say what it
  provides. Quote the operative words where the passages give them.
- **What it means in practice** - unpack the requirements, conditions or
  ingredients a reader would need to know.
- **Related provisions** - use the entity relations to point out connected
  sections, cases or courts that bear on the question.
- **Limits** - if the passages establish the provision but not the detail
  asked for, say so plainly instead of filling the gap.

Check a passage concerns the same offence or provision as the question before
relying on it. Cite passages as [1], [2]. Use Markdown headings and bullets.
Aim for three to six short paragraphs.

Question: {question}
Answer:"""


class GraphRAG:
    """Hybrid retrieval: Chroma vectors + knowledge-graph traversal."""

    def __init__(self, retriever=None, store: GraphStore | None = None):
        if retriever is None:
            from retriever import retriever as default_retriever
            retriever = default_retriever
        self.retriever = retriever
        self.store = store if store is not None else GraphStore()

    # Which act a Q&A file is about. A record in ipc_qa.json says "under
    # section 412" with no act beside it, so the citation is unresolvable from
    # the sentence alone -- but the file it came from settles it.
    SOURCE_ACT_HINTS = {
        "ipc_qa": "Indian Penal Code",
        "crpc_qa": "Code of Criminal Procedure",
        "constitution_qa": "Constitution of India",
    }

    @classmethod
    def _seeds_from_hits(cls, hits, limit: int = 6) -> List[str]:
        """Derive graph seeds from the retrieved passages.

        A plain-English question ("what happens if a cheque bounces?") carries
        no citation to look up, so query-side extraction finds nothing and the
        graph contributes nothing. The retrieved passages are topically
        relevant by construction, so the statutes and cases named inside them
        are good seeds.

        Specific nodes are preferred over broad ones: seeding on an Act such
        as the Indian Penal Code returns its hundreds of
        "Section N --[part_of]--> Indian Penal Code" edges, which are true and
        useless. A Statute or Case seed returns the relations that answer
        something.
        """
        WEIGHT = {"Statute": 4, "Case": 3, "Court": 2, "Act": 1}
        scored: Dict[str, float] = {}
        for h in hits:
            text = getattr(h, "text", "") or ""
            meta = getattr(h, "metadata", {}) or {}
            src = str(meta.get("source_file", "")).lower()

            # Resolve bare section references using the source file's subject.
            hint = next((act for key, act in cls.SOURCE_ACT_HINTS.items()
                         if key in src), None)
            if hint:
                for m in re.finditer(r"[Ss]ection\s+(\d+[A-Z]{0,2})", text):
                    name = f"Section {m.group(1)} {hint}"
                    scored[name] = scored.get(name, 0) + WEIGHT["Statute"]

            for t in le.extract_triples(text):
                for name, kind in ((t["head"], t["head_type"]),
                                   (t["tail"], t["tail_type"])):
                    if kind in WEIGHT and len(name) > 3:
                        scored[name] = scored.get(name, 0) + WEIGHT[kind]
        return [n for n, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:limit]

    # --- retrieval only: fast, no generation ------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> Dict:
        steps: List[Dict] = []

        hits = self.retriever.retrieve(question, top_k=top_k)
        steps.append({"step": "Vector Search", "status": "done",
                      "detail": f"Found {len(hits)} relevant chunks"})

        entities = extract_query_entities(question)
        source = "question"
        if not entities:
            entities = self._seeds_from_hits(hits)
            source = "retrieved passages"
        elif all(e in le.ACTS for e in entities):
            # The question named only whole enactments. Seeding on "Indian
            # Penal Code" returns its hundreds of "Section N part_of IPC"
            # edges -- true and useless. Add the specific provisions that the
            # retrieved passages actually discuss.
            extra = [e for e in self._seeds_from_hits(hits) if e not in entities]
            if extra:
                entities = extra + entities
                source = "question + retrieved passages"
        steps.append({
            "step": "Entity Extraction", "status": "done",
            "detail": (f"From {source}: {', '.join(entities)}" if entities
                       else "No legal entities found in the question or passages"),
        })

        relations = self.store.traverse(entities) if config.GRAPH_ENABLED else []
        steps.append({"step": "Graph Traversal", "status": "done",
                      "detail": f"Found {len(relations)} graph relations"})

        return {
            "vector_results": [
                {"chunk": h.text, "score": round(h.score, 4),
                 "source": h.metadata.get("source_file", "Unknown Source")}
                for h in hits
            ],
            "graph_relations": relations,
            "query_entities": entities,
            "highlighted_nodes": self.store.subgraph_nodes(entities),
            "pipeline_steps": steps,
        }

    def build_prompt(self, question: str, ctx: Dict) -> str:
        vector_context = "\n\n".join(
            f"[{i+1}] (source: {r['source']})\n{r['chunk']}"
            for i, r in enumerate(ctx["vector_results"])
        ) or "No vector results found."
        graph_context = "\n".join(ctx["graph_relations"]) or "No graph relations found."
        return HYBRID_PROMPT.format(vector_context=vector_context,
                                    graph_context=graph_context,
                                    question=question)

    # --- streaming answer, for the existing UI ----------------------------
    def stream(self, question: str, ctx: Dict | None = None
               ) -> Generator[str, None, None]:
        ctx = ctx or self.retrieve(question)
        yield from oc.run_llm_query(self.build_prompt(question, ctx))

    # --- full JSON payload, upstream's /query shape -----------------------
    def query(self, question: str) -> Dict:
        ctx = self.retrieve(question)
        ctx["pipeline_steps"].append(
            {"step": "LLM Generation", "status": "done", "detail": "Answer generated"})
        ctx["answer"] = "".join(oc.run_llm_query(self.build_prompt(question, ctx)))
        return ctx

    # --- ingestion --------------------------------------------------------
    def ingest_chunks(self, chunks: Sequence[str], source: str,
                      use_llm: bool | None = None, max_llm_chunks: int | None = None
                      ) -> Dict:
        """Add a document's chunks to the graph. Returns upstream-style counts."""
        use_llm = config.GRAPH_LLM_EXTRACT if use_llm is None else use_llm
        cap = config.GRAPH_LLM_MAX_CHUNKS if max_llm_chunks is None else max_llm_chunks

        all_triples: List[Dict[str, str]] = []
        for i, ch in enumerate(chunks):
            all_triples.extend(le.extract_triples(ch))
            if use_llm and i < cap:
                all_triples.extend(llm_extract_triples(ch))

        self.store.add_triples(all_triples, source=source)
        self.store.note_document(source)
        entities = {t["head"] for t in all_triples} | {t["tail"] for t in all_triples}
        return {
            "filename": source,
            "chunks_created": len(chunks),
            "entities_extracted": len(entities),
            "relations_extracted": len(all_triples),
            "total_nodes": self.store.graph.number_of_nodes(),
            "total_edges": self.store.graph.number_of_edges(),
        }

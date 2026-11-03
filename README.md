# Sanhita — संहिता

**The code, searchable.** A fully local question-answering and drafting system
for Indian law. No cloud API, no API key, nothing leaves the machine.

*Sanhita* means a code or compilation of texts, as in the Bharatiya Nyaya
Sanhita. The name describes what it indexes: statutes, judgments and contract
clauses — searchable, connected, and answerable with the evidence shown.

| | |
|---|---|
| Generation | **Qwen 2.5 3B-instruct** via Ollama |
| Embeddings | **nomic-embed-text** (768-dim) via Ollama |
| Retrieval | **Hybrid** — Chroma dense vectors + BM25 lexical, fused by RRF |
| Knowledge graph | **NetworkX** — 32k legal entities, 39k relations |
| Extraction | **Apache Tika** — PDF, DOCX, XLSX, PPTX, CSV, HTML, more |
| Interface | One page: ask, draft, evidence, graph |

---

## Quick start

**Prerequisites:** Docker, and [Ollama](https://ollama.com) installed on your
machine.

```bash
git clone https://github.com/AYON-ARYAN/sanhita.git
cd sanhita

# 1. put the corpus in place (see "Getting the corpus" below)
# 2. start Ollama on your machine
ollama serve

# 3. one command
make setup
```

`make setup` pulls the two models, builds the index, graph and lexical index,
then starts the app at **http://localhost:5002**.

> **It takes about an hour** to embed the full 121,112-chunk corpus on an
> Apple M2. It is resumable — interrupt it and run `make setup` again.
> For a usable system in ~10 minutes instead, use `make setup-fast`, which
> indexes the Q&A datasets and judgments only, then run `make index` later to
> finish the contract clauses.

Prefer no Docker? `make dev` runs it straight from a virtualenv.

---

## Getting the corpus

`data/` is **not in this repository** — it is ~176 MB and distributed
separately. Unpack the archive you were sent so the tree looks like this:

```
sanhita/
├── data/
│   ├── csvs/     397 contract-clause files
│   ├── jsons/    4 Q&A datasets (IPC, CrPC, Constitution, general)
│   └── pdfs/     12 court judgments
└── index/        (optional) prebuilt embeddings — skips the hour-long build
```

If the archive includes `index/`, **you can skip the index build entirely**:

```bash
make up          # just start it
make status      # confirm the index is seen
```

---

## Why Ollama runs on your machine, not in a container

On macOS a containerised Ollama cannot reach the Metal GPU, so the model falls
back to CPU and embedding becomes several times slower. The app container
therefore talks to the Ollama on your host via `host.docker.internal`.

If you are on Linux, or you do not mind CPU inference, run everything in
containers instead:

```bash
docker compose --profile bundled-ollama up
```

---

## Commands

```
make setup        first run: models + index + graph + start   (~1 hour)
make setup-fast   same, Q&A and judgments only                (~10 min)

make up           start                 -> http://localhost:5002
make down         stop
make logs         follow the app log
make status       what is running, how much is indexed
make dev          run from .venv, no Docker

make models       pull the Ollama models
make index        build or resume the vector index
make graph        build the knowledge graph
make bm25         rebuild the lexical index
make reindex      all three

make test         smoke tests (15 checks)
make test-http    HTTP tests (50 checks, app must be running)
```

---

## Using it

**Ask** — a question about Indian law. The answer arrives with its citations
in the margin, the retrieved passages in the evidence drawer, and the matched
part of the knowledge graph lit up.

**Draft** — switch the composer to Draft and describe a document: *"a legal
notice for a cheque of Rs 4,50,000 dishonoured for insufficient funds"*. It
picks the document type, pulls real clause language from the 397-file clause
bank, and drafts it. Eight types: s.138 notice, general notice, NDA, services
agreement, employment, lease, bail application, reply to notice.

**Upload** — drop a PDF, DOCX, XLSX, CSV, TXT or HTML onto the composer and
ask about it. Multiple files each get their own section in the answer.

**Save** — any answer or draft exports to PDF, Word or Markdown, with its
sources.

**Graph** — drag the divider to make it bigger, or go full screen. Zoom to
reveal more labels; the subgraph matching your question is highlighted.

---

## How it works

Three stages. Ingestion runs once; the stores hold what it produced; retrieval
and generation run per question.

```
413 files ──Tika──> 39,559 docs ──chunk──> 121,112 chunks
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
            Chroma + BM25 (what text means                NetworkX graph
            and which words it uses)                      (how entities connect)
                        │                                           │
                        └─────────────────┬─────────────────────────┘
                                          ▼
                             Qwen writes the answer from both
```

Dense search finds passages that *mean* the same thing; BM25 finds the ones
that *use the same words*; the graph answers questions about *connections*
("which cases invoke Article 21?") that no passage similarity would surface.

Full write-up, including the measurements behind each decision:
[`docs/Sanhita_Project_Report.pdf`](docs/Sanhita_Project_Report.pdf).

---

## Source layout

| File | Responsibility |
|---|---|
| `config.py` | Every tunable, environment-overridable |
| `ollama_client.py` | Generation and embeddings, retries, preflight |
| `tika_extract.py` | Tika server lifecycle, text extraction |
| `loaders.py` | Corpus discovery and extraction, full skip accounting |
| `chunking.py` | Sentence-aware chunking |
| `embed_and_index.py` | Vector index builder — resumable, idempotent |
| `lexical.py` | BM25 index |
| `retriever.py` | Hybrid retrieval with RRF fusion |
| `legal_entities.py` | Pattern-based legal triple extraction |
| `graph_store.py` | Persistent knowledge graph, hub-aware traversal |
| `graph_rag.py` | Hybrid engine: vectors + graph + generation |
| `build_graph.py` | Graph builder CLI |
| `drafting.py` | Document types, clause retrieval, drafting prompts |
| `exporters.py` | PDF / DOCX / Markdown rendering |
| `app.py` | HTTP interface |
| `templates/index.html` | The interface |

---

## Configuration

Copy `.env.example` to `.env`. Everything has a working default.

The ones that matter most:

| Variable | Default | Why you might change it |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:3b-instruct` | A larger model answers better. 3B is chosen to fit 8 GB alongside the embedder. |
| `LLM_NUM_CTX` | `8192` | Drop to `4096` on 8 GB if you are also building the index. |
| `TOP_K` | `8` | More passages, fuller answers, slower. |
| `ENABLE_GUARD` | `0` | `1` re-enables the off-topic filter. At 3B it misclassifies real legal questions. |
| `HYBRID_SEARCH` | `1` | `0` for dense-only retrieval. |

Changing `EMBED_MODEL` changes the vector space and invalidates the index. The
builder checks the dimension before writing and refuses on a mismatch, so pair
a new embedding model with a new `COLLECTION` name and `make clean-index`.

---

## Troubleshooting

**"Ollama is not reachable"** — run `ollama serve`. In Docker the app looks for
`host.docker.internal:11434`; on Linux the compose file maps that to the host
gateway.

**Answers say the passages do not contain the answer** — the index is probably
still building. Check with `make status`. Ingestion indexes the Q&A datasets
and judgments first, so the system is useful well before it finishes.

**Everything is slow** — a 3B model on CPU is slow. Make sure Ollama is running
on the host rather than in a container, so it can use the GPU. Answers take
roughly 30–60 s on an M2.

**The app dies during an index build** — memory. On an 8 GB machine, Ollama plus
the build plus the web app is too much and the OS kills something. Run
`make down`, build, then `make up`.

---

## Limitations

- **A 3B model is a small model.** It follows complex formatting instructions
  poorly and occasionally repeats a clause when drafting. The model name is one
  line of configuration.
- **No reranking.** Retrieved passages go to the model as ranked.
- **Corpus quality limits answers.** Some Q&A records pair a vague question
  with an absolute answer; retrieved, they produce confident wrong answers. The
  system reports the corpus faithfully.
- **Not legal advice.** It retrieves and summarises a fixed corpus. It does not
  know whether a judgment has been overruled or a section amended. Drafts are
  starting points requiring review by a qualified professional.

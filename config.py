"""Central configuration for the legal-AI RAG stack.

Everything is env-overridable so you can swap models without touching code.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- Ollama ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Generation model (Qwen). Must be a model present in `ollama list`.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")

# Embedding model served by Ollama.
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# nomic-embed-text emits 768-dim vectors. Asserted at index build time so a
# model swap can never silently corrupt an existing collection.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.1"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "1500"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

# Ollama request timeouts (seconds). Embedding batches are the slow path.
OLLAMA_EMBED_TIMEOUT = int(os.getenv("OLLAMA_EMBED_TIMEOUT", "600"))
OLLAMA_GEN_TIMEOUT = int(os.getenv("OLLAMA_GEN_TIMEOUT", "600"))

# ------------------------------------------------------------------ Tika ----
TIKA_URL = os.getenv("TIKA_URL", "http://localhost:9998").rstrip("/")
TIKA_JAR = os.getenv(
    "TIKA_JAR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "vendor", "tika", "tika-server-standard-3.3.3.jar"),
)
TIKA_TIMEOUT = int(os.getenv("TIKA_TIMEOUT", "300"))
TIKA_CONFIG = os.getenv(
    "TIKA_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "vendor", "tika", "tika-config.xml"),
)

# ----------------------------------------------------------------- Chroma ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "index/chroma")
# New collection name: the old `legal_docs` holds all-mpnet-base-v2 vectors,
# which live in a different vector space. Mixing them would silently poison
# retrieval, so Ollama-embedded vectors get their own collection.
COLLECTION = os.getenv("COLLECTION", "legal_docs_ollama")

# --------------------------------------------------------------- Chunking ---
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "30"))

# ------------------------------------------------------------- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "8"))
UPLOAD_TOP_K = int(os.getenv("UPLOAD_TOP_K", "15"))

# ---------------------------------------------------------------- Ingest ----
DATA_DIRS = [d for d in os.getenv("DATA_DIRS", "data").split(",") if d]
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
# Set TIKA_FOR_JSON=1 to push .json/.jsonl through Tika instead of the
# structured record reader.
TIKA_FOR_JSON = os.getenv("TIKA_FOR_JSON", "0") == "1"

# ------------------------------------------------- Embedding task prefixes ---
# nomic-embed-text is an ASYMMETRIC retrieval model: it was trained with task
# prefixes and expects "search_document: " on indexed passages and
# "search_query: " on queries. Ollama does NOT add these for you, and omitting
# them measurably degrades recall. Set both to "" for a symmetric model
# (e.g. mxbai-embed-large, bge-m3).
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")

# ------------------------------------------------------------------ Graph ---
# Knowledge-graph settings (graph RAG). The graph is persisted as JSON
# node-link data rather than held in memory, so it survives restarts.
GRAPH_PATH = os.getenv("GRAPH_PATH", "index/graph/graph.json")
GRAPH_DEPTH = int(os.getenv("GRAPH_DEPTH", "2"))
GRAPH_MAX_RELATIONS = int(os.getenv("GRAPH_MAX_RELATIONS", "40"))
GRAPH_VIZ_MAX_NODES = int(os.getenv("GRAPH_VIZ_MAX_NODES", "400"))
# Hybrid retrieval: include knowledge-graph relations alongside vector hits.
GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "1") == "1"
# LLM triple extraction costs ~100s/chunk locally, so it is opt-in and capped.
GRAPH_LLM_EXTRACT = os.getenv("GRAPH_LLM_EXTRACT", "0") == "1"
GRAPH_LLM_MAX_CHUNKS = int(os.getenv("GRAPH_LLM_MAX_CHUNKS", "4"))
# A node with more incident edges than this is treated as a hub: it can appear
# as an endpoint in graph context but is never expanded through, so traversal
# from "Article 21" does not flood out via "Constitution of India".
GRAPH_HUB_DEGREE = int(os.getenv("GRAPH_HUB_DEGREE", "60"))
# Cap on nodes highlighted for one query. Without it a hub seed lights up the
# whole rendered graph and conveys nothing.
GRAPH_HIGHLIGHT_MAX = int(os.getenv("GRAPH_HIGHLIGHT_MAX", "60"))

# ----------------------------------------------------------------- BM25 ----
# Hybrid retrieval. Dense search alone misses passages whose wording differs
# from the question even when they hold the answer; BM25 recovers those by
# term overlap. Results are fused with Reciprocal Rank Fusion.
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1") == "1"
BM25_PATH = os.getenv("BM25_PATH", "index/bm25/bm25.pkl")
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "50"))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "50"))
RRF_K = int(os.getenv("RRF_K", "60"))
# Allow this much drift between cached index size and collection size before
# rebuilding (the vector index may still be filling).
BM25_STALE_TOLERANCE = int(os.getenv("BM25_STALE_TOLERANCE", "2000"))

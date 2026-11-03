"""Ollama-backed LLM generation and embeddings.

Replaces the previous SambaNova/OpenAI client and the local
SentenceTransformer embedder. Both ingestion and query-time embedding go
through `embed_documents` / `embed_query` here, so the two can never drift
into different vector spaces.
"""
from __future__ import annotations

import json
import time
from typing import Generator, Iterable, List, Sequence

import requests

import config


class OllamaError(RuntimeError):
    pass


# --------------------------------------------------------------- health -----
def server_version() -> str | None:
    """Return the Ollama server version, or None if it is not reachable."""
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version")
    except requests.RequestException:
        return None


def installed_models() -> List[str]:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def _model_present(name: str, available: Sequence[str]) -> bool:
    """Match allowing for the implicit :latest tag."""
    if name in available:
        return True
    base = name.split(":")[0]
    return any(a == name or a.split(":")[0] == base for a in available)


def preflight(require_llm: bool = True, require_embed: bool = True) -> None:
    """Fail loudly and early if Ollama or the configured models are missing."""
    ver = server_version()
    if ver is None:
        raise OllamaError(
            f"Ollama is not reachable at {config.OLLAMA_HOST}. Start it with: ollama serve"
        )
    available = installed_models()
    missing = []
    if require_llm and not _model_present(config.LLM_MODEL, available):
        missing.append(config.LLM_MODEL)
    if require_embed and not _model_present(config.EMBED_MODEL, available):
        missing.append(config.EMBED_MODEL)
    if missing:
        raise OllamaError(
            "Model(s) not installed in Ollama: "
            + ", ".join(missing)
            + "\nInstalled: "
            + (", ".join(available) or "<none>")
            + "\nPull with: "
            + "; ".join(f"ollama pull {m}" for m in missing)
        )


# ------------------------------------------------------------ embeddings ----
def _embed_call(inputs: List[str], attempt_timeout: int) -> List[List[float]]:
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/embed",
        json={"model": config.EMBED_MODEL, "input": inputs},
        timeout=attempt_timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise OllamaError(f"Ollama embed error: {payload['error']}")
    vectors = payload.get("embeddings")
    if not vectors or len(vectors) != len(inputs):
        raise OllamaError(
            f"Expected {len(inputs)} embeddings, got {0 if not vectors else len(vectors)}"
        )
    return vectors


def _embed(texts: Sequence[str], prefix: str, retries: int = 3) -> List[List[float]]:
    """Embed a batch, retrying with exponential backoff on transient failures."""
    if not texts:
        return []
    inputs = [f"{prefix}{t}" for t in texts]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _embed_call(inputs, config.OLLAMA_EMBED_TIMEOUT)
        except (requests.RequestException, OllamaError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise OllamaError(f"Embedding failed after {retries} attempts: {last}")


def embed_documents(texts: Sequence[str]) -> List[List[float]]:
    """Embed passages for storage (uses the document task prefix)."""
    return _embed(texts, config.EMBED_DOC_PREFIX)


def embed_query(text: str) -> List[float]:
    """Embed a single search query (uses the query task prefix)."""
    return _embed([text], config.EMBED_QUERY_PREFIX)[0]


def embed_documents_batched(
    texts: Sequence[str], batch_size: int | None = None
) -> Iterable[tuple[int, List[List[float]]]]:
    """Yield (start_offset, vectors) per batch so callers can stream to disk."""
    size = batch_size or config.EMBED_BATCH
    for i in range(0, len(texts), size):
        yield i, embed_documents(texts[i : i + size])


# ------------------------------------------------------------ generation ----
def run_llm_query(prompt: str, system: str | None = None) -> Generator[str, None, None]:
    """Stream tokens from the configured Qwen model via Ollama /api/chat."""
    system = system or (
        "You are a helpful, concise legal assistant. "
        "Answer only in natural language prose. "
        "Never emit tool-call syntax, function calls, or special tags."
    )
    body = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "options": {
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "num_predict": config.LLM_NUM_PREDICT,
            "num_ctx": config.LLM_NUM_CTX,
        },
    }
    try:
        with requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json=body,
            stream=True,
            timeout=config.OLLAMA_GEN_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    yield f"[LLM ERROR]: {event['error']}"
                    return
                token = (event.get("message") or {}).get("content", "")
                if token:
                    yield token
                if event.get("done"):
                    return
    except requests.RequestException as exc:
        yield f"[LLM ERROR]: {exc}"


def complete(prompt: str, system: str | None = None) -> str:
    """Non-streaming convenience wrapper used by the legal-query classifier."""
    return "".join(run_llm_query(prompt, system=system))

# Sanhita — local legal RAG. Python app only.
#
# Ollama and Tika run as separate services (see compose.yaml). Ollama is NOT
# bundled in this image: on macOS a containerised Ollama cannot reach Metal,
# so embedding would fall back to CPU and run several times slower. The app
# talks to whichever Ollama you point OLLAMA_HOST at — the host's by default.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the healthcheck and the entrypoint's readiness waits.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# punkt at build time so the first request is not waiting on a download
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)" || true

COPY *.py ./
COPY templates ./templates
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# index/ and data/ are bind-mounted, not baked in: the index is ~2 GB and the
# corpus ~176 MB, and both change independently of the code.
VOLUME ["/app/index", "/app/data"]

EXPOSE 5002
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:5002/health || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["serve"]

#!/usr/bin/env bash
# Wait for the services the app needs, then run the requested command.
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
TIKA_URL="${TIKA_URL:-http://tika:9998}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export OLLAMA_HOST TIKA_URL

wait_for () {              # name url seconds
  local name="$1" url="$2" secs="${3:-90}"
  printf 'waiting for %s at %s ' "$name" "$url"
  for _ in $(seq 1 "$secs"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then echo "— up"; return 0; fi
    printf '.'; sleep 1
  done
  echo " — TIMED OUT"
  return 1
}

echo "Sanhita — संहिता"
wait_for "Tika"   "$TIKA_URL/tika" 120 || echo "  Tika unavailable: uploads and ingestion will fail."

if ! wait_for "Ollama" "$OLLAMA_HOST/api/version" 60; then
  cat <<'MSG'

  Ollama is not reachable.

  By default the container talks to the Ollama running on your own machine,
  which on macOS is what gives you GPU acceleration. Start it with:

      ollama serve

  To run Ollama in a container instead (Linux, or if you do not mind CPU):

      docker compose --profile bundled-ollama up

MSG
  exit 1
fi

missing=""
for m in "$LLM_MODEL" "$EMBED_MODEL"; do
  curl -fsS "$OLLAMA_HOST/api/tags" | grep -q "\"${m%%:*}" || missing="$missing $m"
done
if [ -n "$missing" ]; then
  echo "models not installed:$missing"
  echo "pull them with:  make models    (or: ollama pull <model>)"
fi

case "${1:-serve}" in
  serve)       echo "starting on :5002"; exec python app.py ;;
  index)       shift; exec python embed_and_index.py "$@" ;;
  graph)       shift; exec python build_graph.py "$@" ;;
  bm25)        exec python -c "import chromadb,config,lexical;lexical.get(chromadb.PersistentClient(path=config.CHROMA_PATH).get_collection(config.COLLECTION),rebuild=True)" ;;
  smoke)       exec python smoke_test.py ;;
  shell)       exec /bin/bash ;;
  *)           exec "$@" ;;
esac

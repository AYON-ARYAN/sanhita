#!/usr/bin/env bash
# Bring up the whole local stack: Ollama -> Tika -> Flask.
# Nothing here touches the network except a first-time model pull.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

# shellcheck disable=SC1091
set +u; [ -f .env ] && . ./.env; set -u
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
TIKA_URL="${TIKA_URL:-http://localhost:9998}"
LLM_MODEL="${LLM_MODEL:-qwen2.5:3b-instruct}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

echo "Sanhita — Indian legal research"
echo "==> Ollama"
if ! curl -sf --max-time 3 "$OLLAMA_HOST/api/version" >/dev/null; then
  echo "    starting ollama serve"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do
    curl -sf --max-time 2 "$OLLAMA_HOST/api/version" >/dev/null && break
    sleep 1
  done
fi
curl -sf "$OLLAMA_HOST/api/version" >/dev/null \
  && echo "    up: $(curl -s "$OLLAMA_HOST/api/version")" \
  || { echo "    FAILED to start ollama"; exit 1; }

for m in "$LLM_MODEL" "$EMBED_MODEL"; do
  if ! ollama list | awk '{print $1}' | grep -qx "$m"; then
    echo "    pulling $m (one time)"
    ollama pull "$m"
  fi
done

echo "==> Apache Tika"
if ! curl -sf --max-time 3 "$TIKA_URL/tika" >/dev/null; then
  echo "    starting tika-server"
  $PY -c "import tika_extract; tika_extract.start_server(); print('    up')"
else
  echo "    already up"
fi

echo "==> Index"
$PY -c "
from retriever import retriever
n = retriever.count()
print(f'    {n} chunks indexed')
if n == 0:
    print('    EMPTY - build it with:  $PY embed_and_index.py --reset')
"

echo "==> Flask on port ${PORT:-5002}"
exec $PY app.py

# Sanhita — setup and operation.
#   make setup   first time: models, index, graph, then start
#   make up      start it
#   make help    everything else

SHELL := /bin/bash
PY    := .venv/bin/python
LLM   ?= qwen2.5:3b-instruct
EMBED ?= nomic-embed-text

.DEFAULT_GOAL := help

help:
	@echo "Sanhita — संहिता"
	@echo ""
	@echo "  First time"
	@echo "    make setup        pull models, build index + graph, start  (~1 hour)"
	@echo "    make setup-fast   same but index only the Q&A + judgments  (~10 min)"
	@echo ""
	@echo "  Running"
	@echo "    make up           start (Docker)         -> http://localhost:5002"
	@echo "    make down         stop"
	@echo "    make logs         follow the app log"
	@echo "    make dev          run without Docker, using .venv"
	@echo ""
	@echo "  Index"
	@echo "    make models       pull the Ollama models"
	@echo "    make index        build or resume the vector index"
	@echo "    make graph        build the knowledge graph"
	@echo "    make bm25         rebuild the lexical index"
	@echo "    make reindex      index + graph + bm25"
	@echo ""
	@echo "  Checks"
	@echo "    make test         smoke tests (Python layer)"
	@echo "    make test-http    HTTP tests (needs the app running)"
	@echo "    make status       what is running, and how much is indexed"

check-ollama:
	@curl -fsS --max-time 3 $${OLLAMA_HOST:-http://localhost:11434}/api/version >/dev/null \
	  || { echo "Ollama is not running. Start it with:  ollama serve"; exit 1; }
	@echo "Ollama OK"

models: check-ollama
	@for m in $(LLM) $(EMBED); do \
	  if ollama list | awk '{print $$1}' | grep -qx "$$m"; then echo "have $$m"; \
	  else echo "pulling $$m"; ollama pull "$$m"; fi; done

setup: check-ollama models
	@echo "==> building the index (resumable; safe to interrupt)"
	@docker compose run --rm app index
	@echo "==> building the knowledge graph"
	@docker compose run --rm app graph
	@echo "==> building the lexical index"
	@docker compose run --rm app bm25
	@$(MAKE) up

setup-fast: check-ollama models
	@echo "==> indexing the Q&A datasets and judgments only"
	@docker compose run --rm app index --dirs /app/data/jsons /app/data/pdfs
	@docker compose run --rm app graph
	@docker compose run --rm app bm25
	@$(MAKE) up

up:
	@docker compose up -d --build
	@echo "Sanhita is starting -> http://localhost:5002"
	@echo "   follow the log with: make logs"

down:
	@docker compose down

logs:
	@docker compose logs -f app

restart: down up

index:
	@docker compose run --rm app index
graph:
	@docker compose run --rm app graph
bm25:
	@docker compose run --rm app bm25
reindex: index graph bm25

test:
	@docker compose run --rm app smoke
test-http:
	@./test_http.sh

status:
	@docker compose ps 2>/dev/null || true
	@curl -fsS --max-time 5 http://localhost:5002/health 2>/dev/null \
	  | python3 -m json.tool 2>/dev/null || echo "app not responding on :5002"

dev:
	@test -d .venv || python3 -m venv .venv
	@$(PY) -m pip install -q -r requirements.txt
	@$(PY) app.py

clean-index:
	@echo "This deletes index/ — embeddings, graph and BM25. Rebuilding takes ~1 hour."
	@read -p "Type yes to confirm: " a; [ "$$a" = yes ] || exit 1
	@rm -rf index/chroma index/graph index/bm25
	@echo "removed."

.PHONY: help check-ollama models setup setup-fast up down logs restart \
        index graph bm25 reindex test test-http status dev clean-index

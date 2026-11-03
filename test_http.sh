#!/usr/bin/env bash
# HTTP-level checks against a running app (./start.sh must be up).
# Complements smoke_test.py, which tests the Python layer directly.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-http://localhost:5002}"
# The guard is configuration, not a fixed behaviour: default off because the
# 3B classifier false-blocks real legal questions. Assert whichever is active.
GUARD=$(grep -E '^ENABLE_GUARD=' .env 2>/dev/null | tail -1 | cut -d= -f2)
GUARD="${GUARD:-0}"
PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "== /health =="
H=$(curl -s --max-time 20 "$BASE/health")
echo "$H" | grep -q '"reachable": true'      && ok "ollama reachable"   || bad "ollama reachable"
echo "$H" | grep -q '"llm_installed": true'  && ok "llm installed"      || bad "llm installed"
echo "$H" | grep -q '"embed_installed": true'&& ok "embed installed"    || bad "embed installed"
echo "$H" | grep -q '"tika"'                 && ok "tika section"       || bad "tika section"
CHUNKS=$(echo "$H" | sed -n 's/.*"chunks": \([0-9]*\).*/\1/p' | head -1)
[ "${CHUNKS:-0}" -gt 0 ] && ok "index non-empty ($CHUNKS chunks)" || bad "index empty"

echo "== /query =="
CODE=$(curl -s --max-time 290 -o /tmp/q_legal.txt -w '%{http_code}' -X POST "$BASE/query" \
  -F 'query=What does the Constitution of India say about the name and territory of the Union?')
[ "$CODE" = "200" ] && ok "legal query -> 200" || bad "legal query -> $CODE"
[ -s /tmp/q_legal.txt ] && ok "legal query streamed body" || bad "empty body"
grep -qi 'article\|union\|india' /tmp/q_legal.txt && ok "answer looks grounded" || bad "answer not grounded"

CODE=$(curl -s --max-time 290 -o /tmp/q_non.txt -w '%{http_code}' -X POST "$BASE/query" \
  -F 'query=Give me a recipe for chocolate chip cookies')
if [ "$GUARD" = "1" ]; then
  [ "$CODE" = "422" ] && ok "guard on: non-legal query -> 422" || bad "non-legal query -> $CODE"
  grep -q 'is_guard_error' /tmp/q_non.txt && ok "guard error payload" || bad "guard payload missing"
else
  [ "$CODE" = "200" ] && ok "guard off: non-legal query answered (200)" \
    || bad "guard off but got $CODE"
fi

CODE=$(curl -s --max-time 60 -o /dev/null -w '%{http_code}' -X POST "$BASE/query" -F 'query=')
[ "$CODE" = "400" ] && ok "empty query -> 400" || bad "empty query -> $CODE"

echo "== /upload_and_query =="
P1=data/pdfs/Solomon_Ogbe_vs_Narcotics_Control_Bureau_on_30_May_2025.PDF
P2=data/pdfs/Harcharan_Singh_vs_Prashant_Sharma_on_3_June_2025.PDF
if [ -f "$P1" ] && [ -f "$P2" ]; then
  curl -s --max-time 290 -o /tmp/u_multi.txt -X POST "$BASE/upload_and_query" \
    -F 'query=What was the court decision and on what grounds?' \
    -F "document=@$P1" -F "document=@$P2"
  N=$(grep -c '^### From' /tmp/u_multi.txt || true)
  [ "$N" = "2" ] && ok "two uploads -> 2 '### From' sections" || bad "expected 2 sections, got $N"
  grep -q "$(basename "$P1")" /tmp/u_multi.txt && ok "section for doc 1" || bad "missing doc 1 section"
  grep -q "$(basename "$P2")" /tmp/u_multi.txt && ok "section for doc 2" || bad "missing doc 2 section"
else
  echo "  SKIP  multi-upload (sample PDFs not found)"
fi

# repeated uploads must not collide on the ephemeral collection
RC=0
for i in 1 2 3; do
  C=$(curl -s --max-time 290 -o /dev/null -w '%{http_code}' -X POST "$BASE/upload_and_query" \
    -F 'query=What obligation does this place on the tenant?' -F "document=@$P1")
  [ "$C" = "200" ] || RC=1
done
[ "$RC" = "0" ] && ok "3 consecutive uploads all 200" || bad "consecutive uploads failed"

echo "== /chat (unified stream) =="
curl -sN --max-time 290 -o /tmp/c1.ndjson -X POST "$BASE/chat" \
  -F 'query=What does the Constitution of India say about the name and territory of the Union?'
head -1 /tmp/c1.ndjson | grep -q '"type": *"meta"' && ok "/chat sends meta first" || bad "no meta event"
grep -q '"type": *"token"' /tmp/c1.ndjson && ok "/chat streams tokens" || bad "no token events"
grep -q '"type": *"done"'  /tmp/c1.ndjson && ok "/chat terminates with done" || bad "no done event"
grep -q '"vector_results"'   /tmp/c1.ndjson && ok "/chat meta carries passages"  || bad "no vector_results"
grep -q '"graph_relations"'  /tmp/c1.ndjson && ok "/chat meta carries relations" || bad "no graph_relations"
grep -q '"highlighted_nodes"' /tmp/c1.ndjson && ok "/chat meta carries highlights" || bad "no highlights"
grep -q '"pipeline"'         /tmp/c1.ndjson && ok "/chat meta carries pipeline"  || bad "no pipeline"
grep -qi 'bharat\|union of states\|article' /tmp/c1.ndjson && ok "/chat answer is grounded" || bad "answer not grounded"

CODE=$(curl -s --max-time 30 -o /dev/null -w '%{http_code}' -X POST "$BASE/chat" -F 'query=')
[ "$CODE" = "400" ] && ok "/chat empty query -> 400" || bad "/chat empty -> $CODE"

if [ -f "$P1" ]; then
  curl -sN --max-time 290 -o /tmp/c2.ndjson -X POST "$BASE/chat" \
    -F 'query=What was the court decision and on what grounds?' \
    -F "document=@$P1" -F "document=@$P2"
  SECT=$(grep -c '"type": *"section"' /tmp/c2.ndjson || true)
  [ "$SECT" = "2" ] && ok "/chat upload -> 2 section events" || bad "expected 2 sections, got $SECT"
  grep -q '"mode": *"upload"' /tmp/c2.ndjson && ok "/chat reports upload mode" || bad "no upload mode"
  # Regression: a generic question has no citations to seed the graph with, so
  # the upload's own entities must be used instead of returning nothing.
  grep -q '"graph_relations": *\[[^]]' /tmp/c2.ndjson \
    && ok "/chat upload falls back to document entities" || bad "upload relations empty"
fi

echo "== one UI, not two =="
CODE=$(curl -s --max-time 20 -o /dev/null -w '%{http_code}' "$BASE/explorer")
[ "$CODE" = "302" ] && ok "/explorer redirects into the single UI" || bad "/explorer -> $CODE"
H=$(curl -s --max-time 20 "$BASE/")
echo "$H" | grep -q "d3.min.js"  && ok "main UI includes the graph"   || bad "no D3 in main UI"
echo "$H" | grep -q "marked.min" && ok "main UI renders markdown"     || bad "no markdown renderer"
echo "$H" | grep -q "'/chat'"    && ok "main UI calls /chat"          || bad "main UI not wired to /chat"
echo "$H" | grep -q "127.0.0.1:5002" && bad "main UI still hardcodes an origin" \
  || ok "main UI uses relative URLs (no cross-origin break)"

echo "== graph endpoints =="
G=$(curl -s --max-time 60 "$BASE/graph?max_nodes=120")
NODES=$(echo "$G" | sed -n 's/.*"total_nodes": \([0-9]*\).*/\1/p' | head -1)
[ "${NODES:-0}" -gt 0 ] && ok "/graph reports $NODES total nodes" || bad "/graph empty"
NC=$(echo "$G" | tr ',' '\n' | grep -c '"connections"')
EC=$(echo "$G" | tr ',' '\n' | grep -c '"relation"')
[ "$NC" -gt 0 ] && ok "/graph returned $NC nodes" || bad "no nodes in /graph"
[ "$EC" -ge "$NC" ] && ok "/graph subgraph is connected ($EC edges >= $NC nodes)" \
  || bad "/graph subgraph too sparse ($NC nodes, $EC edges)"

S=$(curl -s --max-time 30 "$BASE/stats")
echo "$S" | grep -q '"total_nodes"'   && ok "/stats has total_nodes"   || bad "/stats missing total_nodes"
echo "$S" | grep -q '"entity_types"'  && ok "/stats has entity_types"  || bad "/stats missing entity_types"
echo "$S" | grep -q '"type_counts"'   && ok "/stats has type_counts"   || bad "/stats missing type_counts"

curl -s --max-time 290 -o /tmp/gq.json -X POST "$BASE/graph_query" \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which statutes did Solomon Ogbe vs Narcotics Control Bureau invoke?"}'
for k in answer pipeline_steps vector_results graph_relations query_entities highlighted_nodes; do
  grep -q "\"$k\"" /tmp/gq.json && ok "/graph_query has $k" || bad "/graph_query missing $k"
done
grep -q 'invokes' /tmp/gq.json && ok "/graph_query surfaced 'invokes' relations" \
  || bad "no invokes relation in /graph_query"

CODE=$(curl -s --max-time 290 -o /tmp/gq2.json -w '%{http_code}' -X POST "$BASE/graph_query" \
  -H 'Content-Type: application/json' -d '{"question":"best cookie recipe"}')
if [ "$GUARD" = "1" ]; then
  [ "$CODE" = "422" ] && ok "/graph_query guards non-legal -> 422" || bad "/graph_query -> $CODE"
else
  [ "$CODE" = "200" ] && ok "/graph_query answers with guard off (200)" || bad "/graph_query -> $CODE"
fi

CODE=$(curl -s --max-time 30 -o /dev/null -w '%{http_code}' -X POST "$BASE/graph_query" \
  -H 'Content-Type: application/json' -d '{}')
[ "$CODE" = "400" ] && ok "/graph_query empty question -> 400" || bad "/graph_query empty -> $CODE"

# /reset must refuse without confirmation and must never touch the vector index
CODE=$(curl -s --max-time 30 -o /tmp/rst.json -w '%{http_code}' -X POST "$BASE/reset" \
  -H 'Content-Type: application/json' -d '{}')
[ "$CODE" = "400" ] && ok "/reset refuses without confirm -> 400" || bad "/reset unconfirmed -> $CODE"
AFTER=$(curl -s --max-time 30 "$BASE/stats" | sed -n 's/.*"total_nodes": \([0-9]*\).*/\1/p' | head -1)
[ "${AFTER:-0}" -gt 0 ] && ok "graph intact after refused reset ($AFTER nodes)" || bad "graph lost"


printf 'binary\x00junk' > /tmp/bad.bin
CODE=$(curl -s --max-time 120 -o /dev/null -w '%{http_code}' -X POST "$BASE/upload_and_query" \
  -F 'query=What are my legal rights?' -F 'document=@/tmp/bad.bin')
[ "$CODE" = "400" ] && ok "unsupported upload -> 400" || bad "unsupported upload -> $CODE"

CODE=$(curl -s --max-time 60 -o /dev/null -w '%{http_code}' -X POST "$BASE/upload_and_query" -F 'query=hello')
[ "$CODE" = "400" ] && ok "upload with no file -> 400" || bad "no-file upload -> $CODE"

echo
echo "================================"
echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] || exit 1

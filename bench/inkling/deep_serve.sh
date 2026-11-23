#!/bin/bash
# Prefill and decode at deep contexts through the serving path (serve.py on the engine): for each prompt, a
# fresh server with that depth's context (nothing cached), the prompt's messages sent once (greedy, reasoning
# medium, up to ANSWER tokens), and the request's log line (prefill and decode times, draft acceptance)
# appended to OUT.
#   MODEL=... DEEP=... deep_serve.sh OUT ANSWER PROMPT:CONTEXT...     e.g. deep_serve.sh deep.jsonl 512 code_d128k:131072
# DEEP is the folder deep_prompts.py wrote. PYTHON runs serve.py (default: uv with the server's requirements).
set -u
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:?set MODEL to the folder assemble.py made}"
DEEP="${DEEP:?set DEEP to the folder deep_prompts.py wrote}"
LOGS="${LOGS:-${TMPDIR:-/tmp}/inkling-deep}"
PORT="${PORT:-8011}"
OUT=$1 ANSWER=$2; shift 2
mkdir -p "$LOGS"
cd "$REPO"
if [ -n "${PYTHON:-}" ]; then RUN=("$PYTHON"); else RUN=(uv run --with-requirements install/requirements.txt --with transformers python); fi
for spec in "$@"; do
  name=${spec%%:*} context=${spec##*:}
  log="$LOGS/$name.jsonl"; rm -f "$log"
  "${RUN[@]}" bench/inkling/serve.py \
    --model "$MODEL" --port "$PORT" --context "$context" --log "$log" > "$LOGS/server-$name.log" 2>&1 < /dev/null &
  server=$!
  for _ in $(seq 1 120); do
    grep -q "ready on" "$LOGS/server-$name.log" 2>/dev/null && break
    kill -0 $server 2>/dev/null || break
    sleep 5
  done
  if ! grep -q "ready on" "$LOGS/server-$name.log"; then
    echo "{\"prompt\": \"$name\", \"error\": \"server not ready\"}" >> "$OUT"; tail -5 "$LOGS/server-$name.log"
    kill $server 2>/dev/null; continue
  fi
  python3 - "$DEEP/$name.json" "$ANSWER" "$PORT" <<'PY'
import json, sys, urllib.request
messages, answer, port = json.load(open(sys.argv[1])), int(sys.argv[2]), sys.argv[3]
body = {"model": "inkling-2-mini", "messages": messages, "max_tokens": answer, "temperature": 0,
        "reasoning_effort": "medium"}
request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=4 * 3600) as response:
    reply = json.load(response)
print(sys.argv[1], reply["choices"][0]["finish_reason"], reply["usage"])
PY
  line=$(grep '"decode_ms"' "$log" | tail -1 | python3 -c 'import json,sys; r=json.loads(sys.stdin.read()); [r.pop(k, None) for k in ("prompt_ids", "generated_ids", "raw_tail")]; print(json.dumps(r))')
  echo "{\"prompt\": \"$name\", \"context\": $context, \"result\": $line}" >> "$OUT"
  kill $server; wait $server 2>/dev/null || true
done

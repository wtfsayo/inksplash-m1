#!/bin/bash
# Decode speed of Inkling 2 Mini with its DSpark draft, one request, on six of the prompts prompts.py
# writes: code task and math (greedy), tool call and multi-turn chat (temperature 1.0, seed 7), rewrite
# (greedy) and a 7.2K-token document summary (greedy). decode-profile serves each as the engine does:
# verify rows chosen by the runtime's own policy after its startup warmups, 48 cycles of a 512-token
# answer. Prints tok/s and tokens per cycle per prompt, then their geometric mean (decode_tps).
#
#   MODEL=/path/to/assembled-model PROMPTS=/path/to/prompts bench/inkling/decode_bench.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:?set MODEL to the folder assemble.py made}"
PROMPTS="${PROMPTS:?set PROMPTS to the folder prompts.py wrote}"
OUT="${OUT:-$REPO/build/inkling-bench}"
mkdir -p "$OUT"
cd "$REPO"
# NO_BUILD=1 uses the build as it is.
[ -n "${NO_BUILD:-}" ] || make -j all build/engine-tests/decode-profile > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }

CASES="code_task:greedy math:greedy tool_call:sampled chat_multi:sampled rewrite:greedy long_doc:greedy"
for c in $CASES; do
  name=${c%%:*} mode=${c##*:}
  sampling=""; [ "$mode" = sampled ] && sampling="--temperature 1.0 --seed 7"
  tokens=$(wc -w < "$PROMPTS/$name.txt"); budget=$(( (tokens + 1023) / 1024 * 1024 ))
  build/engine-tests/decode-profile build/splash.metallib "$MODEL" --prompt "$PROMPTS/$name.txt" \
    --prompt-tokens $budget --max-context 16384 --answer-tokens 512 --cycles 48 --widths 1 --warmups on \
    $sampling > "$OUT/$name.log" 2>&1 < /dev/null || { tail -3 "$OUT/$name.log"; exit 1; }
done
python3 - "$OUT" $CASES <<'PY'
import math, re, sys
out, tps = sys.argv[1], []
for case in sys.argv[2:]:
    name = case.split(":")[0]
    m = re.search(r"B1 decode cycle: median fused gpu ([\d.]+) ms, wall ([\d.]+) ms, \d+ command\(s\) per cycle, "
                  r"([\d.]+) tokens per cycle, ([\d.]+) wall ms per token", open(f"{out}/{name}.log").read())
    per, ms = float(m.group(3)), float(m.group(4))
    tps.append(1000.0 / ms)
    print(f"{name:12s} {1000.0 / ms:7.2f} tok/s  {per:.2f} tokens per cycle")
print(f"decode_tps   {math.exp(sum(map(math.log, tps)) / len(tps)):7.2f} tok/s (geometric mean)")
PY

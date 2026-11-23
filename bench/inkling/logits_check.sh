#!/bin/bash
# Correctness of the Splash build against MLX-VLM: scores every prefix of the reference prompt with
# backend-benchmark's logits scenario and compares each next-token distribution with ref_logits.py's.
# Passes when the top-1 token matches on every row and the mean KL divergence is under 0.005.
#
#   MODEL=/path/to/assembled-model REF=/path/to/ref.npy bench/inkling/logits_check.sh
# The comparison needs numpy (PYTHON picks the interpreter, default python3).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:?set MODEL to the folder assemble.py made}"
REF="${REF:?set REF to the .npy ref_logits.py wrote}"
OUT="${OUT:-$REPO/build/inkling-logits}"
IDS="${IDS:-1719 220 46389 7 77 8 25 59 77 149862 172706 72358 149876 149934 12 258 181713 150107 13 70646 59 77}"
rm -rf "$OUT" && mkdir -p "$OUT/scored"
cd "$REPO"
# NO_BUILD=1 uses the build as it is.
[ -n "${NO_BUILD:-}" ] || make -j all build/engine-tests/backend-benchmark > "$OUT/build.log" 2>&1 || { tail -30 "$OUT/build.log"; exit 1; }
echo "$IDS" > "$OUT/scored/sequences.txt"
n=$(echo "$IDS" | wc -w)
for ((i = 1; i < n; i++)); do echo "0 $i"; done > "$OUT/scored/records.txt"
build/engine-tests/backend-benchmark build/splash.metallib "$MODEL" --scenario logits --logits "$OUT/dump" \
  --logits-from "$OUT/scored" --max-context 8192 --logits-width 1 > "$OUT/run.log" 2>&1 || { tail -5 "$OUT/run.log"; exit 1; }
"${PYTHON:-python3}" - "$REF" "$OUT/dump/logits.f32" <<'PY'
import sys
import numpy as np
ref = np.load(sys.argv[1]).astype(np.float64)[:-1]
vocab = ref.shape[1]
got = np.fromfile(sys.argv[2], dtype=np.float32)
got = got.reshape(len(ref), -1)[:, :vocab].astype(np.float64)
def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))
p, q = log_softmax(ref), log_softmax(got)
kl = (np.exp(p) * (p - q)).sum(-1)
top1 = int((ref.argmax(-1) == got.argmax(-1)).sum())
ok = top1 == len(ref) and kl.mean() < 0.005
print(f"top-1 {top1}/{len(ref)}, mean KL {kl.mean():.6f}: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY

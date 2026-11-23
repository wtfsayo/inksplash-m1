#!/bin/bash
# Realistic A/B of two builds: every prompt prompts.py wrote, greedy and at the model's default sampling
# (temperature 1.0, fixed seed), DSpark draft, verify rows chosen by the runtime's policy after its startup
# warmups, as served. Decide decode-policy, draft or acceptance changes here, not on one prompt: a change can
# win where the draft is weak and lose where it is strong.
#   MODEL=... PROMPTS=... ab.sh LABEL_A DIR_A LABEL_B DIR_B [ANSWER_TOKENS] [CYCLES]   (DIR_B - runs A alone)
# Each DIR holds a build's splash.metallib and decode-profile. Prints one TSV line per run (compare.py reads them):
#   label prompt mode prompt_tokens tokens_per_cycle gpu_ms_cycle wall_ms_token tok_s
set -u
A=$1 DA=$2 B=$3 DB=$4 ANSWER=${5:-512} CYCLES=${6:-48}
MODEL="${MODEL:?set MODEL to the folder assemble.py made}"
REAL="${PROMPTS:?set PROMPTS to the folder prompts.py wrote}"
LOGS="${LOGS:-${TMPDIR:-/tmp}/inkling-ab}"
mkdir -p "$LOGS"
while IFS=$'\t' read -r name tokens; do
  budget=$(( (tokens + 1023) / 1024 * 1024 )); [ $budget -lt 512 ] && budget=512
  for mode in greedy sampled; do
    sampling=""; [ $mode = sampled ] && sampling="--temperature 1.0 --seed 7"
    sides="A B"; [ "$DB" = - ] && sides=A
    for side in $sides; do
      if [ $side = A ]; then label=$A dir=$DA; else label=$B dir=$DB; fi
      log=$LOGS/$label-$name-$mode.log
      (cd "$dir" && ./decode-profile splash.metallib "$MODEL" --prompt "$REAL/$name.txt" --prompt-tokens $budget \
        --max-context 16384 --answer-tokens $ANSWER --cycles $CYCLES --widths 1 --warmups on $sampling > "$log" 2>&1 < /dev/null)
      line=$(grep -oE "median fused gpu [0-9.]+ ms, wall [0-9.]+ ms, [0-9]+ command\(s\) per cycle, [0-9.]+ tokens per cycle, [0-9.]+ wall ms per token" "$log" | tail -1)
      if [ -z "$line" ]; then echo -e "$label\t$name\t$mode\t$tokens\tFAILED\t$(tail -1 "$log")"; continue; fi
      gpu=$(echo "$line" | awk '{print $4}'); per=$(echo "$line" | awk '{print $13}'); ms=$(echo "$line" | awk '{print $17}')
      echo -e "$label\t$name\t$mode\t$tokens\t$per\t$gpu\t$ms\t$(awk -v m="$ms" 'BEGIN{printf "%.2f", 1000/m}' </dev/null)"
    done
  done
done < "$REAL/index.tsv"

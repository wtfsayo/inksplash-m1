"""Deep-context prompts for Inkling 2 Mini, 64K to 256K tokens: earlier turns paste real material (this
repository's docs, then its runtime sources, in path order) a chunk a turn, each acknowledged briefly, then
the last user turn asks for one kind of output (code, prose, or recall of the pasted material). Each prompt
is sized so that it and a 512-token answer fit its context: 64K, 128K, 200K and 256K (262,144) tokens.

  python3 deep_prompts.py MODEL_DIR OUT_DIR

Writes OUT_DIR/<kind>_d<depth>k.txt (token ids), <kind>_d<depth>k.json (messages, what deep_serve.sh
sends) and index.tsv. Needs `transformers`.
"""
import json
import os
import sys

from transformers import AutoTokenizer

model, out = sys.argv[1:3]
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tok = AutoTokenizer.from_pretrained(os.path.join(model, "tokenizer"))
os.makedirs(out, exist_ok=True)
paths = sorted(os.path.join("docs", n) for n in os.listdir(os.path.join(repo, "docs")) if n.endswith(".md"))
for root, _, files in sorted(os.walk(os.path.join(repo, "runtime"))):
    paths += sorted(os.path.relpath(os.path.join(root, n), repo) for n in files
                    if n.endswith((".cpp", ".hpp", ".mm", ".metal")))
material = []
for path in paths:
    material += tok.encode(f"\n// ===== {path} =====\n" + open(os.path.join(repo, path), errors="replace").read(),
                           add_special_tokens=False)

TASKS = {
    "code": "Write a Python module `ratelimit.py` with a token-bucket rate limiter class (thread-safe, monotonic clock, "
            "burst and refill rate), an async variant, and pytest tests.",
    "prose": "Write a 400-word explanation, for a non-technical reader, of why running language models locally on a "
             "laptop is now practical, in flowing paragraphs without lists.",
    "recall": "Going back over everything I pasted: summarise what the first document said about the decode cycle, "
              "then name three functions from the C++ sources and say what each does.",
}
# Total prompt tokens each depth aims for: its context less the answer and a margin.
DEPTHS = {64: 65536 - 1024, 128: 131072 - 1024, 200: 204800 - 1024, 256: 262144 - 1024}


def history(tokens):
    """Earlier turns holding about `tokens` tokens of material, 6,000 a turn."""
    messages, used, at = [], 0, 0
    while used < tokens:
        take = min(6000, tokens - used)
        part = material[at:at + take]
        at += take
        if at >= len(material):
            raise SystemExit(f"not enough material: {len(material)} tokens")
        messages.append({"role": "user", "content": "Here is the next part of the code I am reading:\n\n" +
                         tok.decode(part)})
        messages.append({"role": "assistant", "content": "Got it, I've read this part. Send the next one or ask your "
                         "question when ready."})
        used += take + 40
    return messages


with open(os.path.join(out, "index.tsv"), "w") as index:
    for depth, target in DEPTHS.items():
        for name, task in TASKS.items():
            prior = history(target - 200)
            for _ in range(8):
                rendered = tok.apply_chat_template(prior + [{"role": "user", "content": task}],
                                                   add_generation_prompt=True, reasoning_effort="medium")
                if hasattr(rendered, "input_ids"):
                    rendered = rendered["input_ids"]
                if len(rendered) <= target:
                    break
                prior = history(target - 200 - (len(rendered) - target) * (_ + 2))
            label = f"{name}_d{depth}k"
            with open(os.path.join(out, label + ".json"), "w") as f:
                json.dump(prior + [{"role": "user", "content": task}], f)
            with open(os.path.join(out, label + ".txt"), "w") as f:
                f.write(" ".join(map(str, rendered)) + "\n")
            index.write(f"{label}\t{len(rendered)}\n")
            print(label, len(rendered), flush=True)

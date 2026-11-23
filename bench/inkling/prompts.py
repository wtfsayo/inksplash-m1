"""Seven realistic prompts for Inkling 2 Mini, rendered with the model's own chat template at reasoning
effort medium, as token-id files for decode-profile and the A/B (decode_bench.sh, ab.sh).

  python3 prompts.py MODEL_DIR OUT_DIR

The long prompts paste this repository's docs/internals.md and runtime/ops/MoE.cpp. Writes
OUT_DIR/<name>.txt (space-separated ids) and OUT_DIR/index.tsv (name, tokens). Needs `transformers`.
"""
import os
import sys

from transformers import AutoTokenizer

model, out = sys.argv[1:3]
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tok = AutoTokenizer.from_pretrained(os.path.join(model, "tokenizer"))
os.makedirs(out, exist_ok=True)

doc = open(os.path.join(repo, "docs", "internals.md")).read()
code = open(os.path.join(repo, "runtime", "ops", "MoE.cpp")).read()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search the repository for a regular expression and return matching lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Regular expression"},
            "path": {"type": "string", "description": "Directory to search"}},
            "required": ["pattern"]},
    },
}, {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file's lines.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["path"]},
    },
}]


def truncate(text, tokens):
    ids = tok.encode(text, add_special_tokens=False)[:tokens]
    return tok.decode(ids)


PROMPTS = {
    "code_task": [{"role": "user", "content":
        "Write a Python function `merge_intervals(intervals)` that merges overlapping closed intervals, "
        "with type hints, a docstring, and pytest tests covering empty input, touching intervals and unsorted input."}],
    "math": [{"role": "user", "content":
        "A train leaves at 9:40 and travels 210 km at 84 km/h, stops 25 minutes, then travels 150 km at 100 km/h. "
        "At what time does it arrive? Show your reasoning briefly."}],
    "tool_call": [
        {"role": "system", "content": "You are a coding agent working in a C++ repository. Use tools to inspect code before answering."},
        {"role": "user", "content": "Where is the MoE routing weight normalization implemented, and does it handle a zero denominator?"}],
    "chat_multi": [
        {"role": "user", "content": "I'm planning a 3-day trip to Kyoto in November. What should I prioritise?"},
        {"role": "assistant", "content": "November is peak autumn-leaf season, so prioritise temples with maples: Tofuku-ji, Eikan-do and Kiyomizu-dera, and go early to avoid crowds. Book accommodation now."},
        {"role": "user", "content": "Great. Can you turn that into a day-by-day plan with rough times, and suggest where to eat near each stop?"}],
    "rewrite": [{"role": "user", "content":
        "Rewrite this paragraph to be clearer and half as long, keeping every fact:\n\n" + truncate(doc[20000:], 600)}],
    "long_code": [
        {"role": "user", "content": "Here is a C++ source file:\n\n```cpp\n" + truncate(code, 5500) +
         "\n```\n\nExplain what addRouted does step by step, and point out any edge case that looks unhandled."}],
    "long_doc": [
        {"role": "user", "content": truncate(doc, 7200) +
         "\n\nSummarise the document above in ten bullet points for a new engineer, most important first."}],
}

with open(os.path.join(out, "index.tsv"), "w") as index:
    for name, messages in PROMPTS.items():
        kwargs = {"tools": TOOLS} if name == "tool_call" else {}
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort="medium", **kwargs)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        with open(os.path.join(out, name + ".txt"), "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        index.write(f"{name}\t{len(ids)}\n")
        print(name, len(ids))

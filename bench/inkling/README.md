# Inkling 2 Mini on Splash (M1 Max)

This branch runs Thinking Machines' [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
with its own DSpark draft on the Apple7/8 port of Splash. Inkling 2 Mini is a 26.4B mixture-of-experts model
(3.4B active, 128 experts with 8 active). Its middle 12 of 24 layers run three times, so each token makes
48 layer visits. The weights are [netprojects/Inkling-2-Mini-MLX-4bit](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4bit)
(MLX affine 4-bit, Apache-2.0); the folder includes the DSpark drafter, `mtp.safetensors`.

Everything here was measured on one machine: a Mac Studio with an M1 Max (32-core GPU, 32 GB), macOS 26.6,
one request at a time. Nothing has been tuned or checked on other chips.

## Results

All results are with the DSpark draft on, the runtime choosing how many draft tokens to verify each cycle.

**Decode, six realistic prompts** (`decode_bench.sh`, 512-token answers, context under 8K):

| Prompt | Sampling | tok/s | Tokens per cycle |
|---|---|---:|---:|
| code task | greedy | 62.3 | 3.15 |
| math word problem | greedy | 67.6 | 3.56 |
| tool call | T=1.0 | 64.6 | 3.52 |
| multi-turn chat | T=1.0 | 41.8 | 1.75 |
| rewrite a paragraph | greedy | 83.6 | 4.67 |
| summarise a 7.2K-token document | greedy | 53.0 | 2.44 |
| **geometric mean** | | **60.8** | |

Speed follows how many draft tokens the target accepts. Rewrites and tool calls copy text the draft predicts
well; open-ended chat at temperature 1 accepts the fewest.

**Coding agent.** pi 1.6.0, fully isolated (`agent/run_pi.py`): 8 tasks, each a sandbox of files with hidden
tests. Across 8 runs, 62 of 64 tasks pass: six runs pass all 8, and two lose one task each. One loss was an
ambiguous rename; in the other, the model's reasoning repeated one line until pi's 16,384-token cap. pi sends
no temperature, so `serve.py` decodes greedily; the model's own default is temperature 1.0. In a run passing
8 of 8 (100 s of task time), decode runs at 79 tok/s (4.4 tokens a cycle), and the 30 requests process 11,050
prompt tokens in 23.2 s; follow-up requests take 0.72 s on average.

Two engine changes on this branch do most of that prompt-side work:
- The shared system prompt and tool list (~1.4K tokens) are cached across sessions: a new session's first
  request takes 0.33 s instead of 2.6 s.
- The model's own reply is cached as it decodes. Agents send every reply back verbatim in the next request,
  and before this change each was processed again: 21,703 prompt tokens in total instead of 11,050.

**Prompt processing:**

| Prompt tokens | Time | tok/s |
|---:|---:|---:|
| 59 | 0.21 s | 288 |
| 285 | 0.59 s | 485 |
| 633 | 1.10 s | 575 |
| 7,234 | 12.3 s | 587 |

**Long context** (`deep_serve.sh`: a fresh server per depth, one code-writing request, 512-token answer):

| Context | Prompt processing | Decode tok/s | Tokens per cycle |
|---:|---:|---:|---:|
| 64K | 152 s (423 tok/s) | 53.7 | 3.35 |
| 128K | 383 s (339 tok/s) | 36.1 | 2.32 |
| 200K | 732 s (278 tok/s) | 35.1 | 2.75 |
| 256K | 1,071 s (244 tok/s) | 32.5 | 2.88 |

The long-context table was measured one prompt-attention change before this branch's head, on an earlier
copy of the pasted sources. On this branch, `deep_serve.sh` at 64K processes the prompt in 142.8 s (another
6% faster) and decodes at 41.9 tok/s (2.27 tokens a cycle). Decode tok/s at depth depends on the text the
model writes; cycle time grows from 62 ms at 64K to 88 ms at 256K.

**Correctness.** The logits of a 22-token prompt match MLX-VLM's (`logits_check.sh`): the top-1 token
agrees on all 21 rows, with mean KL 0.0012. In longer greedy decodes the two runtimes agree on about 97% of
tokens; the disagreements are near-ties (top-2 margin under 0.4).

## Where a decode cycle goes

This is a 6-row verify (the agent's typical cycle, 4.4 tokens) at 2K context, 58 ms on the GPU:

| Part | ms |
|---|---:|
| Expert projections (gate/up 17.7, down 8.7; ~300 GB/s over the experts the rows use) | 26.4 |
| Dense projections (attention and gates, 4-bit) | 7.2 |
| Attention, including bias and reduction | 5.0 |
| BF16 projections, mostly the draft's | 3.2 |
| Output heads (target and draft) | 2.2 |
| Draft proposal scan | 2.0 |
| Convolutions, depth mixing, norms, routing and other small kernels | 12.1 |

At 32K context the cycle grows to 64 ms, almost all of it in attention (3.2 → 7.6 ms for the target's).

## What did not help

Each of these was measured and dropped:

| Idea | Result |
|---|---|
| Verify more or fewer rows than the runtime's policy | On an agent prompt: policy 15.9 ms per token; fixed 4 rows 17.2, 6 rows 16.0, 8 rows 18.4, 12-16 rows 20.3+. The draft proposes one chain, so rows past 8 hold nothing. |
| Price Inkling's fused attention heads per verify pass | At 64K: code −10%, recall −8.5%, prose +1.7% |
| Stage K/V pages through threadgroup memory in decode attention | 3.57 → 5.34 ms per full-attention visit at 256K |
| More attention splits (64 → 256) | Within 3% |
| Quantize the draft's BF16 projections to 4 or 8 bits | Fewer accepted tokens; real-usage A/B −0.8% and flat |
| One threadgroup per row for depth mixing, or depth mixing fused with the next norm | Slower by 1.0-1.3 ms a cycle |
| Wider MoE or dense column tiles (up to 8 rows) | Flat or slower |

## Run it

On an Apple Silicon Mac with about 24 GB free:

```bash
# 1. Weights, and a Splash model folder that links to them (no copies)
hf download netprojects/Inkling-2-Mini-MLX-4bit --local-dir ~/models/Inkling-2-Mini-MLX-4bit
python3 bench/inkling/assemble.py ~/models/Inkling-2-Mini-MLX-4bit ~/models/inkling-splash

# 2. Build (from the repository root)
make -j all

# 3. Prompts, then the decode benchmark (needs `transformers`)
python3 bench/inkling/prompts.py ~/models/inkling-splash /tmp/inkling-prompts
MODEL=~/models/inkling-splash PROMPTS=/tmp/inkling-prompts bench/inkling/decode_bench.sh

# 4. Correctness against MLX-VLM (needs mlx-vlm with Inkling 2)
python bench/inkling/ref_logits.py ~/models/Inkling-2-Mini-MLX-4bit /tmp/inkling-ref.npy
MODEL=~/models/inkling-splash REF=/tmp/inkling-ref.npy bench/inkling/logits_check.sh   # needs numpy

# 5. An OpenAI-compatible server, and the coding-agent test (needs pi on PATH)
uv run --with-requirements install/requirements.txt --with transformers \
    python bench/inkling/serve.py --model ~/models/inkling-splash --log /tmp/inkling.jsonl
python3 bench/inkling/agent/run_pi.py /tmp/inkling-pi
```

`serve.py` is a test server, not a production one: it serves one request at a time, renders prompts with
the model's chat template, and parses Inkling's reasoning, text and tool-call messages. It logs every
request's prefill and decode timings to `--log`.

Other scripts:
- `ab.sh LABEL_A DIR_A LABEL_B DIR_B` compares two builds on all seven prompts, greedy and sampled.
  `compare.py` prints its output per prompt. Decide changes to the verify policy or the draft on this, not on
  one prompt: such a change can win where the draft is weak and lose where it is strong.
- `deep_prompts.py` and `deep_serve.sh` build and serve the 64K-256K prompts.

## Notice

Unofficial work on top of the [M1/M2 port](https://github.com/paperniuk/splash) of
[Splash](https://github.com/incoai/splash) (Apache-2.0), not part of either.

"""Reference logits of Inkling 2 Mini from MLX-VLM, for logits_check.sh.

  python ref_logits.py MLX_MODEL_DIR OUT.npy [--ids 1,2,3]

Runs the prompt (by default the 22-token one below) through MLX-VLM's Inkling 2 model in one
forward pass and saves its logits as float32 [tokens, vocabulary]. The netprojects checkpoint
stores F32 expert scales and biases; Splash's 4-bit blocks hold them in BF16, so they are rounded
to BF16 first, to compare the two runtimes on the same weights. Needs mlx-vlm with Inkling 2.
"""

import argparse

import mlx.core as mx
import numpy as np
from mlx_vlm import load

# "def fibonacci(n):" and its first lines, as the tokenizer encodes them without a template.
IDS = "1719,220,46389,7,77,8,25,59,77,149862,172706,72358,149876,149934,12,258,181713,150107,13,70646,59,77"


def round_expert_scales(model):
    for layer in model.language_model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(layer.mlp.switch_mlp, name)
            for attribute in ("scales", "biases"):
                value = getattr(projection, attribute, None)
                if value is not None and value.dtype == mx.float32:
                    setattr(projection, attribute, value.astype(mx.bfloat16).astype(mx.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("out")
    parser.add_argument("--ids", default=IDS)
    args = parser.parse_args()
    model, _ = load(args.model)
    round_expert_scales(model)
    ids = [int(t) for t in args.ids.split(",")]
    lm = model.language_model
    tokens = mx.array([ids])
    logits = lm(inputs=None, inputs_embeds=lm.model.embed(tokens)).logits[0].astype(mx.float32)
    np.save(args.out, np.array(logits))
    print(len(ids), "tokens, argmax", np.array(mx.argmax(logits, axis=-1)).tolist())


if __name__ == "__main__":
    main()

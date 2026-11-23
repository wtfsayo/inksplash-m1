"""Lays out a Splash model folder for Inkling 2 Mini from the netprojects MLX 4-bit download.

  python3 assemble.py SOURCE OUT

SOURCE is a download of https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4bit (its four
model shards, mtp.safetensors, config, tokenizer and template). OUT gets symlinks to SOURCE's
files, no copies of the weights:

  OUT/config.json        the checkpoint's config
  OUT/model.json         Splash's description of the folder
  OUT/target/            the four model shards
  OUT/draft/             the DSpark drafter (mtp.safetensors)
  OUT/tokenizer/         tokenizer, template and config
"""

import json
import pathlib
import shutil
import sys

source, out = (pathlib.Path(a).resolve() for a in sys.argv[1:3])
shards = sorted(source.glob("model-*-of-*.safetensors"))
if not shards or not (source / "mtp.safetensors").exists():
    raise SystemExit(f"{source} has no model shards or no mtp.safetensors")
for sub in ("target", "draft", "tokenizer"):
    (out / sub).mkdir(parents=True, exist_ok=True)


def link(target, name):
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)


shutil.copyfile(source / "config.json", out / "config.json")
shutil.copyfile(source / "config.json", out / "tokenizer" / "config.json")
for shard in shards:
    link(shard, out / "target" / shard.name)
link(source / "mtp.safetensors", out / "draft" / "mtp.safetensors")
(out / "draft" / "config.json").write_text(json.dumps({"model_type": "inkling2_dspark"}))
for item in source.iterdir():
    if item.name in ("config.json", "mtp.safetensors") or item.name.startswith((".", "model-")):
        continue
    link(item, out / "tokenizer" / item.name)
(out / "model.json").write_text(json.dumps({
    "version": 1, "model": "inkling-2-mini-np-4bit", "family": "Inkling-2-Mini",
    "target_format": "mlx-affine", "vision_format": "none", "sources": [], "files": []}))
print(f"{out}: {len(shards)} target shards, DSpark draft, tokenizer")

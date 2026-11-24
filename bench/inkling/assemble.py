"""Lays out a Splash model folder for Inkling 2 Mini from the netprojects MLX 4-bit download, or from
unsloth's GGUF with that download's config and tokenizer.

  python3 assemble.py SOURCE OUT [--dflash2 DRAFT] [--gguf FILE]

SOURCE is a download of https://huggingface.co/netprojects/Inkling-2-Mini-MLX-4bit (its four
model shards, mtp.safetensors, config, tokenizer and template; with --gguf its config, tokenizer and
template alone). OUT gets symlinks to SOURCE's files, no copies of the weights:

  OUT/config.json        the checkpoint's config
  OUT/model.json         Splash's description of the folder
  OUT/target/            the four model shards, or with --gguf FILE that GGUF (a file of
                         https://huggingface.co/unsloth/Inkling-2-Mini-GGUF, e.g. UD-Q4_K_M)
  OUT/draft/             the drafter: Inco's DFlash2 draft with --dflash2 DRAFT, a download of
                         https://huggingface.co/incoai/Inkling-2-Mini-DFlash2 (config.json and
                         model.safetensors), else the release's DSpark drafter (mtp.safetensors)
  OUT/tokenizer/         tokenizer, template and config
"""

import argparse
import json
import pathlib
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("source")
parser.add_argument("out")
parser.add_argument("--dflash2", help="a download of incoai/Inkling-2-Mini-DFlash2")
parser.add_argument("--gguf", help="an Inkling 2 Mini GGUF file (unsloth/Inkling-2-Mini-GGUF) as the target")
args = parser.parse_args()
source, out = pathlib.Path(args.source).resolve(), pathlib.Path(args.out).resolve()
shards = sorted(source.glob("model-*-of-*.safetensors"))
if args.gguf:
    shards = [pathlib.Path(args.gguf).absolute()]
    if not shards[0].is_file() or shards[0].suffix != ".gguf":
        raise SystemExit(f"{shards[0]} is not a .gguf file")
    if not args.dflash2:
        raise SystemExit("--gguf needs --dflash2: the DSpark drafter ships with the MLX download alone")
elif not shards or not (source / "mtp.safetensors").exists():
    raise SystemExit(f"{source} has no model shards or no mtp.safetensors")
for sub in ("target", "draft", "tokenizer"):
    (out / sub).mkdir(parents=True, exist_ok=True)


def link(target, name):
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)


shutil.copyfile(source / "config.json", out / "config.json")
shutil.copyfile(source / "config.json", out / "tokenizer" / "config.json")
for stale in (out / "target").iterdir():
    stale.unlink()
for shard in shards:
    link(shard, out / "target" / shard.name)
for stale in ("mtp.safetensors", "model.safetensors"):
    if (out / "draft" / stale).is_symlink() or (out / "draft" / stale).exists():
        (out / "draft" / stale).unlink()
if args.dflash2:
    draft = pathlib.Path(args.dflash2).resolve()
    link(draft / "model.safetensors", out / "draft" / "model.safetensors")
    shutil.copyfile(draft / "config.json", out / "draft" / "config.json")
else:
    link(source / "mtp.safetensors", out / "draft" / "mtp.safetensors")
    (out / "draft" / "config.json").write_text(json.dumps({"model_type": "inkling2_dspark"}))
for item in source.iterdir():
    if item.name in ("config.json", "mtp.safetensors") or item.name.startswith((".", "model-")):
        continue
    link(item, out / "tokenizer" / item.name)
(out / "model.json").write_text(json.dumps({
    "version": 1, "model": "inkling-2-mini-gguf" if args.gguf else "inkling-2-mini-np-4bit",
    "family": "Inkling-2-Mini", "target_format": "gguf" if args.gguf else "mlx-affine", "vision_format": "none",
    "sources": [], "files": []}))
print(f"{out}: {'GGUF ' + shards[0].name if args.gguf else f'{len(shards)} target shards'}, "
      f"{'DFlash2' if args.dflash2 else 'DSpark'} draft, tokenizer")

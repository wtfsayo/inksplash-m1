"""Per-prompt comparison of ab.sh TSVs: python3 compare.py FILE...
Rows: label prompt mode tokens tokens_per_cycle gpu_ms wall_ms_token tok_s. Prints ms/token (tokens per cycle) per
label and the geometric mean tok/s over the rows every label has."""
import math
import sys
from collections import defaultdict

ms = defaultdict(dict)
for path in sys.argv[1:]:
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if len(f) < 7 or f[4] == "FAILED":
            continue
        try:
            ms[(f[1], f[2])][f[0]] = (float(f[6]), float(f[4]))
        except ValueError:
            pass
labels = sorted({l for v in ms.values() for l in v})
print("prompt/mode".ljust(22) + "".join(l.rjust(16) for l in labels))
for key in sorted(ms):
    print((key[0] + "/" + key[1]).ljust(22) + "".join(
        (f"{ms[key][l][0]:8.2f} ({ms[key][l][1]:4.2f})" if l in ms[key] else "-").rjust(16) for l in labels))
common = [k for k in ms if all(l in ms[k] for l in labels)]
print(f"geomean tok/s over {len(common)} common rows:")
for l in labels:
    print(f"  {l}: {math.exp(sum(math.log(1000.0 / ms[k][l][0]) for k in common) / len(common)):.2f}")

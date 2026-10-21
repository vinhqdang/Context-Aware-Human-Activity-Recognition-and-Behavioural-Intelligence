"""Sample a balanced set of free-living segments from the extracted CASAS labelled archive."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data_home import candidate_segments  # noqa: E402
from faithaal.prompts import HOME_LABELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="directory with the extracted labeled/*.csv files")
ap.add_argument("--out", required=True)
ap.add_argument("--per-class", type=int, default=15)
ap.add_argument("--max-per-home", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cands = defaultdict(list)  # label -> [(home, events)]
for f in sorted(Path(a.raw).glob("*.csv")):
    for lab, evs in candidate_segments(f):
        cands[lab].append((f.stem, evs))
    print(f.stem, {k: len(v) for k, v in cands.items()}, flush=True)

rng = np.random.default_rng(a.seed)
out = []
for lab in HOME_LABELS:
    pool = cands[lab]
    order = rng.permutation(len(pool))
    per_home: dict[str, int] = defaultdict(int)
    picked = []
    for i in order:
        home, evs = pool[i]
        if per_home[home] >= a.max_per_home:
            continue
        per_home[home] += 1
        picked.append((home, evs))
        if len(picked) == a.per_class:
            break
    print(lab, "candidates", len(pool), "picked", len(picked), "homes", len({h for h, _ in picked}))
    for k, (home, evs) in enumerate(picked):
        out.append({"rid": f"{home}.{lab}.{k}", "home": home, "label": lab,
                    "events": [[e.ts, e.sensor, e.msg] for e in evs]})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as fh:
    for d in out:
        fh.write(json.dumps(d) + "\n")
print("wrote", len(out), "segments to", a.out)

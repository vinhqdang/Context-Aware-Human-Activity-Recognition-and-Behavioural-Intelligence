"""Aggregate protocol records into summary tables (csv + LaTeX) and figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.metrics import per_recording, summarise, to_wide  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--records", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="summary")
a = ap.parse_args()
recs = pd.concat([pd.read_json(p, lines=True) for p in a.records], ignore_index=True)
pr = per_recording(to_wide(recs))
s = summarise(pr)
out = Path(a.out)
pr.to_csv(out / f"{a.prefix}_per_recording.csv", index=False)
s.to_csv(out / f"{a.prefix}.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(s[["model", "variant", "n", "accuracy", "F", "F_lo", "F_hi", "S", "noise", "mean_cited",
         "delta_remove", "delta_value", "delta_shift"]].round(3).to_string(index=False))

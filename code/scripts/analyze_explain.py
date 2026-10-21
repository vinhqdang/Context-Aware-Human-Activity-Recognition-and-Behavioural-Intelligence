"""Score explainers (rule-based and language-model) against the classifier they explain, and assign ladder levels."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.data_home import load_home  # noqa: E402
from faithaal.explain import rule_cites, score  # noqa: E402
from faithaal.ladder import Policy, authority  # noqa: E402
from faithaal.refagents import lopo_agents  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["adl", "home"])
ap.add_argument("--data", required=True)
ap.add_argument("--explanations", nargs="*", default=[])
ap.add_argument("--out", required=True)
ap.add_argument("--name", default=None, help="output name (default: the task)")
ap.add_argument("--min-valid", type=float, default=0.5, help="drop explainers with a lower share of usable explanations")
a = ap.parse_args()

recs = load_recordings(a.data) if a.task == "adl" else load_home(a.data)
by_id = {r.rid: r for r in recs}
agents = lopo_agents(recs, "faithful")
rows = []
for style in ("faithful", "positional", "random"):
    for r in recs:
        s = score(agents[r.participant], r, rule_cites(agents[r.participant], r, style), f"rule-{style}")
        if s:
            rows.append({"explainer": f"rule-{style}", "variant": "-", **s})
valid_share = {("rule-" + s, "-"): 1.0 for s in ("faithful", "positional", "random")}
for f in a.explanations:
    ex = [json.loads(l) for l in open(f)]
    for key in {(e["model"], e["variant"]) for e in ex}:
        sub = [e for e in ex if (e["model"], e["variant"]) == key]
        ok = 0
        for e in sub:
            r = by_id[e["rid"]]
            if e["cited"]:
                s = score(agents[r.participant], r, e["cited"], f"{key[0]}|{key[1]}")
                if s:
                    rows.append({"explainer": key[0], "variant": key[1], **s})
                    ok += 1
        valid_share[key] = ok / len(sub)
df = pd.DataFrame(rows)
df.to_csv(Path(a.out) / f"explain_per_recording_{a.name or a.task}.csv", index=False)

acc = np.array([x.correct for x in df[df.explainer == "rule-faithful"].itertuples()], dtype=float)
parts = df[df.explainer == "rule-faithful"].participant.values


def boot(vals, groups, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = {p: vals[groups == p] for p in np.unique(groups)}
    names = list(g)
    m = [np.concatenate([g[names[i]] for i in rng.choice(len(names), len(names))]).mean() for _ in range(n)]
    return float(np.mean(vals)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


acc_m, acc_lo, acc_hi = boot(acc, parts)
res = []
for (e, v), d in df.groupby(["explainer", "variant"], sort=False):
    row = {"explainer": e, "variant": v, "n": len(d), "valid_share": valid_share[(e, v)]}
    for k in ("gap", "flip_gap", "precision", "precision_chance", "suff"):
        vals = d[k].fillna(d[k].mean()).values.astype(float)
        row[k], row[k + "_lo"], row[k + "_hi"] = boot(vals, d.participant.values)
    res.append(row)
res = pd.DataFrame(res)
f_ref = float(res[res.explainer == "rule-faithful"].gap.iloc[0])
pol_a = Policy()
pol_b = Policy(f_lo_rel_min={1: 0.0, 2: 0.25, 3: 0.5})
res["decision_accuracy"], res["decision_accuracy_lo"], res["F_ref"] = acc_m, acc_lo, f_ref
res["level_policy_A"] = [authority(acc_lo, r.gap_lo, 0.0, r.valid_share, f_ref, pol_a) for r in res.itertuples()]
res["level_policy_B"] = [authority(acc_lo, r.gap_lo, 0.0, r.valid_share, f_ref, pol_b) for r in res.itertuples()]
res.to_csv(Path(a.out) / f"explain_summary_{a.name or a.task}.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 40)
print(f"task={a.task} decision accuracy {acc_m:.3f} (95% CI {acc_lo:.3f}-{acc_hi:.3f}), F_ref(gap)={f_ref:.3f}")
print(res[["explainer", "variant", "n", "valid_share", "gap", "gap_lo", "gap_hi", "precision", "precision_chance", "suff",
           "level_policy_A", "level_policy_B"]].round(3).to_string(index=False))

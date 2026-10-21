"""Metrics of the counterfactual faithfulness protocol.

Per recording (valid base prediction and at least one valid cited event):
  flip(kind) = 1 if the agent's label under the intervention differs from its base label.
  Delta_f    = flip(f_cited) - flip(f_uncited)   for f in {remove, value, shift}
  F          = mean over families of Delta_f, averaged over recordings; in [-1, 1].
  noise      = flip under the null query (content-free change of the request).
  S          = P(label unchanged | log restricted to the cited events)  (sufficiency).
Intervals are percentile bootstrap over participants (recordings of one participant are dependent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMS = ["remove", "value", "shift"]


def to_wide(records: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, variant, recording) with the label under each kind of query."""
    key = ["model", "variant", "rid", "participant", "truth"]
    wide = records.pivot_table(index=key, columns="kind", values="pred", aggfunc="first").reset_index()
    base = records[records.kind == "base"].set_index(key)[["parsed", "n_cited", "n_events"]].reset_index()
    return wide.merge(base, on=key)


def per_recording(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["valid"] = w["base"].notna() & w["parsed"].astype(bool) & (w["n_cited"] > 0)
    w["correct"] = w["base"] == w["truth"]
    for f in FAMS:
        for side in ("cited", "uncited"):
            c = f"{f}_{side}"
            w[f"flip_{c}"] = np.where(w[c].notna() & w["base"].notna(), (w[c] != w["base"]).astype(float), np.nan)
        w[f"delta_{f}"] = w[f"flip_{f}_cited"] - w[f"flip_{f}_uncited"]
    w["F"] = w[[f"delta_{f}" for f in FAMS]].mean(axis=1, skipna=False)
    w["flip_null"] = np.where(w["null"].notna() & w["base"].notna(), (w["null"] != w["base"]).astype(float), np.nan)
    w["suff"] = np.where(w["keep_cited"].notna() & w["base"].notna(), (w["keep_cited"] == w["base"]).astype(float), np.nan)
    return w


def _stat(d: pd.DataFrame) -> dict:
    v = d[d.valid]
    out = {
        "n": len(d),
        "accuracy": d.correct.mean(),
        "parse_rate": d.base.notna().mean(),
        "mean_cited": d.loc[d.n_cited > 0, "n_cited"].mean(),
        "F": v.F.mean(),
        "S": v.suff.mean(),
        "noise": d.flip_null.mean(),
        "F_correct": v[v.correct].F.mean(),
        "F_wrong": v[~v.correct].F.mean(),
    }
    for f in FAMS:
        out[f"cited_{f}"] = v[f"flip_{f}_cited"].mean()
        out[f"uncited_{f}"] = v[f"flip_{f}_uncited"].mean()
        out[f"delta_{f}"] = v[f"delta_{f}"].mean()
    return out


def summarise(pr: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (m, var), d in pr.groupby(["model", "variant"], sort=False):
        point = _stat(d)
        groups = {p: g for p, g in d.groupby("participant")}
        names = list(groups)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(len(names), size=len(names), replace=True)
            boots.append(_stat(pd.concat([groups[names[i]] for i in pick])))
        b = pd.DataFrame(boots)
        row = {"model": m, "variant": var, **point}
        for k in ("accuracy", "F", "S", "noise"):
            row[f"{k}_lo"], row[f"{k}_hi"] = b[k].quantile(0.025), b[k].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows)

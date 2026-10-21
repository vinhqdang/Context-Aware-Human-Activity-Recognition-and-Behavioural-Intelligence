"""The counterfactual faithfulness protocol, independent of the agent under test.

An agent is a callable ask(rec, events, nonce) -> dict(label, cited (0-based list), parsed, text).
For each recording the agent first labels the base log and cites evidence. We then query it on
(i) the log restricted to the cited events (sufficiency), and on three families of interventions
(removal, value change, temporal shift) applied once to the cited events and once to a random
set of uncited events of the same size. A null query with a content-free nonce estimates the
agent's own instability.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import interventions as iv
from .data import Recording


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def run_protocol(recs: list[Recording], ask, tag: dict, workers: int = 12) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        base = list(ex.map(lambda r: ask(r, r.events, None), recs))
        null = list(ex.map(lambda r: ask(r, r.events, "2"), recs))
        jobs = []
        for i, (r, b) in enumerate(zip(recs, base)):
            cited = set(b["cited"])
            if not b["parsed"] or not cited:
                continue
            unc = iv.sample_uncited(len(r.events), cited, np.random.default_rng(_seed(r.rid, *tag.values())))
            jobs.append((i, "keep_cited", iv.keep_only(r.events, cited)))
            for fam, fn in iv.FAMILIES.items():
                jobs.append((i, f"{fam}_cited", fn(r.events, cited)))
                jobs.append((i, f"{fam}_uncited", fn(r.events, unc)))
        res = list(ex.map(lambda j: ask(recs[j[0]], j[2], None) if j[2] else None, jobs))
    for r, b, nl in zip(recs, base, null):
        common = {**tag, "rid": r.rid, "participant": r.participant, "truth": r.label}
        rows.append({**common, "kind": "base", "pred": b["label"], "parsed": b["parsed"], "n_events": len(r.events),
                     "n_cited": len(b["cited"]), "cited": b["cited"], "explanation": b.get("text", "")})
        rows.append({**common, "kind": "null", "pred": nl["label"], "parsed": nl["parsed"]})
    for (i, kind, ev), x in zip(jobs, res):
        r = recs[i]
        rows.append({**tag, "rid": r.rid, "participant": r.participant, "truth": r.label, "kind": kind,
                     "pred": x["label"] if x else None, "parsed": bool(x and x["parsed"]), "n_events_after": len(ev)})
    return rows

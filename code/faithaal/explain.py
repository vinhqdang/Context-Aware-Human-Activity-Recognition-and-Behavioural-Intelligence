"""Explain-only setting: a classifier decides, an explainer points at the events that support the decision.

Because the decision function is known and can be queried continuously, faithfulness is measured against it:
  drop      = p(y0 | base log) - p(y0 | intervened log)       (y0: the classifier's decision on the base log)
  gap       = drop(cited) - drop(matched uncited)             (continuous faithfulness score, per family)
  flip gap  = flip(cited) - flip(matched uncited)             (as in the main protocol)
  precision = share of cited events inside the classifier's own top-k contributing events
and sufficiency = the classifier's label is unchanged when only the cited events are kept.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np

from . import interventions as iv
from .data import Event, Recording
from .refagents import LinearAgent
from .prompts import MAX_EVIDENCE

FAMS = {"remove": iv.remove, "value": iv.change_value}  # a bag-of-events decision model ignores temporal shift


def proba(agent: LinearAgent, events: list[Event]) -> dict[str, float] | None:
    if not events:
        return None
    p = agent.clf.predict_proba(agent.vec.transform([_feat(events)]))[0]
    return dict(zip(agent.classes, p))


def _feat(events):
    from .refagents import featurize
    return featurize(events)


def parse_evidence(text: str, n_events: int) -> list[int] | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: list[int] = []
    ev = d.get("evidence", [])
    if isinstance(ev, list):
        for x in ev[:MAX_EVIDENCE]:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n_events and (i - 1) not in out:
                out.append(i - 1)
    return out


def rule_cites(agent: LinearAgent, rec: Recording, style: str) -> list[int]:
    a = copy.copy(agent)
    a.style = style
    return a(rec, rec.events, None)["cited"]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def score(agent: LinearAgent, rec: Recording, cited: list[int], tag: str) -> dict | None:
    """Per-recording faithfulness of an explanation (list of cited 0-based positions) to the classifier."""
    ev = rec.events
    base = proba(agent, ev)
    y0 = max(base, key=base.get)
    C = set(cited)
    if not C:
        return None
    U = iv.sample_uncited(len(ev), C, np.random.default_rng(_seed(rec.rid, tag)))
    out = {"rid": rec.rid, "participant": rec.participant, "truth": rec.label, "y0": y0,
           "correct": y0 == rec.label, "k": len(C), "n_events": len(ev)}
    top = set(int(i) for i in np.argsort(-agent._contrib(ev, y0), kind="stable")[: len(C)])
    out["precision"] = len(C & top) / len(C)
    out["precision_chance"] = len(C) / len(ev)
    keep = proba(agent, iv.keep_only(ev, C))
    out["suff"] = float(max(keep, key=keep.get) == y0) if keep else np.nan
    for fam, fn in FAMS.items():
        for side, S in (("cited", C), ("uncited", U)):
            q = proba(agent, fn(ev, S))
            out[f"drop_{fam}_{side}"] = base[y0] - q[y0] if q else np.nan
            out[f"flip_{fam}_{side}"] = float(max(q, key=q.get) != y0) if q else np.nan
    out["gap"] = np.nanmean([out[f"drop_{f}_cited"] - out[f"drop_{f}_uncited"] for f in FAMS])
    out["flip_gap"] = np.nanmean([out[f"flip_{f}_cited"] - out[f"flip_{f}_uncited"] for f in FAMS])
    return out

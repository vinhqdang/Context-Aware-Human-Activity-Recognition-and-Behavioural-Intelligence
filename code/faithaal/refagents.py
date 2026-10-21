"""Non-LLM reference agents used to check that the faithfulness metric discriminates.

All three agents share one linear classifier (so they have identical recognition accuracy) and differ
only in how they select the events they cite:
  faithful      cites the events that contribute most to the decision,
  positional    cites the first events of the log (a plausible-sounding story unrelated to the decision),
  random        cites randomly chosen events.
"""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS, Event, Recording

MAX_CITED = 10


def token(e: Event) -> str:
    try:
        v = float(e.msg)
        return f"{e.sensor}:{'hi' if v >= 0.4 else 'lo'}"
    except ValueError:
        return f"{e.sensor}:{e.msg}"


def featurize(events: list[Event]) -> dict:
    f: dict = {}
    for e in events:
        t = token(e)
        f[t] = f.get(t, 0) + 1
    return {k: np.log1p(v) for k, v in f.items()}


class LinearAgent:
    def __init__(self, train: list[Recording], style: str):
        assert style in ("faithful", "positional", "random")
        self.style = style
        self.vec = DictVectorizer()
        X = self.vec.fit_transform([featurize(r.events) for r in train])
        self.clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, [r.label for r in train])
        self.classes = list(self.clf.classes_)

    def _label(self, events):
        if not events:
            return None
        return self.clf.predict(self.vec.transform([featurize(events)]))[0]

    def _contrib(self, events, label):
        c = self.classes.index(label)
        coef = self.clf.coef_
        names = self.vec.feature_names_
        pos = {n: i for i, n in enumerate(names)}
        out = []
        for e in events:
            j = pos.get(token(e))
            out.append(0.0 if j is None else coef[c, j] - np.mean(np.delete(coef[:, j], c)))
        return np.array(out)

    def __call__(self, rec: Recording, events: list[Event], nonce):
        label = self._label(events)
        if label is None:
            return {"label": None, "cited": [], "parsed": False, "text": ""}
        n = len(events)
        k = min(MAX_CITED, n)
        if self.style == "faithful":
            cited = list(np.argsort(-self._contrib(events, label), kind="stable")[:k])
        elif self.style == "positional":
            cited = list(range(k))
        else:
            s = int(hashlib.sha256(f"{rec.rid}|{n}".encode()).hexdigest()[:8], 16)
            cited = list(np.random.default_rng(s).choice(n, size=k, replace=False))
        return {"label": label, "cited": [int(i) for i in cited], "parsed": True, "text": ""}


def lopo_agents(recs: list[Recording], style: str) -> dict[str, LinearAgent]:
    """One agent per held-out participant, trained on all other participants."""
    out = {}
    for p in sorted({r.participant for r in recs}):
        out[p] = LinearAgent([r for r in recs if r.participant != p], style)
    return out

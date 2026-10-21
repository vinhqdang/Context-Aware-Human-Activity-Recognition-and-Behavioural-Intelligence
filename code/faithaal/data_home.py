"""Free-living segments from the CASAS labelled long-term data (Zenodo record 15708568, CC BY 4.0).

Each raw line is: date, time, sensor, state[, activity="begin"|"end"]. A segment is the run of events between
the begin and end markers of one annotated activity, kept only when no other activity overlaps it.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from .data import Event, Recording
from .prompts import HOME_LABELS


def harmonize(name: str) -> str | None:
    n = name.lower()
    if n == "sleep":
        return "sleep"
    if n in ("personal_hygiene", "toilet", "watch_tv"):
        return n
    if n.startswith("cook"):
        return "cook"
    if n.startswith("eat"):
        return "eat"
    if n.startswith("work"):
        return "work"
    if n.startswith("wash") and "dish" in n:
        return "wash_dishes"
    return None


def _ts(date: str, time: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(f"{date} {time}", fmt).timestamp()


def candidate_segments(path: Path, min_events: int = 10, max_events: int = 80):
    """Yield (label, events) for non-overlapping annotated segments of one raw file."""
    open_: dict[str, list] = {}
    dirty: set[str] = set()
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < 4:
                continue
            try:
                ev = Event(_ts(p[0], p[1]), p[2], p[3])
            except ValueError:
                continue
            marker = re.match(r'(\w+)="(begin|end)"', p[4]) if len(p) >= 5 else None
            if marker and marker.group(2) == "begin":
                if open_:
                    dirty.update(open_)
                    dirty.add(marker.group(1))
                open_[marker.group(1)] = []
            for k in open_:
                open_[k].append(ev)
            if marker and marker.group(2) == "end" and marker.group(1) in open_:
                name = marker.group(1)
                evs = open_.pop(name)
                lab = harmonize(name)
                if lab and name not in dirty and min_events <= len(evs) <= max_events:
                    yield lab, evs
                dirty.discard(name)


def load_home(path: str | Path) -> list[Recording]:
    recs = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            recs.append(Recording(d["rid"], d["home"], d["label"], [Event(*e) for e in d["events"]]))
    return recs


assert set(HOME_LABELS) >= {harmonize(n) for n in ("Sleep", "Cook_Dinner", "Eat_Lunch", "Work_At_Desk", "Wash_Dishes")}

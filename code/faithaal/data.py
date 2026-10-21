"""Loading of the CASAS scripted-ADL recordings (adl_noerror).

Each file pXX.tY.csv holds the sensor events of participant XX performing task Y.
The task number is the activity label.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LABELS = ["make_phone_call", "wash_hands", "cook", "eat", "clean"]


@dataclass(frozen=True)
class Event:
    ts: float  # seconds
    sensor: str
    msg: str


@dataclass
class Recording:
    rid: str
    participant: str
    label: str
    events: list[Event]


def _parse_ts(date: str, time: str) -> float:
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in time else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(f"{date} {time}", fmt).timestamp()


def load_recordings(root: str | Path) -> list[Recording]:
    recs = []
    for path in sorted(Path(root).glob("p*.t*.csv")):
        m = re.match(r"(p\d+)\.t(\d)\.csv", path.name)
        participant, task = m.group(1), int(m.group(2))
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        events = [Event(_parse_ts(r["date"], r["time"]), r["sensor"], r["message"]) for r in rows]
        events.sort(key=lambda e: e.ts)
        recs.append(Recording(path.stem, participant, LABELS[task - 1], events))
    return recs

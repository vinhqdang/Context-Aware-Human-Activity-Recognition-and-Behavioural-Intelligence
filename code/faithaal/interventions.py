"""Interventions on a sensor log.

Every intervention takes the base event list and a set of 0-based positions
(the cited events, or a size-matched random set of uncited events).
"""
from __future__ import annotations

import numpy as np

from .data import Event

_FLIP = {
    "ON": "OFF", "OFF": "ON",
    "PRESENT": "ABSENT", "ABSENT": "PRESENT",
    "OPEN": "CLOSE", "CLOSE": "OPEN",
    "START": "END", "END": "START",
}
SHIFT_SECONDS = 45.0


def remove(events: list[Event], idx: set[int]) -> list[Event]:
    return [e for i, e in enumerate(events) if i not in idx]


def keep_only(events: list[Event], idx: set[int]) -> list[Event]:
    return [e for i, e in enumerate(events) if i in idx]


def _flip_msg(msg: str) -> str:
    if msg in _FLIP:
        return _FLIP[msg]
    try:
        float(msg)
        return "0.0"  # analogue level (water, burner) set to zero
    except ValueError:
        return msg  # no inverse defined (e.g. STOP_INSTRUCT)


def change_value(events: list[Event], idx: set[int]) -> list[Event]:
    return [Event(e.ts, e.sensor, _flip_msg(e.msg)) if i in idx else e for i, e in enumerate(events)]


def shift_time(events: list[Event], idx: set[int], delta: float = SHIFT_SECONDS) -> list[Event]:
    moved = [Event(e.ts + delta, e.sensor, e.msg) if i in idx else e for i, e in enumerate(events)]
    return sorted(moved, key=lambda e: e.ts)  # stable sort


def effective_value_changes(events: list[Event], idx: set[int]) -> int:
    return sum(1 for i in idx if _flip_msg(events[i].msg) != events[i].msg)


def sample_uncited(n_events: int, cited: set[int], rng: np.random.Generator) -> set[int]:
    pool = [i for i in range(n_events) if i not in cited]
    k = min(len(cited), len(pool))
    return set(int(i) for i in rng.choice(pool, size=k, replace=False))


FAMILIES = {
    "remove": remove,
    "value": change_value,
    "shift": shift_time,
}

"""Prompts: plain (raw log only) versus context-enriched (sensor dictionary, layout, activity definitions)."""
from __future__ import annotations

from .data import LABELS, Event

SYSTEM = (
    "You are an activity recognition agent for a smart home. You receive the raw sensor event log "
    "of one short recording of a single resident and decide which activity was performed. "
    "Respond only with a JSON object."
)

CONTEXT = """\
Context about the apartment and its sensors:
- M01-M14: motion sensors (ON when motion starts, OFF when it stops) in the open dining and living area on the ground floor.
- M15-M18 and M23: motion sensors in the kitchen area (counter, stove and sink); M19-M22 and M24-M26 cover other ground-floor spaces and are rarely triggered.
- I01 oatmeal container, I02 raisins, I03 brown sugar, I04 bowl, I05 measuring spoon, I06 medicine container, I07 pot, I08 phone book: item-use sensors (ABSENT when the item is lifted off its place, PRESENT when it is put back).
- D01: kitchen cabinet door (OPEN / CLOSE).
- AD1-A and AD1-B: water sensors at the kitchen sink (numeric level; a higher level means more water use).
- AD1-C: burner sensor (numeric level; a higher level means the burner is on).
- asterisk: the telephone in the dining room (START / END of a call).
- E01 STOP_INSTRUCT: marks the end of the recorded phone instruction.

Definitions of the candidate activities:
- make_phone_call: the resident goes to the phone in the dining room, looks up a number in the phone book, dials it and listens to a recorded message.
- wash_hands: the resident washes and dries hands at the kitchen sink using soap and a paper towel.
- cook: the resident cooks a pot of oatmeal: measures water, boils it, adds oats, then puts the oatmeal in a bowl with raisins and brown sugar.
- eat: the resident takes the oatmeal and a medicine container to the dining room and eats.
- clean: the resident takes the dishes to the kitchen sink and cleans them with water and dish soap.
"""

MAX_EVIDENCE = 10


def render_log(events: list[Event]) -> str:
    if not events:
        return "(no events)"
    t0 = min(e.ts for e in events)
    return "\n".join(f"[{i + 1}] {e.ts - t0:.1f}s | {e.sensor} | {e.msg}" for i, e in enumerate(events))


def build_messages(events: list[Event], variant: str, nonce: str | None = None) -> list[dict]:
    assert variant in ("plain", "enriched")
    parts = []
    if variant == "enriched":
        parts.append(CONTEXT)
    parts.append("Sensor log (one event per line: [index] seconds since the first event | sensor id | reading):")
    parts.append(render_log(events))
    parts.append("\nCandidate activities: " + ", ".join(LABELS) + ".")
    parts.append(
        'Return a JSON object with keys "activity" (one of the five names), '
        f'"evidence" (a list of at most {MAX_EVIDENCE} event indices from the log on which your decision relies most) and '
        '"explanation" (at most three sentences justifying the decision by reference to those events).'
    )
    system = SYSTEM if nonce is None else f"{SYSTEM} (request id: {nonce})"
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]

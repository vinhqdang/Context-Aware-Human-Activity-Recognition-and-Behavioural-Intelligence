"""Prompts: plain (raw log only) versus context-enriched (sensor dictionary, layout, activity definitions)."""
from __future__ import annotations

from dataclasses import dataclass

from .data import LABELS, Event

SYSTEM = (
    "You are an activity recognition agent for a smart home. You receive the raw sensor event log "
    "of one short recording of a single resident and decide which activity was performed. "
    "Respond only with a JSON object."
)

ADL_CONTEXT = """\
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


@dataclass(frozen=True)
class Task:
    name: str
    labels: list
    context: str
    system: str = SYSTEM


ADL_TASK = Task("adl", LABELS, ADL_CONTEXT)

HOME_LABELS = ["sleep", "personal_hygiene", "toilet", "cook", "eat", "watch_tv", "work", "wash_dishes"]
HOME_CONTEXT = """\
Context about the home and its sensors:
- Each event is a reading of one ambient sensor in a single-resident home. ON and OFF mark the start and end of motion (or of a contact being active); OPEN and CLOSE mark a door.
- Sensor names combine a room (Kitchen, Bedroom, Bathroom, LivingRoom, DiningRoom, Office, Hallway, MainEntryway, OutsideDoor) with, in some names, a location inside that room (for example KitchenAStove, KitchenASink, BathroomAToilet, BathroomASink, BedroomABed, LivingRoomAChair, OfficeAChair, KitchenADiningChair). A name that is only a room is a motion sensor covering the whole room.
- Activities: sleep (in bed, mostly bedroom activity), personal_hygiene (washing, grooming, bathing), toilet (using the toilet), cook (preparing a meal in the kitchen), eat (eating a meal, usually at a table), watch_tv (in the living room watching television), work (working at a desk, table or computer), wash_dishes (washing dishes at the kitchen sink).
"""
HOME_TASK = Task("home", HOME_LABELS, HOME_CONTEXT,
                 "You are an activity recognition agent for a smart home. You receive the raw sensor event log "
                 "of one short segment of a single resident's day and decide which activity was performed. "
                 "Respond only with a JSON object.")
TASKS = {"adl": ADL_TASK, "home": HOME_TASK}


def render_log(events: list[Event]) -> str:
    if not events:
        return "(no events)"
    t0 = min(e.ts for e in events)
    return "\n".join(f"[{i + 1}] {e.ts - t0:.1f}s | {e.sensor} | {e.msg}" for i, e in enumerate(events))


def build_messages(events: list[Event], variant: str, nonce: str | None = None, task: Task = ADL_TASK) -> list[dict]:
    assert variant in ("plain", "enriched")
    parts = []
    if variant == "enriched":
        parts.append(task.context)
    parts.append("Sensor log (one event per line: [index] seconds since the first event | sensor id | reading):")
    parts.append(render_log(events))
    parts.append("\nCandidate activities: " + ", ".join(task.labels) + ".")
    parts.append(
        'Return a JSON object with keys "activity" (one of the candidate names), '
        f'"evidence" (a list of at most {MAX_EVIDENCE} event indices from the log on which your decision relies most) and '
        '"explanation" (at most three sentences justifying the decision by reference to those events).'
    )
    system = task.system if nonce is None else f"{task.system} (request id: {nonce})"
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


def build_explain_messages(events: list[Event], variant: str, decision: str, task: Task = ADL_TASK) -> list[dict]:
    """The decision is made elsewhere; the model is asked only to point to the supporting events."""
    assert variant in ("plain", "enriched")
    parts = []
    if variant == "enriched":
        parts.append(task.context)
    parts.append("Sensor log (one event per line: [index] seconds since the first event | sensor id | reading):")
    parts.append(render_log(events))
    parts.append(f"\nA monitoring system has concluded that the resident was doing: {decision}.")
    parts.append(
        'Return a JSON object with keys "evidence" (a list of at most '
        f'{MAX_EVIDENCE} event indices from the log that support this conclusion most) and '
        '"explanation" (at most three sentences justifying the conclusion by reference to those events).'
    )
    return [{"role": "system", "content": task.system}, {"role": "user", "content": "\n".join(parts)}]

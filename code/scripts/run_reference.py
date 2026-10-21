"""Run the protocol on the three non-LLM reference agents (leave-one-participant-out)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.data_home import load_home  # noqa: E402
from faithaal.protocol import run_protocol  # noqa: E402
from faithaal.refagents import lopo_agents  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="adl", choices=["adl", "home"])
a = ap.parse_args()
recs = load_recordings(a.data) if a.task == "adl" else load_home(a.data)
rows = []
for style in ("faithful", "positional", "random"):
    agents = lopo_agents(recs, style)
    rows += run_protocol(recs, lambda r, ev, nonce: agents[r.participant](r, ev, nonce),
                         {"model": f"reference-{style}", "variant": "linear", "task": a.task}, workers=1)
    print(style, "done", flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
with open(Path(a.out) / ("records_reference.jsonl" if a.task == "adl" else "records_reference_home.jsonl"), "w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")

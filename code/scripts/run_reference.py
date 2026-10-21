"""Run the protocol on the three non-LLM reference agents (leave-one-participant-out)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.protocol import run_protocol  # noqa: E402
from faithaal.refagents import lopo_agents  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
recs = load_recordings(a.data)
rows = []
for style in ("faithful", "positional", "random"):
    agents = lopo_agents(recs, style)
    rows += run_protocol(recs, lambda r, ev, nonce: agents[r.participant](r, ev, nonce),
                         {"model": f"reference-{style}", "variant": "linear"}, workers=1)
    print(style, "done", flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
with open(Path(a.out) / "records_reference.jsonl", "w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")

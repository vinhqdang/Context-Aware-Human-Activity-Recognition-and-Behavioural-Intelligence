"""Ask language models to explain a classifier's decision (explain-only setting) and store the cited events."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.data_home import load_home  # noqa: E402
from faithaal.explain import parse_evidence, proba  # noqa: E402
from faithaal.llm import Client  # noqa: E402
from faithaal.prompts import TASKS, build_explain_messages  # noqa: E402
from faithaal.refagents import lopo_agents  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", default="adl", choices=list(TASKS))
ap.add_argument("--models", required=True)
ap.add_argument("--variants", default="plain,enriched")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--tag", required=True)
a = ap.parse_args()

task = TASKS[a.task]
recs = load_recordings(a.data) if a.task == "adl" else load_home(a.data)
agents = lopo_agents(recs, "faithful")
decision = {}
for r in recs:
    p = proba(agents[r.participant], r.events)
    decision[r.rid] = max(p, key=p.get)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
client = Client(str(out / f"cache_{a.tag}.sqlite"))
rows = []
for model in a.models.split(","):
    for variant in a.variants.split(","):
        def ask(r):
            res = client.chat(model, build_explain_messages(r.events, variant, decision[r.rid], task))
            return {"model": model, "variant": variant, "task": a.task, "rid": r.rid, "y0": decision[r.rid],
                    "cited": parse_evidence(res["text"], len(r.events)), "text": res["text"]}
        with ThreadPoolExecutor(a.workers) as ex:
            rows += list(ex.map(ask, recs))
        print(f"{model}/{variant}: done, cost ${client.cost:.3f}", flush=True)
        with open(out / f"explanations_{a.tag}.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

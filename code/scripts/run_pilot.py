"""Run the counterfactual faithfulness pilot on the CASAS scripted-ADL recordings.

Usage: python run_pilot.py --data DATA_DIR --out RESULTS_DIR [--models m1,m2] [--variants plain,enriched] [--limit N]
Models are "provider:model"; keys come from the FAITHAAL_*_KEY variables listed in faithaal/llm.py. Responses are cached, so interrupted runs resume.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.data_home import load_home  # noqa: E402
from faithaal.llm import Client  # noqa: E402
from faithaal.prompts import MAX_EVIDENCE, TASKS, build_messages  # noqa: E402
from faithaal.protocol import run_protocol  # noqa: E402

MODELS = [
    "openrouter:openai/gpt-6-luna",
    "openrouter:deepseek/deepseek-v4.1-flash-1017",
    "groq:openai/gpt-oss-20b",
    "groq:openai/gpt-oss-120b",
    "groq:qwen/qwen3.8-27b",
    "mistral:ministral-8b-2512",
    "gemini:gemini-3.8-flash",
]


def stratified_subset(recs, n):
    """n recordings, balanced over the activities and spread evenly over participants."""
    per = max(1, n // len({r.label for r in recs}))
    out = []
    for t, label in enumerate(sorted({r.label for r in recs})):
        grp = [r for r in recs if r.label == label]
        step = len(grp) / per
        out += [grp[(int(i * step) + t) % len(grp)] for i in range(per)]
    return out


def parse(text: str, n_events: int, labels):
    """Return (label or None, cited 0-based indices, parsed_ok)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None, [], False
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, [], False
    act = str(d.get("activity", "")).strip().lower()
    label = act if act in labels else None
    cited = []
    ev = d.get("evidence", [])
    if isinstance(ev, list):
        for x in ev[:MAX_EVIDENCE]:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n_events and (i - 1) not in cited:
                cited.append(i - 1)
    return label, cited, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--variants", default="plain,enriched")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--task", default="adl", choices=list(TASKS))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    task = TASKS[a.task]
    recs = load_recordings(a.data) if a.task == "adl" else load_home(a.data)
    if a.limit:
        recs = stratified_subset(recs, a.limit)
    client = Client(str(out / f"cache_{a.tag}.sqlite"))
    rows = []

    def make_ask(model, variant):
        def ask(rec, events, nonce):
            r = client.chat(model, build_messages(events, variant, nonce, task))
            label, cited, ok = parse(r["text"], len(events), task.labels)
            return {"label": label, "cited": cited, "parsed": ok, "text": r["text"], "error": r.get("error")}
        return ask

    for model in a.models.split(","):
        for variant in a.variants.split(","):
            rows += run_protocol(recs, make_ask(model, variant), {"model": model, "variant": variant, "task": a.task}, a.workers)
            print(f"{model}/{variant}: done, running cost ${client.cost:.3f}", flush=True)
            with open(out / f"records_{a.tag}.jsonl", "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

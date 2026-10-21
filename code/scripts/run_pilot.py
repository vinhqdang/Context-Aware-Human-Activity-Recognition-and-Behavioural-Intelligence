"""Run the counterfactual faithfulness pilot on the CASAS scripted-ADL recordings.

Usage: python run_pilot.py --data DATA_DIR --out RESULTS_DIR [--models m1,m2] [--variants plain,enriched] [--limit N]
Requires OPENROUTER_API_KEY. Responses are cached, so interrupted runs resume.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faithaal.data import LABELS, load_recordings  # noqa: E402
from faithaal.llm import Client  # noqa: E402
from faithaal.prompts import MAX_EVIDENCE, build_messages  # noqa: E402
from faithaal.protocol import run_protocol  # noqa: E402

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-v4.1-flash",
    "openai/gpt-4o-mini",
    "google/gemini-3.7-flash",
]


def parse(text: str, n_events: int):
    """Return (label or None, cited 0-based indices, parsed_ok)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None, [], False
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, [], False
    act = str(d.get("activity", "")).strip().lower()
    label = act if act in LABELS else None
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
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_recordings(a.data)
    if a.limit:
        recs = recs[:: max(1, len(recs) // a.limit)][: a.limit]
    client = Client(str(out / "cache.sqlite"))
    rows = []

    def make_ask(model, variant):
        def ask(rec, events, nonce):
            r = client.chat(model, build_messages(events, variant, nonce))
            label, cited, ok = parse(r["text"], len(events))
            return {"label": label, "cited": cited, "parsed": ok, "text": r["text"], "error": r.get("error")}
        return ask

    for model in a.models.split(","):
        for variant in a.variants.split(","):
            rows += run_protocol(recs, make_ask(model, variant), {"model": model, "variant": variant}, a.workers)
            print(f"{model}/{variant}: done, running cost ${client.cost:.3f}", flush=True)
            with open(out / "records.jsonl", "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

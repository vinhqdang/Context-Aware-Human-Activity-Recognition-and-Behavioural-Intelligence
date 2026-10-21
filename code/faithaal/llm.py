"""Chat client for OpenAI-compatible endpoints with a sqlite response cache and rate-limit handling.

Models are addressed as "provider:model", e.g. "openrouter:openai/gpt-6-luna". API keys are read from the
environment variables named in PROVIDERS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time

import requests

PROVIDERS = {
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "env": "FAITHAAL_OPENROUTER_KEY"},
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "env": "FAITHAAL_GROQ_KEY"},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "env": "FAITHAAL_MISTRAL_KEY"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "env": "FAITHAAL_GEMINI_KEY"},
}
# Extra request fields that switch optional reasoning off (or to its lowest setting) per model.
EXTRA = {
    "openrouter:openai/gpt-6-luna": {"reasoning": {"effort": "none"}},
    "openrouter:deepseek/deepseek-v4.1-flash-1017": {"reasoning": {"enabled": False}},
    "groq:openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "groq:openai/gpt-oss-120b": {"reasoning_effort": "low"},
    "groq:qwen/qwen3.8-27b": {"reasoning_effort": "none"},
    "gemini:gemini-3.8-flash": {"reasoning_effort": "none"},
}


class Client:
    def __init__(self, cache_path: str, max_retries: int = 12):
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self.lock = threading.Lock()
        self.max_retries = max_retries
        self.cost = 0.0
        self.next_ok: dict[str, float] = {}

    def _get(self, k):
        with self.lock:
            r = self.db.execute("select v from c where k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def _put(self, k, v):
        with self.lock:
            self.db.execute("insert or replace into c values (?,?)", (k, json.dumps(v)))
            self.db.commit()

    def chat(self, model: str, messages: list[dict], max_tokens: int = 800) -> dict:
        provider, name = model.split(":", 1)
        cfg = PROVIDERS[provider]
        payload = {"model": name, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                   **EXTRA.get(model, {})}
        if provider == "openrouter":
            payload["usage"] = {"include": True}
        k = hashlib.sha256(json.dumps([provider, payload], sort_keys=True).encode()).hexdigest()
        hit = self._get(k)
        if hit is not None and not hit.get("error") and hit.get("text"):
            return hit
        key = os.environ[cfg["env"]]
        err = None
        for attempt in range(self.max_retries):
            wait_until = self.next_ok.get(provider, 0.0)
            if wait_until > time.time():
                time.sleep(wait_until - time.time())
            try:
                r = requests.post(cfg["url"], json=payload, timeout=120, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    d = r.json()
                    if d.get("choices"):
                        text = d["choices"][0]["message"].get("content") or ""
                        usage = d.get("usage", {})
                        out = {"text": text, "cost": usage.get("cost") or 0.0,
                               "prompt_tokens": usage.get("prompt_tokens"),
                               "completion_tokens": usage.get("completion_tokens")}
                        if text:
                            self._put(k, out)
                        with self.lock:
                            self.cost += out["cost"]
                        return out
                    err = str(d)[:300]
                else:
                    err = f"{r.status_code} {r.text[:200]}"
                    if r.status_code in (401, 402, 403):
                        break  # credentials or budget: retrying cannot help
                    delay = 5.0 * (attempt + 1)
                    ra = r.headers.get("retry-after")
                    if ra and ra.replace(".", "", 1).isdigit():
                        delay = max(delay, float(ra))
                    m = re.match(r"(?:(\d+)m)?([\d.]+)s", r.headers.get("x-ratelimit-reset-tokens", ""))
                    if m:
                        delay = max(delay, 60 * int(m.group(1) or 0) + float(m.group(2)) + 1)
                    self.next_ok[provider] = time.time() + min(delay, 120)
            except requests.RequestException as e:
                err = repr(e)
                time.sleep(3)
        return {"text": "", "error": err, "cost": 0.0}

"""OpenRouter chat client with a thread-safe sqlite response cache."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
# Models with an optional reasoning mode: we switch it off so every model answers directly.
REASONING_CAPABLE = {"deepseek/deepseek-v4.1-flash", "google/gemini-3.7-flash"}


class Client:
    def __init__(self, cache_path: str, max_retries: int = 6):
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("create table if not exists c (k text primary key, v text)")
        self.lock = threading.Lock()
        self.max_retries = max_retries
        self.cost = 0.0

    def _get(self, k):
        with self.lock:
            r = self.db.execute("select v from c where k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def _put(self, k, v):
        with self.lock:
            self.db.execute("insert or replace into c values (?,?)", (k, json.dumps(v)))
            self.db.commit()

    def chat(self, model: str, messages: list[dict], max_tokens: int = 400) -> dict:
        payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                   "usage": {"include": True}}
        if model in REASONING_CAPABLE:
            payload["reasoning"] = {"enabled": False}
        k = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        hit = self._get(k)
        if hit is not None:
            return hit
        err = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(URL, json=payload, timeout=120,
                                  headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    d = r.json()
                    if "choices" in d and d["choices"]:
                        text = d["choices"][0]["message"].get("content") or ""
                        usage = d.get("usage", {})
                        out = {"text": text, "cost": usage.get("cost", 0.0),
                               "prompt_tokens": usage.get("prompt_tokens"),
                               "completion_tokens": usage.get("completion_tokens")}
                        self._put(k, out)
                        with self.lock:
                            self.cost += out["cost"] or 0.0
                        return out
                    err = str(d)[:300]
                else:
                    err = f"{r.status_code} {r.text[:300]}"
            except requests.RequestException as e:  # network error
                err = repr(e)
            time.sleep(min(2 ** attempt, 30))
        return {"text": "", "error": err, "cost": 0.0}

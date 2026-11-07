"""Minimal in-process background jobs for long dataset operations started from the API."""
from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def running_for(self, key: str) -> str | None:
        with self._lock:
            for jid, j in self._jobs.items():
                if j["key"] == key and j["state"] == "running":
                    return jid
        return None

    def start(self, key: str, kind: str, fn: Callable[[Callable[[dict], None]], dict]) -> str:
        if existing := self.running_for(key):
            return existing
        jid = uuid.uuid4().hex[:12]
        job = {"id": jid, "key": key, "kind": kind, "state": "running", "started_at": _now(),
               "finished_at": None, "progress": {}, "result": None, "error": None}
        with self._lock:
            self._jobs[jid] = job

        def report(progress: dict) -> None:
            job["progress"] = progress

        def run():
            try:
                job["result"] = fn(report)
                job["state"] = "succeeded"
            except Exception as exc:
                job["state"] = "failed"
                job["error"] = f"{exc.__class__.__name__}: {exc}"
                job["trace"] = traceback.format_exc(limit=3)
            job["finished_at"] = _now()

        threading.Thread(target=run, daemon=True, name=f"job-{jid}").start()
        return jid

    def get(self, jid: str) -> dict | None:
        return self._jobs.get(jid)

    def all(self) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: j["started_at"], reverse=True)

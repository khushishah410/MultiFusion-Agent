"""Thin HTTP client for the backend. The UI never touches the dataset files directly."""
from __future__ import annotations

import os

import requests

DEFAULT_URL = "http://127.0.0.1:8000"


class BackendError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60):
        self.base_url = (base_url or os.environ.get("MULTIFUSION_API_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kw):
        try:
            r = requests.request(method, f"{self.base_url}{path}", timeout=kw.pop("timeout", self.timeout), **kw)
        except requests.ConnectionError as exc:
            raise BackendError(f"Backend not reachable at {self.base_url}. Start it with run_backend.ps1.") from exc
        except requests.Timeout as exc:
            raise BackendError(f"Backend timed out on {path}.") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            if isinstance(detail, list):
                detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
            raise BackendError(f"{detail}", r.status_code)
        return r

    def get(self, path: str, **params):
        return self._request("GET", path, params={k: v for k, v in params.items() if v is not None}).json()

    def get_bytes(self, path: str) -> bytes:
        return self._request("GET", path).content

    def post(self, path: str, json: dict | None = None, timeout: float | None = None):
        return self._request("POST", path, json=json or {}, timeout=timeout or self.timeout).json()

    def analyze(self, form: dict, image: tuple[str, bytes, str] | None):
        files = {"image": image} if image else None
        data = {k: v for k, v in form.items() if v not in (None, "")}
        return self._request("POST", "/analyze", data=data, files=files, timeout=180).json()

    def healthy(self) -> bool:
        try:
            return self.get("/health").get("status") == "ok"
        except BackendError:
            return False

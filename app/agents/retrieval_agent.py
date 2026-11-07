"""Lexical (BM25) retrieval over a locally built, traceable index.

Documents come from two places only: report text in a prepared dataset
manifest (returned as *reference cases*) and files the user places in
data/knowledge/. Nothing is fetched from the web at query time.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.analysis import RetrievalResults, RetrievedPassage

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in on for with is are was were be been no not this that there it as at by from".split())
K1, B = 1.5, 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP and len(t) > 1]


def build_index(docs: list[dict], sources: list[dict], out_path: Path) -> dict:
    required = {"doc_id", "title", "text", "source", "kind"}
    seen = set()
    clean = []
    for d in docs:
        missing = required - d.keys()
        if missing:
            raise ValueError(f"Document missing fields {sorted(missing)}: {d.get('doc_id')}")
        if d["doc_id"] in seen or not d["text"].strip():
            continue
        seen.add(d["doc_id"])
        clean.append(d)
    index = {
        "format": "multifusion-bm25-v1",
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "doc_count": len(clean),
        "docs": clean,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index), encoding="utf-8")
    return {k: v for k, v in index.items() if k != "docs"}


class RetrievalAgent:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.docs: list[dict] = []
        self.meta: dict = {}
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self.error: str | None = None
        if index_path.exists():
            try:
                self._load()
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                self.error = f"Index could not be loaded: {exc}"

    def _load(self) -> None:
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.docs = data["docs"]
        self.meta = {k: v for k, v in data.items() if k != "docs"}
        self._tf = [Counter(tokenize(d["title"] + " " + d["text"])) for d in self.docs]
        for tf in self._tf:
            self._df.update(tf.keys())
        lengths = [sum(tf.values()) for tf in self._tf]
        self._avgdl = sum(lengths) / len(lengths) if lengths else 0.0

    @property
    def active(self) -> bool:
        return bool(self.docs) and self.error is None

    def status(self) -> dict:
        if self.error:
            return {"status": "error", "detail": self.error}
        if not self.active:
            return {"status": "not_configured",
                    "detail": "No index found. Build one with: python -m scripts.build_retrieval_index"}
        return {"status": "active", **self.meta}

    def search(self, query: str, k: int = 5) -> RetrievalResults:
        if not self.active:
            st = self.status()
            return RetrievalResults(status=st["status"], query=query, note=st["detail"])
        q_terms = tokenize(query)
        n = len(self.docs)
        scored = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values())
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self._avgdl))
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        passages = []
        for score, i in scored[:k]:
            d = self.docs[i]
            passages.append(RetrievedPassage(
                doc_id=d["doc_id"], title=d["title"], text=d["text"][:800], source=d["source"],
                source_url=d.get("source_url"), source_date=d.get("source_date"), kind=d["kind"],
                synthetic=bool(d.get("synthetic", False)), similarity=round(score, 3),
            ))
        note = None if passages else "No indexed document shares terms with the query."
        return RetrievalResults(status="active", query=query, passages=passages,
                                index_info={"built_at": self.meta.get("built_at"), "doc_count": n,
                                            "sources": self.meta.get("sources", [])}, note=note)

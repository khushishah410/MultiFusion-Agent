"""Build the BM25 retrieval index from the active manifest's reports and data/knowledge/.

Only add knowledge files you are permitted to use. Accepted formats in data/knowledge/:
  *.jsonl  one object per line: {"id", "title", "text", "source", "source_url"?, "source_date"?}
  *.md / *.txt  one document per file (source = file name)

    python -m scripts.build_retrieval_index [--manifest PATH] [--no-manifest]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.retrieval_agent import build_index
from app.core.config import get_settings
from app.services.manifest_service import load_records


def manifest_docs(manifest: Path) -> list[dict]:
    docs = []
    for r in load_records(manifest):
        text = " ".join(f"{name}: {val}" for name, val in (("Indication", r.clinical_indication),
                                                           ("Findings", r.radiology_findings),
                                                           ("Impression", r.radiology_impression)) if val)
        if text:
            docs.append({"doc_id": f"{r.dataset_source}:{r.case_id}", "title": f"Report {r.case_id}",
                         "text": text, "source": r.dataset_source, "kind": "reference_case",
                         "synthetic": r.synthetic, "split": r.split})
    return docs


def knowledge_docs(folder: Path) -> list[dict]:
    docs = []
    if not folder.exists():
        return docs
    for path in sorted(folder.rglob("*")):
        if path.suffix == ".jsonl":
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                d = json.loads(line)
                missing = {"id", "title", "text", "source"} - d.keys()
                if missing:
                    raise ValueError(f"{path.name} line {n}: missing {sorted(missing)}")
                docs.append({"doc_id": f"kb:{d['id']}", "title": d["title"], "text": d["text"], "source": d["source"],
                             "source_url": d.get("source_url"), "source_date": d.get("source_date"), "kind": "document"})
        elif path.suffix in (".md", ".txt") and path.name.lower() != "readme.md":
            docs.append({"doc_id": f"kb:{path.stem}", "title": path.stem, "text": path.read_text(encoding="utf-8"),
                         "source": path.name, "kind": "document"})
    return docs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--exclude-split", default="test",
                    help="Do not index reports from this split, so test cases cannot be retrieved (default: test)")
    args = ap.parse_args(argv)
    s = get_settings()
    docs, sources = [], []
    if not args.no_manifest:
        manifest = args.manifest or s.active_manifest()[0]
        if manifest and manifest.exists():
            mdocs = [d for d in manifest_docs(manifest) if d.get("split") != args.exclude_split]
            docs += mdocs
            sources.append({"type": "manifest_reports", "manifest": manifest.name, "documents": len(mdocs),
                            "excluded_split": args.exclude_split})
    kdocs = knowledge_docs(s.resolve(s.knowledge_dir))
    docs += kdocs
    if kdocs:
        sources.append({"type": "knowledge_dir", "documents": len(kdocs)})
    if not docs:
        print("Nothing to index: no manifest reports and no knowledge files.")
        return 1
    meta = build_index(docs, sources, s.resolve(s.retrieval_index))
    print(f"Indexed {meta['doc_count']} documents -> {s.resolve(s.retrieval_index)}")
    print(json.dumps(meta["sources"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

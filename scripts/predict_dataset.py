"""Run the configured image model over a manifest split and write a predictions JSONL file.

    python -m scripts.predict_dataset --split test [--limit 200] [--manifest PATH]

Output: reports/evaluation/predictions_<dataset>_<split>.jsonl, one line per image:
  {"case_id", "image_index", "dataset_source", "model": {...}, "scores": {label: score},
   "above_operating_point": {label: bool}, "latency_ms", "error"}
Labels are never read here; they are only joined in scripts.evaluate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.agents.image_agent import ImageAgent
from app.agents.model_adapters import build_adapter
from app.core.config import get_settings
from app.schemas.analysis import ModelStatus
from app.services.manifest_service import ManifestStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--split", default="test", help="train | val | test | all")
    ap.add_argument("--limit", type=int, help="Only the first N cases (quick checks)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    s = get_settings()
    manifest = args.manifest.resolve() if args.manifest else s.active_manifest()[0]
    store = ManifestStore(manifest, None, "cli")
    if store.error:
        print(f"Error: {store.error}", file=sys.stderr)
        return 1
    adapter = build_adapter(s)
    info = adapter.info()
    if info.status != ModelStatus.loaded:
        print(f"Error: image model unavailable: {info.reason}", file=sys.stderr)
        return 1
    agent = ImageAgent(adapter, max_bytes=10**10, min_side=s.min_image_side, max_side=s.max_image_side)
    records = [r for r in store.records if args.split == "all" or r.split == args.split]
    if args.limit:
        records = records[:args.limit]
    dataset = records[0].dataset_source if records else manifest.stem
    out = args.output or s.evaluation_dir_abs / f"predictions_{dataset}_{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = {"backend": info.backend, "name": info.name, "version": info.version, "labels": info.labels,
             "training_datasets": info.training_datasets}
    n_ok = n_err = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records, 1):
            for idx in range(len(rec.image_paths)):
                path = store.image_path(rec, idx)
                row = {"case_id": rec.case_id, "image_index": idx, "dataset_source": rec.dataset_source, "model": model}
                if path is None:
                    row |= {"scores": None, "error": "image missing on disk", "latency_ms": None}
                else:
                    res = agent.analyze(path.read_bytes())
                    row |= {"scores": {x.label: x.score for x in res.scores} or None,
                            "above_operating_point": {x.label: x.above_operating_point for x in res.scores
                                                      if x.above_operating_point is not None},
                            "latency_ms": res.latency_ms, "error": res.error}
                n_ok += row["error"] is None
                n_err += row["error"] is not None
                fh.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(records)} cases")
    print(f"Wrote {out} ({n_ok} images scored, {n_err} errors) for {len(records)} cases in split '{args.split}'.")
    if info.training_datasets and dataset in info.training_datasets:
        print(f"Warning: {info.name} was trained on data including {dataset}; this is not a held-out evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

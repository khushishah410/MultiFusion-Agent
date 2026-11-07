"""Compute per-label metrics from a predictions file and the manifest's labels.

    python -m scripts.evaluate --input reports/evaluation/predictions_iu_xray_test.jsonl
        [--output reports/evaluation/metrics.json] [--manifest PATH] [--label-map config/label_maps/iu_xray.yaml]
        [--split test|val|train|all] [--uncertainty-policy exclude|positive|negative] [--threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.services.evaluation_service import UNCERTAINTY_POLICIES, evaluate, save_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--label-map", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--uncertainty-policy", default="exclude", choices=UNCERTAINTY_POLICIES)
    ap.add_argument("--threshold", type=float, default=0.5, help="Used only when no operating-point flags exist")
    args = ap.parse_args(argv)
    s = get_settings()
    if not args.input.is_file():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1
    manifest = args.manifest or s.active_manifest()[0]
    if manifest is None or not manifest.exists():
        print("Error: no manifest; pass --manifest", file=sys.stderr)
        return 1
    first = json.loads(args.input.read_text(encoding="utf-8").splitlines()[0])
    dataset = first.get("dataset_source") or manifest.stem
    label_map = args.label_map or s.resolve(f"config/label_maps/{dataset}.yaml")
    split = None if args.split == "all" else args.split
    report = evaluate(args.input, manifest, label_map if label_map.exists() else None, split,
                      args.uncertainty_policy, args.threshold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output or s.evaluation_dir_abs / f"metrics_{dataset}_{args.split}_{stamp}.json"
    save_report(report, out)
    print(f"Cases evaluated: {report['cases_evaluated']} / {report['cases_in_split']} "
          f"(failed {report['failed']}, missing {report['missing_predictions']})")
    for e in report["per_label"]:
        auc = f"{e['auroc']:.3f}" if e["auroc"] is not None else "  -  "
        print(f"  {e['label']:<28} n={e['n']:<5} pos={e['positives']:<4} AUROC={auc}")
    for w in report["warnings"]:
        print(f"Warning: {w}")
    print(f"Report: {out} (+ .md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a manifest from any folder with a chosen builder (no registry entry needed).

    python -m scripts.build_manifest --dataset-id my_cxr --builder generic --root D:\\data\\my_cxr [--field-map map.yaml]
Builders: iu_xray_xml, nih_csv, chexpert_csv, mimic_csv, generic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.services.manifest_builders import BUILDERS
from app.services.manifest_service import build_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--builder", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--field-map", type=Path)
    args = ap.parse_args(argv)
    options = {"field_map": str(args.field_map)} if args.field_map else {}
    meta = build_manifest(args.dataset_id, args.builder, args.root.resolve(), get_settings().manifest_dir_abs, options)
    print(json.dumps({k: v for k, v in meta.items() if k != "build_issues"}, indent=2))
    print(f"{len(meta['build_issues'])} build issue(s) recorded in {args.dataset_id}.meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

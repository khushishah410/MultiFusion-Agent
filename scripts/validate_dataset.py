"""Validate a manifest against its image folder (same as `dataset_manager validate`, but for any manifest).

    python -m scripts.validate_dataset --manifest data/manifests/iu_xray.jsonl --root data/raw/iu_xray [--no-image-check]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.services.manifest_service import validate_manifest, write_validation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="Dataset root that image_paths are relative to")
    ap.add_argument("--no-image-check", action="store_true")
    ap.add_argument("--max-images", type=int)
    args = ap.parse_args(argv)
    report = validate_manifest(args.manifest, args.root, not args.no_image_check, args.max_images)
    json_path, md_path = write_validation(report, args.manifest.parent)
    print(md_path.read_text(encoding="utf-8"))
    return 0 if report.ok else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Optional: write downsized copies of a manifest's images for faster browsing/training.

Originals are never modified. Copies go to data/processed/<dataset>/<size>/ and a new
manifest <dataset>_<size>.jsonl points at them (image_paths stay relative).

    python -m scripts.prepare_images --dataset iu_xray --size 512
"""
from __future__ import annotations

import argparse
import json
import sys

from PIL import Image

from app.core.config import get_settings
from app.services.manifest_service import ManifestStore, _portable


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", type=int, default=512, help="Longest side in pixels")
    args = ap.parse_args(argv)
    s = get_settings()
    manifest = s.manifest_dir_abs / f"{args.dataset}.jsonl"
    store = ManifestStore(manifest, None, "cli")
    if store.error:
        print(f"Error: {store.error}", file=sys.stderr)
        return 1
    out_root = s.resolve("data/processed") / args.dataset / str(args.size)
    out_root.mkdir(parents=True, exist_ok=True)
    new_id = f"{args.dataset}_{args.size}"
    written = missing = 0
    with (s.manifest_dir_abs / f"{new_id}.jsonl").open("w", encoding="utf-8") as fh:
        for rec in store.records:
            for i, rel in enumerate(rec.image_paths):
                src = store.image_path(rec, i)
                if src is None:
                    missing += 1
                    continue
                dst = out_root / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(src) as im:
                        im.thumbnail((args.size, args.size))
                        im.save(dst)
                written += 1
            fh.write(rec.model_dump_json() + "\n")
    meta = store.meta | {"dataset_id": new_id, "dataset_root": _portable(out_root),
                         "derived_from": manifest.name, "resized_longest_side": args.size}
    (s.manifest_dir_abs / f"{new_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{written} images resized into {out_root}; {missing} missing. Manifest: {new_id}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

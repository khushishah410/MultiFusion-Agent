"""Generate the synthetic demo dataset (no medical content) and its manifest.

Images are procedural gradients stamped "SYNTHETIC"; they are not radiographs.
Records have no labels, so they cannot be mistaken for ground truth.

    python -m scripts.generate_demo_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.services.manifest_service import build_manifest, validate_manifest, write_validation

N_CASES = 12
PLACEHOLDER = "[SYNTHETIC] Placeholder text for software testing. Contains no clinical information."


def make_image(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 90 + 60 * np.sin(x / (18 + seed)) * np.cos(y / (23 + seed))
    arr = np.clip(base + rng.normal(0, 12, (size, size)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, 22], fill=0)
    draw.text((6, 5), f"SYNTHETIC #{seed:02d} - NOT AN X-RAY", fill=255)
    return img


def generate(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(1, N_CASES + 1):
        case_id = f"SYN-{i:03d}"
        image_ids = [f"syn_{i:03d}_a"]
        if i == N_CASES:
            image_ids.append(f"syn_{i:03d}_missing")  # deliberately absent: demonstrates missing-file warnings
        for image_id in image_ids:
            if not image_id.endswith("missing"):
                make_image(i).save(out_dir / f"{image_id}.png")
        records.append({
            "case_id": case_id,
            "clinical_indication": PLACEHOLDER,
            "radiology_findings": PLACEHOLDER,
            "radiology_impression": PLACEHOLDER,
            "image_ids": image_ids,
            "image_paths": [f"{image_id}.png" for image_id in image_ids],
            "labels": {},
            "label_source": "none (synthetic)",
            "dataset_source": "synthetic_demo",
            "synthetic": True,
            "metadata": {"generator": "scripts/generate_demo_data.py", "seed": i},
        })
    (out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    s = get_settings()
    out_dir = s.data_root_abs / "synthetic_demo"
    generate(out_dir)
    meta = build_manifest("synthetic_demo", "generic", out_dir, s.manifest_dir_abs,
                          {"files": [str(out_dir / "records.jsonl")]},
                          {"official_source": "generated locally", "license": "project-generated, no patient data"})
    report = validate_manifest(s.manifest_dir_abs / "synthetic_demo.jsonl", out_dir)
    write_validation(report, s.manifest_dir_abs)
    print(f"Synthetic demo: {meta['records_written']} records in {s.manifest_dir_abs / 'synthetic_demo.jsonl'}")
    print(f"Validation: {report.records_with_missing_images} record(s) with a missing image (1 is intentional).")
    print("Build the retrieval index with: python -m scripts.build_retrieval_index")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Manifest generation, validation, splitting and read access."""
from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from app.core.config import REPO_ROOT
from app.core.paths import UnsafePathError, is_unsafe_relative, safe_join
from app.schemas.dataset import ManifestRecord, PublicCase, ValidationIssue, ValidationReport
from app.services.manifest_builders import BUILDERS

MAX_ISSUES = 200
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
SPLIT_ALIASES = {"valid": "val", "validation": "val", "validate": "val"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def group_key(rec: ManifestRecord) -> str:
    return rec.patient_id or rec.case_id


def hashed_split(key: str, fractions=SPLIT_FRACTIONS) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
    acc = 0.0
    for name, frac in fractions:
        acc += frac
        if bucket < acc:
            return name
    return fractions[-1][0]


def assign_splits(records: list[ManifestRecord]) -> dict:
    """Keep source-provided splits; otherwise split deterministically by grouping key.

    A source 'trainval' split (NIH) is divided into train/val by group so groups never straddle.
    """
    method = Counter()
    for rec in records:
        if rec.split:
            rec.split = SPLIT_ALIASES.get(rec.split.lower(), rec.split.lower())
        if rec.split == "trainval":
            rec.split = hashed_split(group_key(rec), (("train", 0.85), ("val", 0.15)))
            method["source_trainval_then_grouped_hash"] += 1
        elif rec.split:
            method["source"] += 1
        else:
            rec.split = hashed_split(group_key(rec))
            method["grouped_hash_70_15_15"] += 1
    return dict(method)


def leakage_check(records: list[ManifestRecord]) -> dict:
    groups: dict[str, set] = defaultdict(set)
    images: dict[str, set] = defaultdict(set)
    for rec in records:
        groups[group_key(rec)].add(rec.split)
        for p in rec.image_paths:
            images[p].add(rec.split)
    leaking_groups = sorted(g for g, s in groups.items() if len(s) > 1)
    leaking_images = sorted(i for i, s in images.items() if len(s) > 1)
    with_patient = sum(1 for r in records if r.patient_id)
    return {
        "grouping_key": "patient_id" if with_patient == len(records) and records else
                        ("patient_id where available, else case_id" if with_patient else "case_id"),
        "groups": len(groups),
        "groups_in_multiple_splits": len(leaking_groups),
        "example_groups_in_multiple_splits": ["<redacted>"] * min(5, len(leaking_groups)),
        "images_in_multiple_splits": len(leaking_images),
        "example_images_in_multiple_splits": leaking_images[:5],
        "ok": not leaking_groups and not leaking_images,
    }


def normalize_records(raw: list[dict], issues: list[ValidationIssue]) -> tuple[list[ManifestRecord], list[str], int]:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    malformed = 0
    out = []
    for row in raw:
        source_id = row.get("case_id")
        if isinstance(source_id, (int, float)):
            row["case_id"] = str(source_id)
        if isinstance(row.get("case_id"), str):
            row["case_id"] = row["case_id"].strip()
            if row["case_id"] != source_id:
                row.setdefault("metadata", {})["source_case_id"] = source_id
        if row.get("patient_id") is not None:
            row["patient_id"] = str(row["patient_id"]).strip() or None
        try:
            rec = ManifestRecord.model_validate(row)
        except ValidationError as exc:
            malformed += 1
            issues.append(ValidationIssue(kind="malformed_record", case_id=str(row.get("case_id")),
                                          detail="; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
                                                           for e in exc.errors()[:3])))
            continue
        if rec.case_id in seen:
            duplicates.append(rec.case_id)
            issues.append(ValidationIssue(kind="duplicate_case_id", case_id=rec.case_id,
                                          detail="Later occurrence dropped; first occurrence kept."))
            continue
        bad_paths = [p for p in rec.image_paths if is_unsafe_relative(p)]
        if bad_paths:
            malformed += 1
            issues.append(ValidationIssue(kind="unsafe_path", case_id=rec.case_id,
                                          detail=f"Rejected paths outside the dataset root: {bad_paths[:3]}"))
            continue
        seen[rec.case_id] = len(out)
        out.append(rec)
    return out, sorted(set(duplicates)), malformed


def build_manifest(dataset_id: str, builder_name: str, source_root: Path, manifest_dir: Path,
                   options: dict | None = None, source_info: dict | None = None) -> dict:
    """Run a builder, normalize, split and write <manifest_dir>/<dataset_id>.jsonl plus metadata.

    The output file is replaced atomically, so re-running never duplicates records.
    Original dataset files are only read.
    """
    if builder_name not in BUILDERS:
        raise ValueError(f"Unknown manifest builder {builder_name!r}; available: {sorted(BUILDERS)}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {source_root}")
    raw, issues = BUILDERS[builder_name](source_root, dataset_id, options or {})
    records, duplicates, malformed = normalize_records(raw, issues)
    split_method = assign_splits(records)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{dataset_id}.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.model_dump_json(exclude_none=False) + "\n")
    os.replace(tmp, out)

    meta = {
        "dataset_id": dataset_id,
        "builder": builder_name,
        "dataset_root": _portable(source_root),
        "generated_at": _now(),
        "source": source_info or {},
        "options": {k: str(v) for k, v in (options or {}).items()},
        "records_read": len(raw),
        "records_written": len(records),
        "duplicate_case_ids": len(duplicates),
        "malformed_records": malformed,
        "split_method": split_method,
        "synthetic": any(r.synthetic for r in records),
        "build_issues": [i.model_dump() for i in issues[:MAX_ISSUES]],
    }
    (manifest_dir / f"{dataset_id}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _portable(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def load_records(manifest: Path) -> list[ManifestRecord]:
    records = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(ManifestRecord.model_validate_json(line))
    return records


def validate_manifest(manifest: Path, dataset_root: Path, check_images: bool = True,
                      max_image_checks: int | None = None) -> ValidationReport:
    issues: list[ValidationIssue] = []
    total = malformed = 0
    records: list[ManifestRecord] = []
    ids = Counter()
    with manifest.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = ManifestRecord.model_validate_json(line)
            except ValidationError as exc:
                malformed += 1
                issues.append(ValidationIssue(kind="malformed_record", detail=f"line {n}: {exc.errors()[0]['msg']}"))
                continue
            ids[rec.case_id] += 1
            records.append(rec)
    duplicates = sorted(k for k, v in ids.items() if v > 1)
    for d in duplicates:
        issues.append(ValidationIssue(kind="duplicate_case_id", case_id=d, detail=f"appears {ids[d]} times"))

    missing_files = unreadable = checked = 0
    records_missing = 0
    widths, heights = [], []
    for rec in records:
        rec_missing = False
        if not rec.image_paths:
            continue
        for p in rec.image_paths:
            try:
                full = safe_join(dataset_root, p)
            except UnsafePathError as exc:
                issues.append(ValidationIssue(kind="unsafe_path", case_id=rec.case_id, detail=str(exc)))
                rec_missing = True
                continue
            if not full.is_file():
                missing_files += 1
                rec_missing = True
                if missing_files <= 20:
                    issues.append(ValidationIssue(kind="missing_image", case_id=rec.case_id, detail=p))
                continue
            if check_images and (max_image_checks is None or checked < max_image_checks):
                checked += 1
                try:
                    with Image.open(full) as im:
                        im.verify()
                    with Image.open(full) as im:
                        widths.append(im.width)
                        heights.append(im.height)
                except (UnidentifiedImageError, OSError, SyntaxError) as exc:
                    unreadable += 1
                    issues.append(ValidationIssue(kind="unreadable_image", case_id=rec.case_id, detail=f"{p}: {exc}"))
        records_missing += rec_missing

    fields = ["clinical_indication", "symptoms", "radiology_findings", "radiology_impression", "image_ids",
              "image_paths", "labels", "label_uncertainty", "mesh_major", "patient_id"]
    coverage = {f: sum(1 for r in records if getattr(r, f)) for f in fields}
    size_stats = {}
    if widths:
        size_stats = {"width_min": min(widths), "width_median": statistics.median(widths), "width_max": max(widths),
                      "height_min": min(heights), "height_median": statistics.median(heights),
                      "height_max": max(heights)}
    leakage = leakage_check(records)
    report = ValidationReport(
        dataset_id=manifest.stem, manifest=manifest.name, generated_at=_now(),
        total_records=total, valid_records=len(records) - sum(ids[d] - 1 for d in duplicates),
        duplicate_case_ids=duplicates[:50], malformed_records=malformed,
        records_with_missing_images=records_missing, missing_image_files=missing_files,
        unreadable_images=unreadable, images_checked=checked, image_size_stats=size_stats,
        field_coverage=coverage, split_counts=dict(Counter(r.split or "unassigned" for r in records)),
        leakage=leakage, issues=issues[:MAX_ISSUES],
        ok=bool(records) and not malformed and not duplicates and leakage["ok"] and unreadable == 0,
    )
    return report


def write_validation(report: ValidationReport, manifest_dir: Path) -> tuple[Path, Path]:
    json_path = manifest_dir / f"{report.dataset_id}.validation.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md = [f"# Validation summary: {report.dataset_id}", "", f"Generated: {report.generated_at}", "",
          f"- Records: {report.total_records} ({report.valid_records} valid, {report.malformed_records} malformed, "
          f"{len(report.duplicate_case_ids)} duplicated ids)",
          f"- Records with missing images: {report.records_with_missing_images} "
          f"({report.missing_image_files} missing files)",
          f"- Images checked: {report.images_checked}, unreadable: {report.unreadable_images}",
          f"- Splits: {report.split_counts}",
          f"- Leakage check: {'passed' if report.leakage.get('ok') else 'FAILED'} "
          f"(grouping key: {report.leakage.get('grouping_key')})",
          f"- Field coverage: {report.field_coverage}",
          f"- Overall: {'OK' if report.ok else 'needs attention'}", ""]
    md_path = manifest_dir / f"{report.dataset_id}.validation.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path


class ManifestStore:
    """Read-only, in-memory view of the active manifest for the API."""

    def __init__(self, manifest: Path | None, dataset_root: Path | None, how: str):
        self.manifest = manifest
        self.how = how
        self.records: list[ManifestRecord] = []
        self.by_id: dict[str, ManifestRecord] = {}
        self.meta: dict = {}
        self.error: str | None = None
        self.dataset_root = dataset_root
        if manifest is None:
            self.error = "No manifest available. Run: python -m scripts.generate_demo_data"
            return
        if not manifest.exists():
            self.error = f"Configured manifest {manifest.name} does not exist."
            return
        try:
            self.records = load_records(manifest)
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            self.error = f"Manifest could not be read: {exc}"
            return
        self.by_id = {r.case_id: r for r in self.records}
        meta_path = manifest.with_name(manifest.stem + ".meta.json")
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.dataset_root is None and self.meta.get("dataset_root"):
            root = Path(self.meta["dataset_root"])
            self.dataset_root = root if root.is_absolute() else REPO_ROOT / root

    def image_path(self, rec: ManifestRecord, index: int) -> Path | None:
        if self.dataset_root is None or index >= len(rec.image_paths):
            return None
        try:
            p = safe_join(self.dataset_root, rec.image_paths[index])
        except UnsafePathError:
            return None
        return p if p.is_file() else None

    def to_public(self, rec: ManifestRecord) -> PublicCase:
        data = rec.model_dump(exclude={"patient_id"})
        data["image_exists"] = [self.image_path(rec, i) is not None for i in range(len(rec.image_paths))]
        return PublicCase(**data)

    def search(self, q: str | None = None, split: str | None = None, label: str | None = None,
               has_images: bool | None = None) -> list[ManifestRecord]:
        out = self.records
        if q:
            ql = q.lower()
            out = [r for r in out if ql in r.case_id.lower() or any(
                ql in (t or "").lower() for t in (r.clinical_indication, r.radiology_findings,
                                                  r.radiology_impression, " ".join(r.mesh_major)))]
        if split:
            out = [r for r in out if r.split == split]
        if label:
            out = [r for r in out if r.labels.get(label) == 1]
        if has_images is not None:
            out = [r for r in out if bool(r.image_paths) == has_images]
        return out

    def summary(self) -> dict:
        labels = Counter()
        uncertain = Counter()
        for r in self.records:
            labels.update(k for k, v in r.labels.items() if v == 1)
            uncertain.update(r.label_uncertainty)
        sources = sorted({r.dataset_source for r in self.records})
        return {
            "manifest": self.manifest.name if self.manifest else None,
            "selected_by": self.how,
            "error": self.error,
            "case_count": len(self.records),
            "image_count": sum(len(r.image_paths) for r in self.records),
            "synthetic": any(r.synthetic for r in self.records),
            "dataset_sources": sources,
            "split_counts": dict(Counter(r.split or "unassigned" for r in self.records)),
            "label_counts": dict(labels.most_common()),
            "uncertain_label_counts": dict(uncertain.most_common()),
            "generated_at": self.meta.get("generated_at"),
            "duplicate_case_ids": self.meta.get("duplicate_case_ids"),
            "malformed_records": self.meta.get("malformed_records"),
            "source": self.meta.get("source", {}),
        }

    def report_texts(self) -> list[str]:
        return [" ".join(filter(None, [r.clinical_indication, r.radiology_findings, r.radiology_impression]))
                for r in self.records if not r.synthetic]

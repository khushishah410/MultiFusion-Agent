"""Source-specific readers that turn raw dataset files into normalized record dicts.

Each builder returns (records, issues). Records still go through schema
validation, de-duplication and split assignment in manifest_service.
Original values are kept in `metadata` when they are normalized.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import yaml

from app.schemas.dataset import ValidationIssue

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
Builder = Callable[[Path, str, dict], tuple[list[dict], list[ValidationIssue]]]


def index_images(root: Path) -> dict[str, str]:
    """Map lower-cased file stem -> path relative to root (first occurrence wins)."""
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            out.setdefault(p.stem.lower(), p.relative_to(root).as_posix())
    return out


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text or None


# ------------------------------------------------------------------ IU X-Ray
def build_iu_xray(root: Path, dataset_id: str, options: dict):
    """Parse Open-i ecgen-radiology XML reports.

    Structure (as used by published loaders): root <eCitation> with <pmcId id>,
    <uId id>, MedlineCitation/Article/Abstract/AbstractText[@Label], <MeSH><major>/<minor>,
    and one <parentImage id> per image; images are <id>.png in NLMCXR_png.tgz.
    """
    images = index_images(root)
    records, issues = [], []
    xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        issues.append(ValidationIssue(kind="missing_metadata", detail="No .xml report files found under the dataset root."))
    for path in xml_files:
        try:
            doc = ET.parse(path).getroot()
        except ET.ParseError as exc:
            issues.append(ValidationIssue(kind="malformed_xml", detail=f"{path.name}: {exc}"))
            continue
        uid = doc.find("uId")
        pmc = doc.find("pmcId")
        source_id = (uid.get("id") if uid is not None else None) or (
            f"CXR{pmc.get('id')}" if pmc is not None and pmc.get("id") else None)
        if not source_id:
            issues.append(ValidationIssue(kind="malformed_record", detail=f"{path.name}: no uId/pmcId"))
            continue
        sections = {}
        for el in doc.iter("AbstractText"):
            label = (el.get("Label") or "").strip().upper()
            if label:
                sections[label] = _clean("".join(el.itertext()))
        mesh_major = [_clean(e.text) for e in doc.iter("major") if _clean(e.text)]
        mesh_minor = [_clean(e.text) for e in doc.iter("minor") if _clean(e.text)]
        image_ids = [img.get("id") for img in doc.iter("parentImage") if img.get("id")]
        image_paths = [images.get(i.lower(), f"{i}.png") for i in image_ids]
        headings = sorted({m.split("/")[0].strip() for m in mesh_major})
        labels = {h: 1 for h in headings if h.lower() not in ("normal", "no indexing")}
        records.append({
            "case_id": source_id.strip(),
            "patient_id": None,
            "clinical_indication": sections.get("INDICATION"),
            "radiology_findings": sections.get("FINDINGS"),
            "radiology_impression": sections.get("IMPRESSION"),
            "image_ids": image_ids,
            "image_paths": image_paths,
            "labels": labels,
            "label_uncertainty": [],
            "label_source": "MeSH major headings of the report (weak, positives only)",
            "mesh_major": mesh_major,
            "dataset_source": dataset_id,
            "metadata": {
                "source_file": path.relative_to(root).as_posix(),
                "comparison": sections.get("COMPARISON"),
                "mesh_minor": mesh_minor,
                "labels_are_positives_only": True,
                "mesh_normal": any(m.lower().startswith("normal") for m in mesh_major),
                "grouping_key_note": "No patient id is published; case_id is used for split grouping.",
            },
        })
    return records, issues


# ------------------------------------------------------------------ NIH CXR14
NIH_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
              "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


def build_nih(root: Path, dataset_id: str, options: dict):
    issues = []
    csvs = sorted(root.rglob("Data_Entry_2017*.csv"))
    if not csvs:
        return [], [ValidationIssue(kind="missing_metadata", detail="Data_Entry_2017*.csv not found.")]
    images = index_images(root)
    official = {}
    for name, split in (("train_val_list.txt", "trainval"), ("test_list.txt", "test")):
        for f in root.rglob(name):
            for line in f.read_text(encoding="utf-8").split():
                official[line.strip()] = split
    records = []
    with csvs[0].open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Image Index") or "").strip()
            if not name:
                issues.append(ValidationIssue(kind="malformed_record", detail="row without Image Index"))
                continue
            findings = {f.strip() for f in (row.get("Finding Labels") or "").split("|") if f.strip()}
            unknown = findings - set(NIH_LABELS) - {"No Finding"}
            if unknown:
                issues.append(ValidationIssue(kind="unknown_label", case_id=name, detail=f"{sorted(unknown)}"))
            records.append({
                "case_id": Path(name).stem,
                "patient_id": (row.get("Patient ID") or "").strip() or None,
                "image_ids": [Path(name).stem],
                "image_paths": [images.get(Path(name).stem.lower(), name)],
                "labels": {lab: int(lab in findings) for lab in NIH_LABELS},
                "label_source": "NLP-mined from reports (Wang et al. 2017); weak labels",
                "dataset_source": dataset_id,
                "split": official.get(name),
                "metadata": {k: row.get(k) for k in ("Follow-up #", "Patient Age", "Patient Gender", "View Position")
                             if k in row} | {"finding_labels_raw": row.get("Finding Labels")},
            })
    return records, issues


# ------------------------------------------------------------------ CheXpert
CHEXPERT_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                   "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                   "Pleural Other", "Fracture", "Support Devices"]


def _chexpert_labels(row: dict, label_names: list[str]):
    labels, uncertain, raw = {}, [], {}
    for lab in label_names:
        value = (row.get(lab) or "").strip()
        raw[lab] = value
        if value in ("", "nan"):
            continue
        v = float(value)
        if v == -1:
            uncertain.append(lab)
        elif v in (0, 1):
            labels[lab] = int(v)
    return labels, uncertain, raw


def _resolve_listed_path(root: Path, listed: str) -> str:
    """CSV paths include the archive's top folder; return the first suffix that exists under root."""
    parts = listed.replace("\\", "/").split("/")
    for k in range(len(parts)):
        candidate = "/".join(parts[k:])
        if (root / candidate).is_file():
            return candidate
    return "/".join(parts)


def build_chexpert(root: Path, dataset_id: str, options: dict):
    records, issues = [], []
    csvs = [p for p in root.rglob("*.csv") if p.name in ("train.csv", "valid.csv", "test.csv")]
    if not csvs:
        return [], [ValidationIssue(kind="missing_metadata", detail="train.csv / valid.csv not found.")]
    for path in sorted(csvs):
        split = {"train.csv": "train", "valid.csv": "val", "test.csv": "test"}[path.name]
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                listed = (row.get("Path") or "").strip()
                if not listed:
                    issues.append(ValidationIssue(kind="malformed_record", detail=f"{path.name}: row without Path"))
                    continue
                m = re.search(r"patient(\d+)/study(\d+)/(view\d+_\w+)", listed)
                case_id = f"patient{m.group(1)}_study{m.group(2)}_{m.group(3)}" if m else listed
                labels, uncertain, raw = _chexpert_labels(row, [l for l in CHEXPERT_LABELS if l in row])
                records.append({
                    "case_id": case_id,
                    "patient_id": f"patient{m.group(1)}" if m else None,
                    "image_ids": [case_id],
                    "image_paths": [_resolve_listed_path(root, listed)],
                    "labels": labels,
                    "label_uncertainty": uncertain,
                    "label_source": "CheXpert labeler (1 positive, 0 negative, -1 uncertain; blank = not mentioned)",
                    "dataset_source": dataset_id,
                    "split": split,
                    "metadata": {"raw_labels": raw, "listed_path": listed} |
                                {k: row.get(k) for k in ("Sex", "Age", "Frontal/Lateral", "AP/PA") if k in row},
                })
    return records, issues


# ------------------------------------------------------------------ MIMIC-CXR-JPG
def _read_csv_any(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_mimic(root: Path, dataset_id: str, options: dict):
    issues = []
    lab_f = next(iter(root.rglob("mimic-cxr-2.0.0-chexpert.csv*")), None)
    split_f = next(iter(root.rglob("mimic-cxr-2.0.0-split.csv*")), None)
    if not lab_f or not split_f:
        return [], [ValidationIssue(kind="missing_metadata",
                                    detail="mimic-cxr-2.0.0-chexpert.csv(.gz) and mimic-cxr-2.0.0-split.csv(.gz) are required.")]
    studies: dict[str, dict] = {}
    for row in _read_csv_any(split_f):
        st = studies.setdefault(row["study_id"], {"subject_id": row["subject_id"], "dicoms": [], "split": row["split"]})
        st["dicoms"].append(row["dicom_id"])
    records = []
    for row in _read_csv_any(lab_f):
        sid = row["study_id"]
        st = studies.get(sid)
        if st is None:
            issues.append(ValidationIssue(kind="missing_split", case_id=sid, detail="study not in split file"))
            continue
        subj = st["subject_id"]
        labels, uncertain, raw = _chexpert_labels(row, [l for l in CHEXPERT_LABELS if l in row])
        rel_dir = f"files/p{subj[:2]}/p{subj}/s{sid}"
        records.append({
            "case_id": f"s{sid}",
            "patient_id": subj,
            "image_ids": st["dicoms"],
            "image_paths": [f"{rel_dir}/{d}.jpg" for d in st["dicoms"]],
            "labels": labels,
            "label_uncertainty": uncertain,
            "label_source": "CheXpert labeler on MIMIC-CXR reports",
            "dataset_source": dataset_id,
            "split": {"validate": "val"}.get(st["split"], st["split"]),
            "metadata": {"raw_labels": raw},
        })
    return records, issues


# ------------------------------------------------------------------ generic
LIST_FIELDS = {"symptoms", "image_ids", "image_paths", "label_uncertainty", "mesh_major"}


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if text.startswith("["):
        return json.loads(text)
    return [v.strip() for v in re.split(r"[|;]", text) if v.strip()]


def build_generic(root: Path, dataset_id: str, options: dict):
    """Read JSON / JSONL / CSV files already using (or mapped to) the normalized field names.

    options["field_map"]: {source column: normalized field}; options["files"]: explicit file list.
    """
    field_map = options.get("field_map") or {}
    if isinstance(field_map, (str, Path)):
        field_map = yaml.safe_load(Path(field_map).read_text(encoding="utf-8"))
    files = [Path(f) for f in options.get("files", [])] or sorted(
        p for p in root.rglob("*") if p.suffix.lower() in (".jsonl", ".json", ".csv")
        and not p.name.startswith(".") and p.name != "manifest_meta.json")
    images = index_images(root)
    records, issues = [], []
    for path in files:
        try:
            if path.suffix.lower() == ".jsonl":
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            elif path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("records", [])
            else:
                with path.open(newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
        except (json.JSONDecodeError, UnicodeDecodeError, csv.Error) as exc:
            issues.append(ValidationIssue(kind="malformed_file", detail=f"{path.name}: {exc}"))
            continue
        for row in rows:
            if not isinstance(row, dict):
                issues.append(ValidationIssue(kind="malformed_record", detail=f"{path.name}: non-object row"))
                continue
            rec = {field_map.get(k, k): v for k, v in row.items()}
            for f in LIST_FIELDS:
                if f in rec:
                    try:
                        rec[f] = [str(x) for x in _as_list(rec[f])]
                    except json.JSONDecodeError:
                        issues.append(ValidationIssue(kind="malformed_field", case_id=str(rec.get("case_id")),
                                                      detail=f"{f} is not a list"))
                        rec[f] = []
            if isinstance(rec.get("labels"), str):
                rec["labels"] = json.loads(rec["labels"]) if rec["labels"].strip() else {}
            if not rec.get("image_paths") and rec.get("image_ids"):
                rec["image_paths"] = [images.get(i.lower(), i) for i in rec["image_ids"]]
            rec.setdefault("dataset_source", dataset_id)
            known = {"case_id", "patient_id", "clinical_indication", "symptoms", "radiology_findings",
                     "radiology_impression", "image_ids", "image_paths", "labels", "label_uncertainty",
                     "label_source", "mesh_major", "dataset_source", "split", "synthetic", "metadata"}
            extra = {k: v for k, v in rec.items() if k not in known}
            rec = {k: v for k, v in rec.items() if k in known}
            rec["metadata"] = (rec.get("metadata") or {}) | extra | {"source_file": path.name}
            records.append(rec)
    return records, issues


BUILDERS: dict[str, Builder] = {
    "iu_xray_xml": build_iu_xray,
    "nih_csv": build_nih,
    "chexpert_csv": build_chexpert,
    "mimic_csv": build_mimic,
    "generic": build_generic,
}

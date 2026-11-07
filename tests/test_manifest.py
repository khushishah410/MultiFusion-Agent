import json

import pytest

from app.core.paths import UnsafePathError, safe_join
from app.services.manifest_service import (
    ManifestStore,
    build_manifest,
    leakage_check,
    load_records,
    validate_manifest,
)
from tests.conftest import png_bytes, write_iu_fixture, write_jsonl


def test_iu_xml_builder_end_to_end(iu_prepared):
    settings, ops, meta = iu_prepared
    recs = load_records(settings.manifest_dir_abs / "iu_xray.jsonl")
    assert meta["records_written"] == len(recs) == 6
    r1 = next(r for r in recs if r.case_id == "CXR1")
    assert r1.clinical_indication == "Fixture indication 1 cough"
    assert r1.labels == {"Cardiomegaly": 1} and r1.mesh_major == ["Cardiomegaly/mild"]
    assert r1.metadata["labels_are_positives_only"] and len(r1.image_paths) == 2
    r2 = next(r for r in recs if r.case_id == "CXR2")
    assert r2.labels == {} and r2.metadata["mesh_normal"]
    rep = json.loads((settings.manifest_dir_abs / "iu_xray.validation.json").read_text())
    assert rep["missing_image_files"] == 1 and rep["records_with_missing_images"] == 1
    assert rep["images_checked"] == 11 and rep["leakage"]["ok"]


def test_rerun_does_not_duplicate(iu_prepared):
    settings, ops, _ = iu_prepared
    ops.prepare("iu_xray")
    lines = (settings.manifest_dir_abs / "iu_xray.jsonl").read_text().strip().splitlines()
    assert len(lines) == 6


def test_generic_builder_with_iu_style_fields_duplicates_and_malformed(tmp_path):
    root = tmp_path / "ds"
    (root / "img").mkdir(parents=True)
    (root / "img" / "a1.png").write_bytes(png_bytes())
    write_jsonl(root / "cases.jsonl", [
        {"case_id": " c1 ", "clinical_indication": "cough", "radiology_findings": "f", "radiology_impression": "i",
         "image_ids": ["a1"], "image_paths": ["img/a1.png"], "mesh_major": ["Opacity/lung"]},
        {"case_id": "c1", "clinical_indication": "duplicate"},
        {"clinical_indication": "no id"},
        {"case_id": "c3", "image_paths": ["../../etc/passwd"]},
        {"case_id": "c4", "image_ids": "a1|missing_one"},
    ])
    meta = build_manifest("toy", "generic", root, tmp_path / "m")
    recs = {r.case_id: r for r in load_records(tmp_path / "m" / "toy.jsonl")}
    assert set(recs) == {"c1", "c4"}
    assert recs["c1"].metadata["source_case_id"] == " c1 "
    assert recs["c4"].image_paths == ["img/a1.png", "missing_one"]
    assert meta["duplicate_case_ids"] == 1 and meta["malformed_records"] == 2
    kinds = {i["kind"] for i in meta["build_issues"]}
    assert {"duplicate_case_id", "malformed_record", "unsafe_path"} <= kinds


def test_validation_flags_missing_and_unsafe_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.png").write_bytes(png_bytes())
    (root / "bad.png").write_bytes(b"not a png")
    m = write_jsonl(tmp_path / "x.jsonl", [
        {"case_id": "a", "image_paths": ["ok.png", "gone.png"], "dataset_source": "t", "split": "train"},
        {"case_id": "b", "image_paths": ["bad.png"], "dataset_source": "t", "split": "test"},
        {"case_id": "c", "image_paths": ["../escape.png"], "dataset_source": "t", "split": "test"},
        {"case_id": "a", "dataset_source": "t", "split": "train"},
    ])
    rep = validate_manifest(m, root)
    assert rep.duplicate_case_ids == ["a"] and rep.missing_image_files == 1 and rep.unreadable_images == 1
    assert any(i.kind == "unsafe_path" for i in rep.issues) and not rep.ok


def test_safe_join_blocks_traversal(tmp_path):
    for bad in ["../x.png", "/etc/passwd", "C:\\Windows\\x.png", "a/../../x", "..\\x.png", ""]:
        with pytest.raises(UnsafePathError):
            safe_join(tmp_path, bad)
    assert safe_join(tmp_path, "sub/ok.png") == (tmp_path / "sub" / "ok.png").resolve()


def test_chexpert_uncertainty_preserved_and_patient_split(tmp_path):
    root = tmp_path / "chex" / "CheXpert-v1.0-small" / "train" / "patient00001" / "study1"
    root.mkdir(parents=True)
    (root / "view1_frontal.jpg").write_bytes(png_bytes(fmt="JPEG"))
    base = tmp_path / "chex"
    (base / "train.csv").write_text(
        "Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Cardiomegaly,Edema,Pleural Effusion\n"
        "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,F,50,Frontal,AP,,1.0,-1.0,0.0\n"
        "CheXpert-v1.0-small/train/patient00001/study2/view1_frontal.jpg,F,50,Frontal,AP,1.0,,,\n")
    build_manifest("chexpert", "chexpert_csv", base, tmp_path / "m")
    recs = load_records(tmp_path / "m" / "chexpert.jsonl")
    r = recs[0]
    assert r.labels == {"Cardiomegaly": 1, "Pleural Effusion": 0} and r.label_uncertainty == ["Edema"]
    assert r.patient_id == "patient00001" and r.split == "train"
    assert r.image_paths[0] == "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"


def test_nih_builder_and_official_test_split(tmp_path):
    root = tmp_path / "nih"
    (root / "images").mkdir(parents=True)
    (root / "images" / "00000001_000.png").write_bytes(png_bytes())
    (root / "Data_Entry_2017.csv").write_text(
        "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position\n"
        "00000001_000.png,Cardiomegaly|Effusion,0,1,58,M,PA\n00000002_000.png,No Finding,0,2,81,M,PA\n")
    (root / "test_list.txt").write_text("00000002_000.png\n")
    build_manifest("nih_cxr14", "nih_csv", root, tmp_path / "m")
    recs = {r.case_id: r for r in load_records(tmp_path / "m" / "nih_cxr14.jsonl")}
    assert recs["00000001_000"].labels["Cardiomegaly"] == 1 and recs["00000001_000"].labels["Hernia"] == 0
    assert recs["00000001_000"].image_paths == ["images/00000001_000.png"]
    assert recs["00000002_000"].split == "test" and sum(recs["00000002_000"].labels.values()) == 0


def test_grouped_split_has_no_patient_leakage(tmp_path):
    rows = [{"case_id": f"c{i}", "patient_id": f"p{i // 3}", "dataset_source": "t"} for i in range(300)]
    write_jsonl(tmp_path / "src" / "r.jsonl", rows)
    build_manifest("t", "generic", tmp_path / "src", tmp_path / "m")
    recs = load_records(tmp_path / "m" / "t.jsonl")
    assert leakage_check(recs)["ok"]
    assert {r.split for r in recs} == {"train", "val", "test"}


def test_store_hides_patient_id_and_absolute_paths(iu_prepared):
    settings, ops, _ = iu_prepared
    store = ManifestStore(settings.manifest_dir_abs / "iu_xray.jsonl", None, "test")
    pub = store.to_public(store.records[0]).model_dump()
    assert "patient_id" not in pub
    assert not any(p.startswith(("/", "\\")) or ":" in p for p in pub["image_paths"])

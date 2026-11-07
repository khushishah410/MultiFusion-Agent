import json

import pytest

from app.services.evaluation_service import evaluate, list_reports, save_report
from tests.conftest import write_jsonl

MODEL = {"backend": "fake", "name": "fake", "labels": ["Cardiomegaly", "Edema"], "training_datasets": []}


def manifest(tmp_path, rows):
    return write_jsonl(tmp_path / "m.jsonl", [{"dataset_source": "toy", "split": "test", **r} for r in rows])


def preds(tmp_path, rows):
    return write_jsonl(tmp_path / "p.jsonl", [{"model": MODEL, "image_index": 0, "latency_ms": 10, "error": None, **r}
                                              for r in rows])


def test_metrics_match_hand_computation(tmp_path):
    m = manifest(tmp_path, [{"case_id": f"c{i}", "labels": {"Cardiomegaly": y}} for i, y in enumerate([1, 1, 0, 0])])
    p = preds(tmp_path, [{"case_id": f"c{i}", "scores": {"Cardiomegaly": s}} for i, s in enumerate([0.9, 0.4, 0.6, 0.1])])
    rep = evaluate(p, m, None, "test")
    e = next(x for x in rep["per_label"] if x["label"] == "Cardiomegaly")
    assert (e["tp"], e["fn"], e["fp"], e["tn"]) == (1, 1, 1, 1)
    assert e["sensitivity"] == 0.5 and e["specificity"] == 0.5 and e["auroc"] == 0.75
    assert rep["decision_rule"] == "score >= 0.5" and "ci_note" in e  # too few cases for a CI


def test_missing_and_failed_predictions_counted(tmp_path):
    m = manifest(tmp_path, [{"case_id": "a", "labels": {"Cardiomegaly": 1}}, {"case_id": "b", "labels": {}},
                            {"case_id": "c", "labels": {}}])
    p = preds(tmp_path, [{"case_id": "a", "scores": {"Cardiomegaly": 0.7}},
                         {"case_id": "b", "scores": None, "error": "image missing on disk"},
                         {"case_id": "zzz", "scores": {"Cardiomegaly": 0.1}}])
    rep = evaluate(p, m, None, "test")
    assert rep["missing_predictions"] == 1 and rep["failed"] == 1 and rep["cases_evaluated"] == 1
    assert any("not in the manifest" in w for w in rep["warnings"])


@pytest.mark.parametrize("policy,expected_n,expected_pos", [("exclude", 2, 1), ("positive", 3, 2), ("negative", 3, 1)])
def test_uncertainty_policies(tmp_path, policy, expected_n, expected_pos):
    m = manifest(tmp_path, [{"case_id": "a", "labels": {"Edema": 1}}, {"case_id": "b", "labels": {"Edema": 0}},
                            {"case_id": "c", "label_uncertainty": ["Edema"]}])
    p = preds(tmp_path, [{"case_id": c, "scores": {"Edema": s}} for c, s in [("a", .8), ("b", .2), ("c", .5)]])
    rep = evaluate(p, m, None, "test", uncertainty_policy=policy)
    e = rep["per_label"][0]
    assert (e["n"], e["positives"]) == (expected_n, expected_pos)
    if policy == "exclude":
        assert rep["class_distribution"]["Edema"]["uncertain_excluded"] == 1


def test_positives_only_labels_with_label_map_and_contamination_warning(tmp_path):
    lm = tmp_path / "map.yaml"
    lm.write_text("map:\n  Cardiomegaly: Cardiomegaly\n  Pleural Effusion: Edema\n")
    rows = [{"case_id": "a", "labels": {"Cardiomegaly": 1}, "metadata": {"labels_are_positives_only": True}},
            {"case_id": "b", "labels": {}, "metadata": {"labels_are_positives_only": True}}]
    m = write_jsonl(tmp_path / "m.jsonl", [{"dataset_source": "iu_xray", "split": "test", **r} for r in rows])
    model = MODEL | {"training_datasets": ["iu_xray"]}
    p = write_jsonl(tmp_path / "p.jsonl", [{"case_id": c, "model": model, "scores": {"Cardiomegaly": s, "Edema": 0.1},
                                            "above_operating_point": {"Cardiomegaly": s > .5, "Edema": False}}
                                           for c, s in [("a", .9), ("b", .2)]])
    rep = evaluate(p, m, lm, "test")
    car = next(x for x in rep["per_label"] if x["label"] == "Cardiomegaly")
    assert car["positives"] == 1 and car["negatives"] == 1 and car["auroc"] == 1.0
    assert "operating points" in rep["decision_rule"]
    assert any("NOT A HELD-OUT" in w for w in rep["warnings"])
    assert any("closed-world" in w for w in rep["warnings"])


def test_bootstrap_ci_present_with_enough_cases(tmp_path):
    ys = [1] * 10 + [0] * 10
    ss = [0.6 + i * 0.01 for i in range(10)] + [0.3 + i * 0.03 for i in range(10)]
    m = manifest(tmp_path, [{"case_id": f"c{i}", "labels": {"Cardiomegaly": y}} for i, y in enumerate(ys)])
    p = preds(tmp_path, [{"case_id": f"c{i}", "scores": {"Cardiomegaly": s}} for i, s in enumerate(ss)])
    e = evaluate(p, m, None, "test")["per_label"][0]
    lo, hi = e["ci95"]["auroc"]
    assert lo <= e["auroc"] <= hi


def test_report_files_and_empty_listing(tmp_path):
    assert list_reports(tmp_path) == []
    m = manifest(tmp_path, [{"case_id": "a", "labels": {"Cardiomegaly": 1}}, {"case_id": "b", "labels": {"Cardiomegaly": 0}}])
    p = preds(tmp_path, [{"case_id": "a", "scores": {"Cardiomegaly": .9}}, {"case_id": "b", "scores": {"Cardiomegaly": .1}}])
    out = save_report(evaluate(p, m, None, "test"), tmp_path / "metrics.json")
    assert out.with_suffix(".md").exists()
    assert list_reports(tmp_path)[0]["file"] == "metrics.json"
    assert json.loads(out.read_text())["interpretation"].startswith("Retrospective")

"""Per-label evaluation of image-model predictions against manifest labels."""
from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from app.schemas.dataset import ManifestRecord
from app.services.manifest_service import load_records

UNCERTAINTY_POLICIES = ("exclude", "positive", "negative")
N_BOOT = 1000
MIN_PER_CLASS_FOR_CI = 5


def load_label_map(path: Path | None) -> dict[str, str]:
    """{dataset label: model label}. Missing file -> identity mapping only."""
    if not path or not Path(path).exists():
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (data.get("map") or {}).items()}


def ground_truth(rec: ManifestRecord, label_map: dict[str, str], model_labels: set[str],
                 policy: str) -> dict[str, int]:
    """Labels in model-label space for one record. Absent key = not evaluable for that label."""
    gt: dict[str, int] = {}
    for ds_label, value in rec.labels.items():
        m = label_map.get(ds_label, ds_label)
        if m in model_labels:
            gt[m] = max(gt.get(m, 0), int(value))
    if rec.metadata.get("labels_are_positives_only"):
        # Closed-world assumption: a mapped label not assigned to the report counts as negative.
        for m in set(label_map.values()) & model_labels:
            gt.setdefault(m, 0)
    for ds_label in rec.label_uncertainty:
        m = label_map.get(ds_label, ds_label)
        if m not in model_labels or gt.get(m) == 1:
            continue
        if policy == "positive":
            gt[m] = 1
        elif policy == "negative":
            gt.setdefault(m, 0)
        else:
            gt.pop(m, None)
            gt[f"__uncertain__{m}"] = 1
    return gt


def _binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())

    def div(a, b):
        return a / b if b else None

    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    f1 = div(2 * prec * sens, prec + sens) if prec is not None and sens is not None and (prec + sens) else None
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": sens, "recall": sens, "specificity": spec,
            "precision": prec, "f1": f1, "accuracy": div(tp + tn, len(y))}


def _bootstrap_ci(y, s, pred, rng) -> dict:
    stats = defaultdict(list)
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        stats["auroc"].append(roc_auc_score(yb, s[idx]))
        m = _binary_metrics(yb, pred[idx])
        for k in ("sensitivity", "specificity"):
            if m[k] is not None:
                stats[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in stats.items() if len(v) >= N_BOOT * 0.5}


def evaluate(predictions_path: Path, manifest_path: Path, label_map_path: Path | None = None,
             split: str | None = "test", uncertainty_policy: str = "exclude", threshold: float = 0.5,
             seed: int = 0) -> dict:
    if uncertainty_policy not in UNCERTAINTY_POLICIES:
        raise ValueError(f"uncertainty_policy must be one of {UNCERTAINTY_POLICIES}")
    records = {r.case_id: r for r in load_records(manifest_path)}
    in_split = {cid: r for cid, r in records.items() if split is None or r.split == split}
    preds_by_case: dict[str, list[dict]] = defaultdict(list)
    model_info: dict = {}
    for line in predictions_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            preds_by_case[p["case_id"]].append(p)
            model_info = model_info or p.get("model", {})
    label_map = load_label_map(label_map_path)
    model_labels = set(model_info.get("labels") or {k for ps in preds_by_case.values() for p in ps
                                                     for k in (p.get("scores") or {})})

    warnings: list[str] = []
    failed, latencies, missing = [], [], []
    case_scores: dict[str, dict[str, float]] = {}
    case_flags: dict[str, dict[str, bool]] = {}
    for cid in in_split:
        preds = preds_by_case.get(cid)
        if not preds:
            missing.append(cid)
            continue
        ok = [p for p in preds if not p.get("error") and p.get("scores")]
        latencies += [p["latency_ms"] for p in preds if p.get("latency_ms") is not None]
        if not ok:
            failed.append(cid)
            continue
        # several images per case: take the maximum score per label (documented aggregation)
        scores, flags = {}, {}
        for p in ok:
            for label, val in p["scores"].items():
                scores[label] = max(scores.get(label, -math.inf), val)
            for label, val in (p.get("above_operating_point") or {}).items():
                flags[label] = flags.get(label, False) or bool(val)
        case_scores[cid], case_flags[cid] = scores, flags
    extra_preds = sorted(set(preds_by_case) - set(records))
    if extra_preds:
        warnings.append(f"{len(extra_preds)} prediction case_id(s) are not in the manifest and were ignored.")

    use_op = bool(case_flags) and all(case_flags.values())
    decision_rule = ("model's published operating points (above_operating_point flags)" if use_op
                     else f"score >= {threshold}")

    per_label, distribution = [], {}
    uncertain_counts = defaultdict(int)
    rng = np.random.default_rng(seed)
    for label in sorted(model_labels):
        ys, ss, ps = [], [], []
        for cid, scores in case_scores.items():
            gt = ground_truth(in_split[cid], label_map, model_labels, uncertainty_policy)
            if f"__uncertain__{label}" in gt:
                uncertain_counts[label] += 1
            if label not in gt or label not in scores:
                continue
            ys.append(gt[label]); ss.append(scores[label])
            ps.append(int(case_flags[cid].get(label, False)) if use_op else int(scores[label] >= threshold))
        if not ys:
            continue
        y, s, pr = np.array(ys), np.array(ss), np.array(ps)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        distribution[label] = {"positive": n_pos, "negative": n_neg,
                               "uncertain_excluded": uncertain_counts[label] if uncertainty_policy == "exclude" else 0}
        entry = {"label": label, "n": len(y), "positives": n_pos, "negatives": n_neg,
                 "prevalence": round(n_pos / len(y), 4), **_binary_metrics(y, pr), "auroc": None, "ci95": {}}
        if n_pos and n_neg:
            entry["auroc"] = float(roc_auc_score(y, s))
            if n_pos >= MIN_PER_CLASS_FOR_CI and n_neg >= MIN_PER_CLASS_FOR_CI:
                entry["ci95"] = _bootstrap_ci(y, s, pr, rng)
            else:
                entry["ci_note"] = f"fewer than {MIN_PER_CLASS_FOR_CI} cases in a class; no confidence interval"
        else:
            entry["auroc_note"] = "only one class present; AUROC undefined"
        if n_pos and (n_pos / len(y) < 0.05):
            entry["imbalance_note"] = "prevalence below 5%; precision and F1 are unstable"
        per_label.append(entry)

    def macro(key):
        vals = [e[key] for e in per_label if e.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    dataset_ids = sorted({r.dataset_source for r in in_split.values()})
    trained_on = set(model_info.get("training_datasets") or [])
    overlap = sorted(trained_on & set(dataset_ids))
    if overlap:
        warnings.append(
            f"NOT A HELD-OUT EVALUATION: the model's published training data includes {overlap}. "
            "Its training split is not known to match this split, so these metrics are likely optimistic.")
    if not label_map and any(r.metadata.get("labels_are_positives_only") for r in in_split.values()):
        warnings.append("Dataset labels are positives-only but no label map was given; negatives cannot be formed.")
    if any(r.synthetic for r in in_split.values()):
        warnings.append("Synthetic records are present; metrics on them are meaningless.")
    if len(case_scores) < 100:
        warnings.append(f"Only {len(case_scores)} cases evaluated; estimates are imprecise.")
    if any(r.metadata.get("labels_are_positives_only") for r in in_split.values()):
        warnings.append("Ground truth derives from report MeSH terms (weak labels) with a closed-world "
                        "assumption: unmentioned findings are treated as negative.")

    return {
        "report_id": uuid.uuid4().hex[:12],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {k: model_info.get(k) for k in ("backend", "name", "version", "training_datasets")},
        "dataset": dataset_ids,
        "manifest": manifest_path.name,
        "split": split or "all",
        "predictions_file": predictions_path.name,
        "label_map": label_map_path.name if label_map_path else None,
        "uncertainty_policy": uncertainty_policy,
        "decision_rule": decision_rule,
        "case_aggregation": "max score over a case's images",
        "cases_in_split": len(in_split),
        "cases_evaluated": len(case_scores),
        "successful": len(case_scores),
        "failed": len(failed),
        "missing_predictions": len(missing),
        "avg_latency_ms": round(float(np.mean(latencies)), 1) if latencies else None,
        "per_label": per_label,
        "macro": {k: macro(k) for k in ("auroc", "sensitivity", "specificity", "precision", "f1", "accuracy")},
        "class_distribution": distribution,
        "warnings": warnings,
        "interpretation": "Retrospective agreement with dataset labels on one split. Not evidence of clinical validity.",
    }


def save_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [f"# Evaluation {report['report_id']}", "", f"- Generated: {report['generated_at']}",
          f"- Model: {report['model'].get('name')} {report['model'].get('version') or ''}",
          f"- Dataset / split: {report['dataset']} / {report['split']}",
          f"- Cases evaluated: {report['cases_evaluated']} of {report['cases_in_split']} "
          f"(failed {report['failed']}, missing {report['missing_predictions']})",
          f"- Decision rule: {report['decision_rule']}; uncertainty policy: {report['uncertainty_policy']}", "",
          "| Label | n | pos | AUROC | Sens | Spec | F1 |", "|---|---|---|---|---|---|---|"]

    def f(v):
        return "-" if v is None else f"{v:.3f}"

    for e in report["per_label"]:
        md.append(f"| {e['label']} | {e['n']} | {e['positives']} | {f(e['auroc'])} | {f(e['sensitivity'])} | "
                  f"{f(e['specificity'])} | {f(e['f1'])} |")
    md += ["", "## Warnings"] + [f"- {w}" for w in report["warnings"]] + ["", report["interpretation"]]
    out.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return out


def list_reports(eval_dir: Path) -> list[dict]:
    out = []
    for p in sorted(eval_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "per_label" in data:
            out.append(data | {"file": p.name})
    return out

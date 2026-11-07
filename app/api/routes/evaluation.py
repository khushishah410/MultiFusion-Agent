from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import AppState, get_state
from app.core.paths import UnsafePathError, safe_join
from app.services.evaluation_service import UNCERTAINTY_POLICIES, evaluate, list_reports, save_report

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class RunRequest(BaseModel):
    predictions_file: str = Field(description="File name inside the evaluation directory (from scripts.predict_dataset)")
    split: str | None = "test"
    uncertainty_policy: str = "exclude"
    threshold: float = Field(0.5, ge=0, le=1)


@router.get("/summary")
def summary(state: AppState = Depends(get_state)) -> dict:
    reports = list_reports(state.settings.evaluation_dir_abs) if state.settings.evaluation_dir_abs.exists() else []
    if not reports:
        return {"count": 0, "latest": None,
                "how_to_generate": ["python -m scripts.predict_dataset --split test",
                                    "python -m scripts.evaluate --input reports/evaluation/predictions_<dataset>_test.jsonl"]}
    return {"count": len(reports), "latest": reports[0]}


@router.get("/reports")
def reports(state: AppState = Depends(get_state)) -> list[dict]:
    d = state.settings.evaluation_dir_abs
    return list_reports(d) if d.exists() else []


@router.get("/predictions")
def prediction_files(state: AppState = Depends(get_state)) -> list[str]:
    d = state.settings.evaluation_dir_abs
    return sorted(p.name for p in d.glob("*.jsonl")) if d.exists() else []


@router.post("/run")
def run(req: RunRequest, state: AppState = Depends(get_state)) -> dict:
    if req.uncertainty_policy not in UNCERTAINTY_POLICIES:
        raise HTTPException(422, f"uncertainty_policy must be one of {UNCERTAINTY_POLICIES}")
    eval_dir = state.settings.evaluation_dir_abs
    try:
        pred_path = safe_join(eval_dir, req.predictions_file)
    except UnsafePathError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not pred_path.is_file():
        raise HTTPException(404, "Predictions file not found in the evaluation directory.")
    manifest = state.store.manifest
    if manifest is None or not manifest.exists():
        raise HTTPException(409, "No active manifest to evaluate against.")
    first = json.loads(pred_path.read_text(encoding="utf-8").splitlines()[0])
    dataset = first.get("dataset_source") or manifest.stem
    label_map = state.settings.resolve(f"config/label_maps/{dataset}.yaml")
    report = evaluate(pred_path, manifest, label_map if label_map.exists() else None,
                      req.split, req.uncertainty_policy, req.threshold)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = save_report(report, eval_dir / f"metrics_{dataset}_{req.split or 'all'}_{stamp}.json")
    return report | {"file": out.name}

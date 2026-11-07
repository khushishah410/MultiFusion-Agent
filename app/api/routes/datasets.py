from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.api.dependencies import AppState, get_state
from app.services.dataset_ops import ConfirmationRequired, DatasetOpError

router = APIRouter(tags=["datasets"])


class ValidateRequest(BaseModel):
    dataset_id: str
    check_images: bool = True
    max_images: int | None = Field(None, ge=1)


class DownloadRequest(BaseModel):
    metadata_only: bool = False
    confirm_large: bool = False


class ImportRequest(BaseModel):
    source_dir: str = Field(min_length=1)


def _ops_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ConfirmationRequired as exc:
        raise HTTPException(409, str(exc)) from exc
    except (DatasetOpError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------- registry (/datasets)
@router.get("/datasets")
def list_datasets(state: AppState = Depends(get_state)) -> list[dict]:
    return [spec.public_dict() for spec in state.dataset_ops.registry.values()]


@router.get("/datasets/status")
def datasets_status(state: AppState = Depends(get_state)) -> list[dict]:
    return [state.dataset_ops.status(d) for d in state.dataset_ops.registry]


@router.get("/datasets/jobs")
def list_jobs(state: AppState = Depends(get_state)) -> list[dict]:
    return state.jobs.all()


@router.get("/datasets/jobs/{job_id}")
def get_job(job_id: str, state: AppState = Depends(get_state)) -> dict:
    job = state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id.")
    return job


@router.post("/datasets/validate")
def validate(req: ValidateRequest, state: AppState = Depends(get_state)) -> dict:
    return _ops_call(state.dataset_ops.validate, req.dataset_id, req.check_images, req.max_images)


@router.get("/datasets/{dataset_id}")
def inspect_dataset(dataset_id: str, check_remote: bool = False, state: AppState = Depends(get_state)) -> dict:
    info = _ops_call(state.dataset_ops.inspect, dataset_id, check_remote)
    info.pop("storage_dir", None)
    return info


@router.post("/datasets/{dataset_id}/download", status_code=202)
def start_download(dataset_id: str, req: DownloadRequest, state: AppState = Depends(get_state)) -> dict:
    ops = state.dataset_ops
    files = _ops_call(ops.plan_download, dataset_id, req.metadata_only)
    total = sum(f.documented_size_mb or 0 for f in files)
    if total > state.settings.large_download_mb and not req.confirm_large:
        raise HTTPException(409, f"Download is about {total:,.0f} MB; set confirm_large=true to proceed.")

    def work(report):
        return ops.download(dataset_id, req.metadata_only, True, False,
                            lambda name, done, tot: report({"file": name, "bytes": done, "total": tot}))

    return {"job_id": state.jobs.start(f"dl:{dataset_id}", "download", work), "documented_total_mb": total}


@router.post("/datasets/{dataset_id}/import")
def import_local(dataset_id: str, req: ImportRequest, state: AppState = Depends(get_state)) -> dict:
    return _ops_call(state.dataset_ops.import_local, dataset_id, Path(req.source_dir))


@router.post("/datasets/{dataset_id}/prepare", status_code=202)
def start_prepare(dataset_id: str, state: AppState = Depends(get_state)) -> dict:
    ops = state.dataset_ops
    _ops_call(ops.spec, dataset_id)

    def work(report):
        report({"step": "building manifest"})
        meta = ops.prepare(dataset_id)
        report({"step": "validating"})
        rep = ops.validate(dataset_id, check_images=True)
        return {"records_written": meta["records_written"], "validation_ok": rep["ok"],
                "missing_image_files": rep["missing_image_files"]}

    return {"job_id": state.jobs.start(f"prep:{dataset_id}", "prepare", work)}


@router.post("/datasets/{dataset_id}/activate")
def activate(dataset_id: str, state: AppState = Depends(get_state)) -> dict:
    res = _ops_call(state.dataset_ops.activate, dataset_id)
    state.reload()
    return res | {"note": "Activated and reloaded."}


# ---------------------------------------------------------------- active manifest (/dataset)
@router.get("/dataset/summary")
def dataset_summary(state: AppState = Depends(get_state)) -> dict:
    summary = state.store.summary()
    manifest = state.store.manifest
    if manifest:
        vpath = manifest.with_name(manifest.stem + ".validation.json")
        if vpath.exists():
            v = json.loads(vpath.read_text(encoding="utf-8"))
            summary["validation"] = {k: v.get(k) for k in (
                "generated_at", "ok", "missing_image_files", "records_with_missing_images", "unreadable_images",
                "malformed_records", "duplicate_case_ids", "field_coverage", "leakage", "image_size_stats")}
    return summary


def _filtered(state, q, split, label, has_images):
    if state.store.error:
        raise HTTPException(503, state.store.error)
    return state.store.search(q, split, label, has_images)


@router.get("/dataset/cases")
def list_cases(q: str | None = Query(None, max_length=200), split: str | None = None, label: str | None = None,
               has_images: bool | None = None, page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=200),
               state: AppState = Depends(get_state)) -> dict:
    rows = _filtered(state, q, split, label, has_images)
    start = (page - 1) * page_size
    items = [state.store.to_public(r).model_dump() for r in rows[start:start + page_size]]
    return {"total": len(rows), "page": page, "page_size": page_size, "items": items}


@router.get("/dataset/cases/{case_id}")
def get_case(case_id: str, state: AppState = Depends(get_state)) -> dict:
    rec = state.store.by_id.get(case_id)
    if rec is None:
        raise HTTPException(404, f"Case {case_id!r} not found.")
    return state.store.to_public(rec).model_dump()


@router.get("/dataset/cases/{case_id}/images/{index}")
def get_case_image(case_id: str, index: int, state: AppState = Depends(get_state)):
    rec = state.store.by_id.get(case_id)
    if rec is None:
        raise HTTPException(404, f"Case {case_id!r} not found.")
    path = state.store.image_path(rec, index)
    if path is None:
        raise HTTPException(404, "Image not available on disk.")
    return FileResponse(path, media_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg")


@router.get("/dataset/export")
def export_cases(fmt: str = Query("csv", pattern="^(csv|json)$"), q: str | None = None, split: str | None = None,
                 label: str | None = None, state: AppState = Depends(get_state)) -> Response:
    rows = [state.store.to_public(r).model_dump() for r in _filtered(state, q, split, label, None)]
    if fmt == "json":
        return Response(json.dumps(rows, indent=2), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=cases.json"})
    buf = io.StringIO()
    cols = ["case_id", "dataset_source", "split", "synthetic", "clinical_indication", "radiology_findings",
            "radiology_impression", "image_ids", "labels", "label_uncertainty", "mesh_major"]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=cases.csv"})

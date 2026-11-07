from __future__ import annotations

import importlib.util
import platform
import sys

from fastapi import APIRouter, Depends

from app.api.dependencies import AppState, get_state
from app.services.evaluation_service import list_reports

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/status")
def status(state: AppState = Depends(get_state)) -> dict:
    s = state.settings
    deps = {name: importlib.util.find_spec(name) is not None
            for name in ("torch", "torchxrayvision", "sklearn", "PIL", "streamlit")}
    reports = list_reports(s.evaluation_dir_abs) if s.evaluation_dir_abs.exists() else []
    summary = state.store.summary()
    return {
        "backend": {"status": "ok", "python": sys.version.split()[0], "platform": platform.platform()},
        "dataset_manifest": {k: summary[k] for k in ("manifest", "selected_by", "error", "case_count",
                                                     "image_count", "synthetic", "dataset_sources")},
        "datasets": [state.dataset_ops.status(d) for d in state.dataset_ops.registry],
        "image_model": state.adapter.info().model_dump(),
        "symptom_agent": {"method": state.symptom_agent.method, "lexicon_version": state.symptom_agent.lexicon_version},
        "retrieval_index": state.retrieval_agent.status(),
        "evaluation_reports": {"count": len(reports), "latest": reports[0]["generated_at"] if reports else None},
        "dependencies": deps,
        "config": {"max_upload_mb": s.max_upload_mb, "retain_uploads": s.retain_uploads,
                   "model_backend": s.model_backend, "model_weights": s.model_weights},
    }


@router.post("/admin/reload")
def reload(state: AppState = Depends(get_state)) -> dict:
    """Re-read the active manifest, retrieval index and label maps (after prepare/activate)."""
    state.reload()
    return {"reloaded": True, "manifest": state.store.summary()["manifest"],
            "retrieval": state.retrieval_agent.status()["status"]}

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.dependencies import AppState, get_state
from app.schemas.analysis import FusionResult, ImageAnalysis, SymptomAnalysis, SymptomRequest

log = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analysis"])


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    data = await upload.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Upload exceeds the {max_bytes / 1e6:.1f} MB limit.")
    return data


def _retrieval_query(sym: SymptomAnalysis) -> str:
    terms = [m.canonical.replace("_", " ") for m in sym.symptom_mentions]
    terms += [u.text for u in sym.unrecognized_terms]
    return " ".join(terms)


def run_pipeline(state: AppState, text: str | None, context: str | None, duration: str | None,
                 image_bytes: bytes | None, case_id: str | None, use_case_image: int | None,
                 retrieve: bool) -> FusionResult:
    timings = {}
    reference = None
    image_is_reference = False
    if case_id:
        rec = state.store.by_id.get(case_id)
        if rec is None:
            raise HTTPException(404, f"Case {case_id!r} not found in the active manifest.")
        reference = state.store.to_public(rec).model_dump()
        if use_case_image is not None and image_bytes is None:
            path = state.store.image_path(rec, use_case_image)
            if path is None:
                raise HTTPException(404, f"Image {use_case_image} of case {case_id} is not available on disk.")
            image_bytes = path.read_bytes()
            image_is_reference = True

    symptoms = None
    if text and text.strip() or context or duration:
        t0 = time.perf_counter()
        symptoms = state.symptom_agent.analyze(text or "", context, duration)
        timings["symptom_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    image = None
    if image_bytes is not None:
        image = state.image_agent.analyze(image_bytes)
        timings["image_ms"] = image.latency_ms

    retrieval = None
    if retrieve and symptoms is not None:
        t0 = time.perf_counter()
        query = _retrieval_query(symptoms) or symptoms.normalized_text
        retrieval = state.retrieval_agent.search(query)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if symptoms is None and image is None:
        raise HTTPException(422, "Provide symptom text, an image, or a dataset case image.")
    diagnosis = state.diagnosis_agent.suggest(symptoms, image)
    log.info("analyze: text_chars=%d image=%s case=%s", len(text or ""), image is not None, bool(case_id))
    return state.fusion_agent.fuse(symptoms, image, retrieval, reference, state.store.summary().get("source") or None,
                                   timings, image_is_reference_image=image_is_reference, diagnosis=diagnosis)


@router.post("", response_model=FusionResult)
async def analyze(
    text: str | None = Form(None, max_length=5000),
    context: str | None = Form(None, max_length=2000),
    duration: str | None = Form(None, max_length=200),
    case_id: str | None = Form(None, description="Attach a dataset case as a reference record"),
    use_case_image: int | None = Form(None, ge=0, description="Analyse image N of the reference case (no upload)"),
    retrieve: bool = Form(True),
    image: UploadFile | None = File(None),
    state: AppState = Depends(get_state),
) -> FusionResult:
    data = await _read_upload(image, state.settings.max_upload_bytes) if image is not None else None
    return await run_in_threadpool(run_pipeline, state, text, context, duration, data, case_id, use_case_image, retrieve)


@router.post("/symptoms", response_model=SymptomAnalysis)
def analyze_symptoms(req: SymptomRequest, state: AppState = Depends(get_state)) -> SymptomAnalysis:
    if not req.text.strip():
        raise HTTPException(422, "Symptom text is empty.")
    return state.symptom_agent.analyze(req.text, req.context, req.duration)


@router.post("/image", response_model=ImageAnalysis)
async def analyze_image(image: UploadFile = File(...), state: AppState = Depends(get_state)) -> ImageAnalysis:
    data = await _read_upload(image, state.settings.max_upload_bytes)
    return await run_in_threadpool(state.image_agent.analyze, data)

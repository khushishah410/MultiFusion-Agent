# Handoff

State as of 2026-10-27. Enough for another developer or assistant to continue without the original conversation.

## Module map

| Path | Responsibility |
|---|---|
| `app/core/config.py` | Typed `Settings` (YAML → .env → env vars `MULTIFUSION_*`); `active_manifest()` picks configured → `active.json` → synthetic demo |
| `app/core/paths.py` | `safe_join`, `is_unsafe_relative` (path traversal guard used everywhere) |
| `app/schemas/analysis.py` | Symptom, image, retrieval, fusion schemas (Pydantic v2) |
| `app/schemas/dataset.py` | `ManifestRecord`, `PublicCase` (no patient_id), `ValidationReport` |
| `app/agents/symptom_agent.py` | `RuleBasedSymptomAgent`, `DatasetTextIndex`, `SymptomExtractor` protocol for future NLP models |
| `app/agents/image_agent.py` | `validate_image`, `to_grayscale_array`, `ImageAgent` |
| `app/agents/model_adapters.py` | `TorchXRayVisionAdapter`, `UnavailableAdapter`, `build_adapter`, training-data map |
| `app/agents/retrieval_agent.py` | BM25 index build and search |
| `app/agents/diagnosis_agent.py` | Point-scored suggested diagnosis + differentials over `config/diagnosis_rules.yaml` |
| `app/agents/fusion_agent.py` | Deterministic fusion with evidence ids; no combined score |
| `app/services/registry.py` | Loads `config/datasets.yaml` |
| `app/services/downloader.py` | Streaming, resumable downloads; status file |
| `app/services/archive.py` | Safe tar/zip extraction |
| `app/services/torrent.py` | Reads single-file `.torrent` metadata; verifies a file against its SHA-1 piece hashes (integrity only, no P2P) |
| `app/services/manifest_builders.py` | IU XML, NIH CSV, CheXpert CSV, MIMIC CSV, generic builders |
| `app/services/manifest_service.py` | Normalise, dedupe, split, write, validate, `ManifestStore` |
| `app/services/dataset_ops.py` | Shared by CLI and API: inspect/download/import/prepare/validate/activate |
| `app/services/evaluation_service.py` | Metrics, CIs, uncertainty policy, contamination warnings, report I/O |
| `app/services/jobs.py` | In-process background jobs for API downloads/prepare |
| `app/api/` | FastAPI app, `AppState` (reloadable), routes: health, analyze, datasets, evaluation |
| `app/frontend/` | Streamlit entry `frontend_ui.py`, `api_client.py`, `views/` (5 pages), `components/ui.py`, `styles/theme.css` |
| `scripts/` | CLIs: dataset_manager, generate_demo_data, build_retrieval_index, download_model, predict_dataset, evaluate, train_model, validate_dataset, build_manifest, prepare_images |
| `config/` | settings, dataset registry, symptom lexicon, label maps |

## Data flow for `POST /analyze`

1. Optional reference case looked up in `ManifestStore`; with `use_case_image` its image bytes are read through `safe_join`.
2. `SymptomAgent.analyze(text, context, duration)` (context and duration appended as labelled lines so offsets stay valid).
3. `ImageAgent.analyze(bytes)`: validation, then adapter inference only if loaded.
4. Retrieval query = canonical names of affirmed symptoms + unrecognised terms (negated ones excluded).
5. `DiagnosisAgent.suggest(symptoms, image)` scores each rule; result is passed into fusion and shown first.
6. `FusionAgent.fuse(...)` → `FusionResult` with mode `demo | text_only | image_validation_only | image_model | multimodal`. `demo` whenever synthetic data contributed. Label disagreements with a reference case are computed only when the analysed image is that case's own image.

Uploads stay in memory. Logs record text length, not text.

## API

`GET /health`, `GET /status`, `POST /admin/reload`, `POST /analyze` (multipart), `POST /analyze/symptoms` (JSON), `POST /analyze/image`, `GET /datasets`, `GET /datasets/status`, `GET /datasets/jobs[/{id}]`, `POST /datasets/validate`, `GET /datasets/{id}`, `POST /datasets/{id}/download|import|prepare|activate`, `GET /dataset/summary`, `GET /dataset/cases`, `GET /dataset/cases/{id}`, `GET /dataset/cases/{id}/images/{n}`, `GET /dataset/export`, `GET /evaluation/summary|reports|predictions`, `POST /evaluation/run`. Static `/datasets/...` routes are declared before `/datasets/{id}`.

## What was verified in the build environment (Linux, Python 3.12.13 via uv)

| Item | Result |
|---|---|
| `pytest` | 93 passed (includes the real-model test, since weights were present) |
| Backend over HTTP | `/health`, `/status`, `/docs` 200; multimodal `/analyze` with a real X-ray returned TorchXRayVision scores (~330 ms image step) |
| Streamlit | All 5 pages rendered with no exceptions (AppTest), and screenshots taken in headless Chromium |
| Synthetic demo | Generated, manifest built and validated (1 intentional missing image), retrieval index built (10 docs, test split excluded) |
| End-to-end scripts | prepare → validate → activate → predict_dataset (real model) → evaluate → build_retrieval_index on a 40-case IU-structured fixture of noise images; pipeline ran; metrics on noise are meaningless and are not reported |
| IU X-Ray real download | **Not done**: Open-i and all torrent web-seed mirrors blocked by the build environment's network. Mirror fallback, size checks and torrent piece verification are tested with fixture archives |
| Windows | **Not run on Windows.** Commands are written for PowerShell; paths use pathlib |

## Known limitations / next steps

0. The diagnosis rules are hand-weighted and unvalidated; the demo case (NIH 00000001_000 + exertional breathlessness and ankle swelling) yields heart failure, which is plausible but not evidence of accuracy.

1. Run the IU download + prepare on Windows and inspect build issues; adjust `config/label_maps/iu_xray.yaml` to the real MeSH heading counts.
2. Frontal/lateral view is not distinguished in IU (first image is used by the probe; evaluation takes max over all images).
3. IU has no patient ids; splits are grouped by report id.
4. Symptom lexicon is small and hand-written; a medical NER model can be plugged in via `SymptomExtractor`.
5. No DICOM support (VinDr-CXR needs external conversion).
6. API has no authentication; keep it on 127.0.0.1. `import` accepts any local folder path.
7. Background jobs live in memory and are lost on backend restart (status files on disk survive).
8. The `all` weights have seen every registered dataset; use `-nih` weights for IU evaluation.

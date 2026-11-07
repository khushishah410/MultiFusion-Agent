# MultiFusionAgent

An educational research prototype that analyses new free-text symptoms and chest X-ray images, keeps the output of each component separate, and records where every statement came from. It runs on a Windows laptop with Python 3.12 and no GPU.

> **Not a medical device.** It has not been clinically validated. Its suggested diagnosis is a transparent rule-based score, not a probability or a clinical diagnosis, and must be confirmed by a doctor.

## What it does

| Component | What it does | Status out of the box |
|---|---|---|
| SymptomAgent | Rule-based extraction from any text: lexicon match (exact and fuzzy for misspellings), NegEx-style negation, hedging cues, durations, context, and unrecognised terms kept verbatim | Works offline |
| ImageAgent | Validates uploads by content (PNG/JPEG, size, dimensions, corruption) and runs TorchXRayVision DenseNet-121 when its weights are present | Validation always; inference after `requirements-model.txt` + `scripts.download_model` |
| RetrievalAgent | BM25 search over a local index built from prepared dataset reports and your own files in `data/knowledge/` | After `scripts.build_retrieval_index` |
| DiagnosisAgent | Suggests the most likely condition and up to 4 alternatives from reported/denied symptoms, context and X-ray model flags, using point rules in `config/diagnosis_rules.yaml` (11 chest conditions, symptoms and confirming tests from cited MedlinePlus pages). Shows why, what argues against it, and confirming tests | Works offline; uses the image model when loaded |
| FusionAgent | Puts the suggested diagnosis first, then what each component found, the evidence ids behind each statement, disagreements, missing parts and uncertainties | Works offline |
| Dataset manager | Registry of 6 public CXR datasets; automatic download for IU X-Ray (Open-i), local import for gated datasets; manifest building, validation, grouped splits, leakage checks | Synthetic demo works offline |
| Evaluation | Per-label sensitivity, specificity, precision, F1, accuracy, AUROC with bootstrap 95% CIs; uncertainty policies; contamination warnings | Needs predictions from a real model and labelled data |

Architecture:

```
Streamlit UI (app/frontend)  ──HTTP──▶  FastAPI (app/api)
                                         │
       ┌──────────────┬──────────────────┼───────────────────┐
  SymptomAgent    ImageAgent ──▶ model adapter     RetrievalAgent (BM25 index)
       └──────────────┴────────▶ FusionAgent ◀─────┘
                                         │
                   ManifestStore (active manifest, read-only)
                   DatasetOps: registry → download/import → manifest → validate
```

## Prerequisites

- Windows 10/11, PowerShell, Python 3.12 (`py -3.12 --version`)
- About 1 GB of disk for the core environment; about 1.5 GB more for the full IU X-Ray images; about 1 GB more for CPU PyTorch.

## Installation (PowerShell, in the project folder)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# synthetic demo data + retrieval index (no network)
.\.venv\Scripts\python.exe -m scripts.generate_demo_data
.\.venv\Scripts\python.exe -m scripts.build_retrieval_index
```

Optional, for real image inference (CPU wheels):

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m scripts.download_model      # shows the size (28.4 MB) and asks first
```

## Running

Terminal 1 (backend):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 (frontend):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\frontend\frontend_ui.py --server.address 127.0.0.1 --server.port 8501
```

- Frontend: http://127.0.0.1:8501
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

Or use the scripts: `.\run_backend.ps1`, `.\run_frontend.ps1`, `.\run_tests.ps1`.

If PowerShell refuses to run them ("running scripts is disabled on this system"), allow scripts for the current window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This does not change the machine-wide policy. Alternatively run the `python.exe` commands above directly.

The app starts without any dataset, model weights, retrieval index or evaluation report; each missing part is shown as unavailable.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

93 tests, no network, GPU or real data (92 pass and 1 is skipped when torchxrayvision is not installed). Downloads are tested with a fake HTTP session. One test runs the real TorchXRayVision model only when its weights are present and is skipped otherwise; the unavailable-model path is always tested.

## Datasets

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_manager list
.\.venv\Scripts\python.exe -m scripts.dataset_manager inspect --dataset iu_xray --check-remote
.\.venv\Scripts\python.exe -m scripts.dataset_manager download --dataset iu_xray --metadata-only   # 1.1 MB reports
.\.venv\Scripts\python.exe -m scripts.dataset_manager download --dataset iu_xray --yes             # + 1.36 GB images
.\.venv\Scripts\python.exe -m scripts.dataset_manager prepare  --dataset iu_xray
.\.venv\Scripts\python.exe -m scripts.dataset_manager validate --dataset iu_xray
.\.venv\Scripts\python.exe -m scripts.dataset_manager activate --dataset iu_xray
.\.venv\Scripts\python.exe -m scripts.build_retrieval_index
.\.venv\Scripts\python.exe -m scripts.dataset_manager status
```

If the Open-i site is down, `download` falls back to the web-seed mirrors listed in the Academic Torrents metadata (archive.org and Academic Torrents servers). You can also fetch the two `.tgz` files yourself (browser or torrent client), put them in `data\raw\iu_xray\`, and run:

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_manager verify  --dataset iu_xray
.\.venv\Scripts\python.exe -m scripts.dataset_manager prepare --dataset iu_xray   # verifies, extracts, builds the manifest
```

Every archive is checked for its exact byte size; `NLMCXR_png.tgz` is also checked piece by piece against the bundled `config/torrents/NLMCXR_png.tgz.torrent` (infohash `5a3a439d...`). An archive that fails is not extracted.

Other options: `--dry-run` (download plan only), `--root D:\data` (storage folder), `cleanup --dataset X` (prints what you may delete; nothing is deleted automatically). Downloads resume from `*.part` files. After `activate`, restart the backend or call `POST /admin/reload`.

Gated datasets (CheXpert, MIMIC-CXR-JPG, VinDr-CXR, PadChest; NIH via Box): obtain them yourself under their terms, then

```powershell
.\.venv\Scripts\python.exe -m scripts.dataset_manager import  --dataset chexpert --source-dir D:\data\CheXpert-v1.0-small
.\.venv\Scripts\python.exe -m scripts.dataset_manager prepare --dataset chexpert
```

Imported folders are referenced in place and never modified. See [DATASET_GUIDE.md](DATASET_GUIDE.md).

## Symptom analysis

Accepts any text, including symptoms absent from every dataset. Known terms are matched to a lexicon (`config/symptom_lexicon.yaml`); spelling variants are matched by string similarity (≥0.85, reported). Everything else that looks like content is returned as an **unrecognised term**, with a check of whether that phrase appears in the active dataset's report text. An empty or unfamiliar description never produces a "no symptoms" or "low risk" result. Red-flag symptoms (chest pain, breathlessness, haemoptysis, syncope, cyanosis, confusion) trigger a fixed notice to seek prompt care, without naming a cause.

## Suggested diagnosis

Each condition in `config/diagnosis_rules.yaml` lists weighted symptoms, context phrases (e.g. smoker) and X-ray labels. A reported symptom adds its weight, a hedged one ("maybe") half, a denied one subtracts it; an X-ray label adds its weight when the model scores it above its published operating point. The highest total (at least 2 points) is the suggestion; the strength is *strong* when it reaches 4 points with both X-ray and symptom support. Symptoms and confirming tests per condition come from the cited MedlinePlus page; which X-ray labels belong to which condition is the author's mapping and should be reviewed. The score is not a probability and the rules have not been validated on patients.

## Image model

TorchXRayVision `densenet121-res224-all` (Apache-2.0). Preprocessing follows the library: grayscale, scale to [-1024, 1024], centre crop, resize to 224. Outputs 18 label scores re-scaled so that 0.5 marks the authors' operating point. See [MODEL_GUIDE.md](MODEL_GUIDE.md), including why this model cannot be evaluated held-out on IU X-Ray.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m scripts.predict_dataset --split test
.\.venv\Scripts\python.exe -m scripts.evaluate --input reports\evaluation\predictions_iu_xray_test.jsonl
```

Reports (`.json` + `.md`) land in `reports/evaluation/` and appear on the Evaluation page. Optional probe training: `python -m scripts.train_model --dataset iu_xray --weights densenet121-res224-nih`.

## Retrieval

`python -m scripts.build_retrieval_index` indexes the active manifest's reports (excluding the test split) plus files in `data/knowledge/` (`.md`, `.txt`, or `.jsonl` with `id,title,text,source[,source_url,source_date]`). Only add material you may use. Retrieved passages are labelled as reference material, never as findings.

## Configuration

`config/settings.yaml`, overridable with environment variables or `.env` (copy `.env.example`): `MULTIFUSION_DATA_ROOT`, `MULTIFUSION_MANIFEST_PATH`, `MULTIFUSION_EVALUATION_DIR`, `MULTIFUSION_MODEL_PATH`, `MULTIFUSION_MODEL_WEIGHTS`, `MULTIFUSION_RETRIEVAL_INDEX`, `MULTIFUSION_MAX_UPLOAD_MB`, `MULTIFUSION_API_URL`, `MULTIFUSION_LOG_LEVEL`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| UI says backend offline | Backend not running or different port | Start `run_backend.ps1`; set `MULTIFUSION_API_URL` |
| Image model "unavailable: Optional packages not installed" | torch/torchxrayvision missing | Install `requirements-model.txt` |
| Image model "Weights file ... not found" | Weights not downloaded | `python -m scripts.download_model` |
| Download "returned an HTML page" or HTTP 403/404 | Link moved, network filter, or login page | Check the official source; download in a browser into `data\raw\iu_xray\` and run `prepare` |
| `ModuleNotFoundError: app` | Command run outside the project folder | `cd` into the folder; use `python -m scripts.<name>` |
| Explorer shows missing images | Only reports downloaded, or extraction incomplete | Download the PNG archive; re-run `prepare` |

## Project layout

See [HANDOFF.md](HANDOFF.md) for module responsibilities and data flow. Deviations from the original spec: Streamlit views live in `app/frontend/views/` (a folder named `pages/` would trigger Streamlit's automatic multipage mode); `requirements-model.txt`, `scripts/download_model.py`, `scripts/predict_dataset.py`, `config/label_maps/` were added.

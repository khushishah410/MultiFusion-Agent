# Model guide

## Selected model

| Item | Value |
|---|---|
| Library | TorchXRayVision 1.5.7 (PyPI, released 2026-10-18), Apache-2.0, https://github.com/mlmed/torchxrayvision |
| Paper | Cohen et al., TorchXRayVision: A library of chest X-ray datasets and models, arXiv:2111.00595 |
| Architecture | DenseNet-121, 224×224 single-channel input, 18 sigmoid outputs |
| Default weights | `densenet121-res224-all` → `nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt` from the library's GitHub release `v1` |
| Weights size | 28,382,008 bytes (measured) ; SHA-256 recorded at download: `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899` |
| Training data (library description) | NIH, PadChest, CheXpert, MIMIC-CXR (CheXpert labels), Google labels on NIH, Open-i (IU X-Ray), RSNA pneumonia |
| Labels | Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum |
| Hardware | CPU; about 0.3–0.45 s per image in the build environment (2 vCPU) after a ~3 s load |

### Why this model

It is the only candidate found with openly downloadable chest X-ray weights, a permissive licence, a maintained pip package that installs on Python 3.12, published preprocessing, and published per-label operating thresholds. ImageNet-only backbones were rejected because they have no chest X-ray training.

### Candidates considered

- Other TorchXRayVision weights (`-nih`, `-chex`, `-pc`, `-mimic_ch`, `-mimic_nb`, `-rsna`, `resnet50-res512-all`): same package; useful for held-out evaluation (below). Select with `MULTIFUSION_MODEL_WEIGHTS` and `scripts.download_model --weights ...`.
- CheXpert baseline weights shipped via TorchXRayVision (`chexpert_weights.zip`): requires a separate archive; not integrated.

## Preprocessing (exact)

1. Decode; RGB/RGBA/P images are converted to RGB and channels averaged (the library's documented `img.mean(2)`); 16-bit images keep max value 65535.
2. `xrv.datasets.normalize(img, max_value)` → range [-1024, 1024].
3. Add channel axis, `XRayCenterCrop()`, `XRayResizer(224)`.
4. Forward pass. With operating thresholds present the library applies sigmoid and `op_norm`, so **0.5 marks the published operating point** per label. `above_operating_point` is reported only for labels that have a threshold.

## Inference status

Verified in the build environment: weights downloaded through `scripts.download_model`, model loaded, and inference run on one NIH ChestX-ray14 sample image taken from the TorchXRayVision test folder, through the agent, the API (`POST /analyze`) and the Streamlit UI. This was a smoke test of the pipeline; no accuracy claim follows from it.

## Interpreting outputs

- Scores are not calibrated probabilities. Do not read 0.62 as "62% chance".
- Operating points come from the authors' data; they may not transfer.
- The model sees only pixels. Symptom text does not influence it, and the fusion step does not merge the two.

## Evaluation caveat: no held-out evaluation for the default weights

The `all` weights were trained on Open-i, NIH, CheXpert, MIMIC and PadChest, which covers every dataset in the registry. Our test splits are not the authors' splits, so metrics of these weights on any of these datasets are likely optimistic. `scripts.evaluate` prints `NOT A HELD-OUT EVALUATION` in that case. For IU X-Ray, use `densenet121-res224-nih` (NIH only) for a cleaner external test, and note that IU labels are weak MeSH headings.

## Evaluation workflow

```powershell
python -m scripts.predict_dataset --split test          # labels are not read here
python -m scripts.evaluate --input reports\evaluation\predictions_iu_xray_test.jsonl
```

- Case score = max over the case's images.
- Decision rule: published operating points when available, else `--threshold` (default 0.5).
- Label maps in `config/label_maps/<dataset>.yaml` map dataset labels to model labels. IU uses a closed-world assumption (unmentioned mapped headings count as negative); edit the map after checking real heading counts.
- Uncertain labels: `--uncertainty-policy exclude|positive|negative`.
- AUROC only when both classes are present; bootstrap 95% CIs (1,000 resamples) only with ≥5 cases per class.

## Optional training: linear probe

`scripts/train_model.py` fits one logistic regression per label on frozen 1024-d DenseNet features. Train split for fitting, val split for choosing C, test split untouched until `--predict-test`. Outputs `models/linear_probe/<dataset>/probe.joblib` and `config.json` (preprocessing, label map, chosen C, val AUROCs). It was unit-tested on synthetic arrays only; it has not been run on a real dataset.

## Adding a model

1. Write an adapter with `info() -> ModelInfo` and `predict(gray, max_value)` returning scores, preprocessing steps, score type and limitations (see `app/agents/model_adapters.py`).
2. Register it in `build_adapter()` under a new `model_backend` name; return `UnavailableAdapter` with a reason when files are missing. Never download implicitly.
3. Record provenance and training datasets so evaluation can warn about contamination.
4. Add a label map per dataset and tests for the unavailable path.

"""Image model adapters.

TorchXRayVisionAdapter wraps the pretrained DenseNet-121 models from
TorchXRayVision (Cohen et al., Apache-2.0, https://github.com/mlmed/torchxrayvision).
Weights are never downloaded implicitly: they must already be in
settings.model_path (use `python -m scripts.download_model`).
"""
from __future__ import annotations

import importlib.util
import logging
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.schemas.analysis import LabelScore, ModelInfo, ModelStatus

log = logging.getLogger(__name__)

# Published training data per weights name (TorchXRayVision model_urls descriptions),
# mapped to this project's dataset ids. "google" labels and the RSNA/Kaggle pneumonia
# challenge both use NIH ChestX-ray14 images.
TXRV_TRAINING_DATA = {
    "densenet121-res224-all": ["nih_cxr14", "padchest", "chexpert", "mimic_cxr_jpg", "iu_xray"],
    "densenet121-res224-nih": ["nih_cxr14"],
    "densenet121-res224-pc": ["padchest"],
    "densenet121-res224-chex": ["chexpert"],
    "densenet121-res224-rsna": ["nih_cxr14"],
    "densenet121-res224-mimic_nb": ["mimic_cxr_jpg"],
    "densenet121-res224-mimic_ch": ["mimic_cxr_jpg"],
}
TXRV_SCORE_TYPE = ("sigmoid output re-scaled with the published operating thresholds (op_norm): 0.5 marks the "
                   "operating point; labels without a threshold keep the plain sigmoid output")
TXRV_LIMITATIONS = [
    "Scores are not calibrated probabilities and must not be read as the chance of disease.",
    "Operating points were chosen by the model authors on their own data; they may not transfer to other scanners or populations.",
    "Model outputs per label are not mutually exclusive and are not a diagnosis.",
    "The model only sees the image; it does not use the symptom text.",
]


class UnavailableAdapter:
    def __init__(self, backend: str, name: str, reason: str):
        self._info = ModelInfo(backend=backend, name=name, status=ModelStatus.unavailable, reason=reason)

    def info(self) -> ModelInfo:
        return self._info

    def predict(self, gray, max_value):
        raise RuntimeError(self._info.reason)


@dataclass
class TxrvPrediction:
    scores: list[LabelScore]
    preprocessing: list[str] = field(default_factory=list)
    score_type: str = TXRV_SCORE_TYPE
    score_limitations: list[str] = field(default_factory=lambda: list(TXRV_LIMITATIONS))


def txrv_weights_filename(weights: str) -> str | None:
    """File name TorchXRayVision expects for `weights`, or None if the package is missing."""
    import torchxrayvision as xrv

    entry = xrv.models.model_urls.get(weights)
    return Path(entry["weights_url"]).name if entry else None


class TorchXRayVisionAdapter:
    def __init__(self, weights: str, model_dir: Path):
        import torch
        import torchxrayvision as xrv

        self._torch, self._xrv = torch, xrv
        self.weights = weights
        entry = xrv.models.model_urls[weights]
        with warnings.catch_warnings():
            # the published weights are a pickled module; torch warns that layer source changed since 2020
            warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
            self.model = xrv.models.DenseNet(weights=weights, cache_dir=str(model_dir))
        self.model.eval()
        self.labels = [l for l in self.model.pathologies]
        ops = getattr(self.model, "op_threshs", None)
        self.has_op = ops is not None
        self.op_defined = ([not bool(np.isnan(v)) for v in ops.tolist()] if self.has_op else [False] * len(self.labels))
        self.resolution = entry.get("input_resolution", 224)
        self._lock = threading.Lock()
        self._info = ModelInfo(
            backend="torchxrayvision", name=f"TorchXRayVision DenseNet-121 ({weights})",
            version=f"torchxrayvision {xrv.__version__}, torch {torch.__version__}",
            status=ModelStatus.loaded, labels=[l for l in self.labels if l],
            provenance={"weights_url": entry["weights_url"], "description": entry.get("description", ""),
                        "license": "Apache-2.0 (TorchXRayVision)",
                        "paper": "Cohen et al., TorchXRayVision, arXiv:2111.00595"},
            training_datasets=TXRV_TRAINING_DATA.get(weights, []),
        )

    def info(self) -> ModelInfo:
        return self._info

    def predict(self, gray: np.ndarray, max_value: float) -> TxrvPrediction:
        xrv, torch = self._xrv, self._torch
        img = xrv.datasets.normalize(gray, max_value)          # -> [-1024, 1024]
        img = img[None, ...]                                   # (1, H, W)
        img = xrv.datasets.XRayCenterCrop()(img)
        img = xrv.datasets.XRayResizer(self.resolution)(img)
        with self._lock, torch.no_grad():
            out = self.model(torch.from_numpy(img).float()[None, ...])[0].numpy()
        scores = []
        for label, value, has_op in zip(self.labels, out.tolist(), self.op_defined):
            if not label:
                continue  # unused output slot in TorchXRayVision's label layout
            scores.append(LabelScore(label=label, score=round(float(value), 4),
                                     above_operating_point=(value > 0.5) if has_op else None))
        steps = [f"xrv.datasets.normalize(img, {max_value:.0f}) to the [-1024, 1024] range",
                 "XRayCenterCrop (square centre crop)", f"XRayResizer({self.resolution})",
                 "DenseNet-121 forward pass on CPU" + (", sigmoid + op_norm" if self.has_op else "")]
        return TxrvPrediction(scores=scores, preprocessing=steps)


def build_adapter(settings: Settings):
    name = f"{settings.model_backend}:{settings.model_weights}"
    if settings.model_backend != "torchxrayvision":
        return UnavailableAdapter(settings.model_backend, name, f"Unknown model backend {settings.model_backend!r}.")
    missing = [m for m in ("torch", "torchxrayvision") if importlib.util.find_spec(m) is None]
    if missing:
        return UnavailableAdapter("torchxrayvision", name,
                                  f"Optional packages not installed: {', '.join(missing)} "
                                  "(pip install -r requirements-model.txt)")
    try:
        filename = txrv_weights_filename(settings.model_weights)
    except Exception as exc:  # broken torch install shows up here
        return UnavailableAdapter("torchxrayvision", name, f"torchxrayvision could not be imported: {exc}")
    if filename is None:
        return UnavailableAdapter("torchxrayvision", name, f"Unknown weights name {settings.model_weights!r}.")
    weights_file = settings.model_path_abs / filename
    if not weights_file.is_file():
        return UnavailableAdapter(
            "torchxrayvision", name,
            f"Weights file {filename} not found in {settings.model_path}. "
            "Download it with: python -m scripts.download_model")
    try:
        adapter = TorchXRayVisionAdapter(settings.model_weights, settings.model_path_abs)
    except Exception as exc:
        log.exception("Model load failed")
        return UnavailableAdapter("torchxrayvision", name, f"Model failed to load: {exc.__class__.__name__}: {exc}")
    log.info("Loaded %s", adapter.info().name)
    return adapter

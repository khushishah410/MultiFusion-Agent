"""Chest X-ray upload validation and model inference.

The agent never fabricates findings. When the configured model cannot be
loaded it returns validation results plus an explicit `unavailable` status.
"""
from __future__ import annotations

import io
import logging
import time
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.schemas.analysis import ImageAnalysis, ImageValidation, LabelScore, ModelInfo, ModelStatus

log = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"PNG", "JPEG"}
Image.MAX_IMAGE_PIXELS = 100_000_000  # decompression-bomb guard (~10k x 10k)


class Prediction(Protocol):
    scores: list[LabelScore]
    preprocessing: list[str]
    score_type: str
    score_limitations: list[str]


class ImageModelAdapter(Protocol):
    def info(self) -> ModelInfo: ...

    def predict(self, gray: np.ndarray, max_value: float) -> Prediction: ...


def validate_image(data: bytes, max_bytes: int, min_side: int, max_side: int) -> tuple[ImageValidation, Image.Image | None]:
    v = ImageValidation(valid=False, size_bytes=len(data))
    if not data:
        v.errors.append("Empty file.")
        return v, None
    if len(data) > max_bytes:
        v.errors.append(f"File is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB.")
        return v, None
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        v.errors.append(f"Not a readable image: {exc.__class__.__name__}: {exc}")
        return v, None
    v.detected_format, v.width, v.height, v.mode = img.format, img.width, img.height, img.mode
    if img.format not in SUPPORTED_FORMATS:
        v.errors.append(f"Format {img.format} is not supported; use one of {sorted(SUPPORTED_FORMATS)}.")
    if min(img.size) < min_side:
        v.errors.append(f"Image is {img.width}x{img.height}; the shorter side must be at least {min_side}px.")
    if max(img.size) > max_side:
        v.errors.append(f"Image is {img.width}x{img.height}; the longer side must be at most {max_side}px.")
    if img.mode not in ("L", "I;16", "I", "RGB", "RGBA", "LA", "P"):
        v.errors.append(f"Unsupported pixel mode {img.mode}.")
    if img.mode in ("RGB", "RGBA"):
        arr = np.asarray(img.convert("RGB"), dtype=np.int16)
        if np.abs(arr[..., 0] - arr[..., 1]).mean() > 8 or np.abs(arr[..., 1] - arr[..., 2]).mean() > 8:
            v.warnings.append("Image has colour content; radiographs are normally grayscale. "
                              "Check that this is a chest X-ray.")
    v.valid = not v.errors
    return v, (img if v.valid else None)


def to_grayscale_array(img: Image.Image) -> tuple[np.ndarray, float, str]:
    """Return (H x W float32 array, max representable value, description)."""
    if img.mode in ("I;16", "I"):
        arr = np.asarray(img, dtype=np.float32)
        max_value = 65535.0 if img.mode == "I;16" or arr.max() > 255 else 255.0
        return arr, max_value, f"{img.mode} read as single channel, max value {max_value:.0f}"
    if img.mode in ("RGB", "RGBA", "P", "LA"):
        arr = np.asarray(img.convert("RGB"), dtype=np.float32).mean(axis=2)
        return arr, 255.0, f"{img.mode} converted to RGB and channels averaged to one grayscale channel"
    return np.asarray(img, dtype=np.float32), 255.0, "8-bit grayscale"


class ImageAgent:
    def __init__(self, adapter: ImageModelAdapter, max_bytes: int, min_side: int, max_side: int):
        self.adapter = adapter
        self.max_bytes = max_bytes
        self.min_side = min_side
        self.max_side = max_side

    def analyze(self, data: bytes) -> ImageAnalysis:
        t0 = time.perf_counter()
        validation, img = validate_image(data, self.max_bytes, self.min_side, self.max_side)
        info = self.adapter.info()
        result = ImageAnalysis(validation=validation, model=info, latency_ms=0.0)
        if img is None:
            result.error = "Image failed validation; no model inference was run."
        elif info.status != ModelStatus.loaded:
            result.error = f"Image model unavailable: {info.reason}. No findings were produced."
        else:
            gray, max_value, description = to_grayscale_array(img)
            try:
                pred = self.adapter.predict(gray, max_value)
                result.preprocessing = [description] + pred.preprocessing
                result.scores = pred.scores
                result.score_type = pred.score_type
                result.score_limitations = pred.score_limitations
            except Exception as exc:  # report inference failure instead of crashing the request
                log.exception("Image inference failed")
                result.model = info.model_copy(update={"status": ModelStatus.error, "reason": str(exc)})
                result.error = f"Inference failed: {exc.__class__.__name__}: {exc}"
        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

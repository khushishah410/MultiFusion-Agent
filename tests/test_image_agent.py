import importlib.util

import pytest

from app.agents.image_agent import ImageAgent, validate_image
from app.agents.model_adapters import UnavailableAdapter, build_adapter
from app.schemas.analysis import ModelStatus
from tests.conftest import FakeAdapter, png_bytes

MB = 1024 * 1024


def agent(adapter=None):
    return ImageAgent(adapter or UnavailableAdapter("x", "x", "weights missing"), 5 * MB, 64, 4000)


def test_valid_png_and_jpeg():
    for fmt in ("PNG", "JPEG"):
        v, img = validate_image(png_bytes(fmt=fmt), 5 * MB, 64, 4000)
        assert v.valid and img is not None and v.detected_format == fmt


def test_format_checked_by_content_not_name():
    v, _ = validate_image(png_bytes(fmt="GIF"), 5 * MB, 64, 4000)
    assert not v.valid and v.detected_format == "GIF" and "not supported" in v.errors[0]


def test_corrupted_and_empty_files():
    data = png_bytes()
    v, _ = validate_image(data[: len(data) // 3], 5 * MB, 64, 4000)
    assert not v.valid and "Not a readable image" in v.errors[0]
    v, _ = validate_image(b"not an image at all", 5 * MB, 64, 4000)
    assert not v.valid
    v, _ = validate_image(b"", 5 * MB, 64, 4000)
    assert v.errors == ["Empty file."]


def test_size_and_dimension_limits():
    v, _ = validate_image(png_bytes((32, 300)), 5 * MB, 64, 4000)
    assert not v.valid and "shorter side" in v.errors[0]
    v, _ = validate_image(png_bytes(), 100, 64, 4000)
    assert not v.valid and "limit" in v.errors[0]


def test_colour_image_warns_and_16bit_accepted():
    v, _ = validate_image(png_bytes(mode="RGB"), 5 * MB, 64, 4000)
    assert v.valid and v.warnings
    v, _ = validate_image(png_bytes(mode="I;16"), 5 * MB, 64, 4000)
    assert v.valid


def test_unavailable_model_reports_no_findings():
    r = agent().analyze(png_bytes())
    assert r.validation.valid
    assert r.model.status == ModelStatus.unavailable and r.scores == []
    assert "No findings were produced" in r.error


def test_missing_weights_gives_unavailable_adapter(settings):
    a = build_adapter(settings)  # settings.model_path points at an empty temp folder
    info = a.info()
    assert info.status == ModelStatus.unavailable
    expected = "not found" if importlib.util.find_spec("torchxrayvision") else "not installed"
    assert expected in info.reason


def test_unknown_backend(settings):
    s = settings.model_copy(update={"model_backend": "nope"})
    assert build_adapter(s).info().status == ModelStatus.unavailable


def test_invalid_image_never_reaches_model():
    calls = []

    class Spy(FakeAdapter):
        def predict(self, gray, max_value):
            calls.append(1)
            return super().predict(gray, max_value)

    r = agent(Spy()).analyze(b"garbage")
    assert calls == [] and r.scores == [] and "failed validation" in r.error


def test_loaded_adapter_scores_passed_through():
    r = agent(FakeAdapter()).analyze(png_bytes())
    assert {s.label for s in r.scores} == {"Cardiomegaly", "Effusion", "Nodule"}
    assert r.latency_ms >= 0 and r.error is None


@pytest.mark.skipif(importlib.util.find_spec("torchxrayvision") is None, reason="optional torchxrayvision not installed")
def test_real_txrv_model_if_weights_present():
    from app.core.config import get_settings

    s = get_settings()
    a = build_adapter(s)
    if a.info().status != ModelStatus.loaded:
        pytest.skip(f"weights not downloaded: {a.info().reason}")
    r = ImageAgent(a, 10 * MB, 64, 10000).analyze(png_bytes((300, 300)))
    assert len(r.scores) == 18 and all(0 <= x.score <= 1 for x in r.scores)
    assert r.score_type and r.score_limitations

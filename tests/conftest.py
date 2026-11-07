"""Shared fixtures. Everything is built in temporary folders; no network, GPU or real data."""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.config import REPO_ROOT, Settings
from app.schemas.analysis import LabelScore, ModelInfo, ModelStatus


def png_bytes(size=(256, 256), mode="L", fmt="PNG", value=None) -> bytes:
    rng = np.random.default_rng(0)
    if mode == "RGB":
        arr = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    elif mode == "I;16":
        arr = rng.integers(0, 65535, (size[1], size[0]), dtype=np.uint16)
    else:
        arr = np.full((size[1], size[0]), value, np.uint8) if value is not None else \
            rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


IU_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <eCitation>
      <pmcId id="{n}"/>
      <uId id="CXR{n}"/>
      <MedlineCitation><Article><Abstract>
        <AbstractText Label="COMPARISON">None.</AbstractText>
        <AbstractText Label="INDICATION">{indication}</AbstractText>
        <AbstractText Label="FINDINGS">{findings}</AbstractText>
        <AbstractText Label="IMPRESSION">{impression}</AbstractText>
      </Abstract></Article></MedlineCitation>
      <MeSH>{mesh}</MeSH>
      {images}
    </eCitation>
""")


def write_iu_fixture(root: Path, n_cases: int = 6, missing_image_case: int | None = 3) -> Path:
    """Tiny fake IU X-Ray folder (structure only; text is fixture text, images are noise)."""
    reports = root / "ecgen-radiology"
    reports.mkdir(parents=True)
    for n in range(1, n_cases + 1):
        image_ids = [f"CXR{n}_1_IM-000{n}-1001", f"CXR{n}_1_IM-000{n}-2001"]
        mesh = "<major>Cardiomegaly/mild</major><minor>x</minor>" if n % 2 else "<major>normal</major>"
        (reports / f"{n}.xml").write_text(IU_XML.format(
            n=n, indication=f"Fixture indication {n} cough", findings="Fixture findings text.",
            impression="Fixture impression.", mesh=mesh,
            images="\n".join(f'<parentImage id="{i}"><url>/x/{i}.jpg</url></parentImage>' for i in image_ids)),
            encoding="utf-8")
        for k, i in enumerate(image_ids):
            if n == missing_image_case and k == 1:
                continue
            (root / f"{i}.png").write_bytes(png_bytes((128, 128)))
    return root


class FakeAdapter:
    """Test double standing in for a loaded model. Its scores are fixed constants, not model output."""

    def __init__(self, scores=None):
        self.scores = scores or {"Cardiomegaly": 0.8, "Effusion": 0.2, "Nodule": 0.55}

    def info(self):
        return ModelInfo(backend="fake", name="FakeAdapter (test double)", version="0", status=ModelStatus.loaded,
                         labels=list(self.scores), training_datasets=["iu_xray"])

    def predict(self, gray, max_value):
        from app.agents.model_adapters import TxrvPrediction
        return TxrvPrediction(scores=[LabelScore(label=k, score=v, above_operating_point=v > 0.5)
                                      for k, v in self.scores.items()], preprocessing=["fake"])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_root=tmp_path / "raw", manifest_dir=tmp_path / "manifests", manifest_path="",
        evaluation_dir=tmp_path / "eval", retrieval_index=tmp_path / "index.json",
        knowledge_dir=tmp_path / "knowledge", model_path=tmp_path / "no_model",
        datasets_config=REPO_ROOT / "config" / "datasets.yaml",
        symptom_lexicon=REPO_ROOT / "config" / "symptom_lexicon.yaml", large_download_mb=200,
    )


@pytest.fixture
def lexicon_path() -> Path:
    return REPO_ROOT / "config" / "symptom_lexicon.yaml"


@pytest.fixture
def iu_prepared(settings):
    """IU-style fixture dataset, prepared into a manifest with the real builder."""
    from app.services.dataset_ops import DatasetOps

    write_iu_fixture(settings.data_root_abs / "iu_xray")
    ops = DatasetOps(settings)
    meta = ops.prepare("iu_xray")
    ops.validate("iu_xray")
    return settings, ops, meta


@pytest.fixture
def api_client(iu_prepared, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api import dependencies, main
    from app.api.main import app

    settings, ops, _ = iu_prepared
    ops.activate("iu_xray")
    state = dependencies.AppState(settings)
    monkeypatch.setattr(main, "get_state", lambda: state)
    app.dependency_overrides[dependencies.get_state] = lambda: state
    with TestClient(app) as client:
        client.state_obj = state
        yield client
    app.dependency_overrides.clear()


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path

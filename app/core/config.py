"""Typed application settings.

Precedence (highest first): environment variables (MULTIFUSION_*), the .env
file, config/settings.yaml, the defaults below.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MULTIFUSION_",
        env_file=REPO_ROOT / ".env",
        yaml_file=REPO_ROOT / "config" / "settings.yaml",
        extra="ignore",
    )

    data_root: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    manifest_path: str = ""
    evaluation_dir: Path = Path("reports/evaluation")
    retrieval_index: Path = Path("indexes/lexical_index.json")
    knowledge_dir: Path = Path("data/knowledge")
    datasets_config: Path = Path("config/datasets.yaml")
    symptom_lexicon: Path = Path("config/symptom_lexicon.yaml")
    diagnosis_rules: Path = Path("config/diagnosis_rules.yaml")

    model_backend: str = "torchxrayvision"
    model_weights: str = "densenet121-res224-all"
    model_path: Path = Path("models/torchxrayvision")
    max_upload_mb: float = 10
    min_image_side: int = 64
    max_image_side: int = 10000

    large_download_mb: float = 200
    api_url: str = "http://127.0.0.1:8000"
    log_level: str = "INFO"
    retain_uploads: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    def resolve(self, path: Path | str) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (REPO_ROOT / path).resolve()

    @property
    def data_root_abs(self) -> Path:
        return self.resolve(self.data_root)

    @property
    def manifest_dir_abs(self) -> Path:
        return self.resolve(self.manifest_dir)

    @property
    def evaluation_dir_abs(self) -> Path:
        return self.resolve(self.evaluation_dir)

    @property
    def model_path_abs(self) -> Path:
        return self.resolve(self.model_path)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def active_manifest(self) -> tuple[Path | None, str]:
        """Return (manifest path, how it was chosen).

        Order: explicit MULTIFUSION_MANIFEST_PATH, the pointer written by
        `dataset_manager activate`, the synthetic demo manifest.
        """
        if self.manifest_path:
            return self.resolve(self.manifest_path), "configured"
        pointer = self.manifest_dir_abs / "active.json"
        if pointer.exists():
            target = json.loads(pointer.read_text(encoding="utf-8")).get("manifest")
            if target:
                return self.resolve(target), "activated"
        demo = self.manifest_dir_abs / "synthetic_demo.jsonl"
        if demo.exists():
            return demo, "synthetic_demo"
        return None, "none"


@lru_cache
def get_settings() -> Settings:
    return Settings()

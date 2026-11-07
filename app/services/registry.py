from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DatasetFile:
    name: str
    url: str
    documented_size_mb: float | None = None
    metadata_only: bool = False
    sha256: str | None = None
    extract: bool = False
    mirrors: list[str] = field(default_factory=list)
    size_bytes: int | None = None
    torrent: str | None = None
    torrent_infohash: str | None = None

    @property
    def urls(self) -> list[str]:
        return [self.url] + [m for m in self.mirrors if m != self.url]


@dataclass
class DatasetSpec:
    id: str
    display_name: str
    official_source: str
    license: str
    modalities: list[str]
    automated_download_allowed: bool
    strategy: str
    manifest_builder: str
    files: list[DatasetFile] = field(default_factory=list)
    instructions: str | None = None
    citation: str | None = None
    documented_counts: str | None = None
    label_source: str | None = None
    patient_grouping: str | None = None
    expected_files: list[str] = field(default_factory=list)
    size_provenance: str | None = None
    raw: dict = field(default_factory=dict)

    @property
    def documented_total_mb(self) -> float | None:
        sizes = [f.documented_size_mb for f in self.files]
        return sum(sizes) if sizes and all(s is not None for s in sizes) else None

    def public_dict(self) -> dict:
        return {
            "id": self.id, "display_name": self.display_name, "official_source": self.official_source,
            "license": self.license, "citation": self.citation, "modalities": self.modalities,
            "automated_download_allowed": self.automated_download_allowed, "strategy": self.strategy,
            "instructions": self.instructions, "documented_counts": self.documented_counts,
            "label_source": self.label_source, "patient_grouping": self.patient_grouping,
            "expected_files": self.expected_files, "size_provenance": self.size_provenance,
            "documented_total_mb": self.documented_total_mb,
            "files": [f.__dict__ for f in self.files],
        }


def load_registry(path: Path) -> dict[str, DatasetSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = {}
    for ds_id, d in data["datasets"].items():
        dl = d.get("download", {})
        specs[ds_id] = DatasetSpec(
            id=ds_id,
            display_name=d["display_name"],
            official_source=d["official_source"],
            license=" ".join(str(d["license"]).split()),
            modalities=d.get("modalities", []),
            automated_download_allowed=bool(d.get("automated_download_allowed", False)),
            strategy=dl.get("strategy", "manual"),
            manifest_builder=d["manifest_builder"],
            files=[DatasetFile(**f) for f in dl.get("files", [])],
            instructions=" ".join(str(dl["instructions"]).split()) if dl.get("instructions") else None,
            citation=" ".join(str(d["citation"]).split()) if d.get("citation") else None,
            documented_counts=d.get("documented_counts"),
            label_source=d.get("label_source"),
            patient_grouping=d.get("patient_grouping"),
            expected_files=d.get("expected_files", []),
            size_provenance=dl.get("size_provenance"),
            raw=d,
        )
        if specs[ds_id].strategy == "http" and not specs[ds_id].automated_download_allowed:
            raise ValueError(f"{ds_id}: http strategy requires automated_download_allowed: true")
    return specs

"""Normalized manifest record and dataset-level schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ManifestRecord(BaseModel):
    case_id: str = Field(min_length=1)
    patient_id: str | None = Field(None, description="Sensitive. Used only for grouping splits; never returned by the API.")
    clinical_indication: str | None = None
    symptoms: list[str] = []
    radiology_findings: str | None = None
    radiology_impression: str | None = None
    image_ids: list[str] = []
    image_paths: list[str] = Field(default=[], description="Relative to the dataset root")
    labels: dict[str, int] = Field(default={}, description="1 = positive, 0 = negative, as stated by the source")
    label_uncertainty: list[str] = Field(default=[], description="Labels the source marks as uncertain (e.g. CheXpert -1)")
    label_source: str | None = None
    mesh_major: list[str] = []
    dataset_source: str
    split: str | None = None
    synthetic: bool = False
    metadata: dict[str, Any] = {}


class PublicCase(BaseModel):
    """Manifest record as exposed by the API: no patient_id, no absolute paths."""
    case_id: str
    clinical_indication: str | None
    symptoms: list[str]
    radiology_findings: str | None
    radiology_impression: str | None
    image_ids: list[str]
    image_paths: list[str]
    image_exists: list[bool]
    labels: dict[str, int]
    label_uncertainty: list[str]
    label_source: str | None
    mesh_major: list[str]
    dataset_source: str
    split: str | None
    synthetic: bool
    metadata: dict[str, Any]


class ValidationIssue(BaseModel):
    kind: str
    case_id: str | None = None
    detail: str


class ValidationReport(BaseModel):
    dataset_id: str
    manifest: str
    generated_at: str
    total_records: int
    valid_records: int
    duplicate_case_ids: list[str] = []
    malformed_records: int = 0
    records_with_missing_images: int = 0
    missing_image_files: int = 0
    unreadable_images: int = 0
    images_checked: int = 0
    image_size_stats: dict[str, Any] = {}
    field_coverage: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    leakage: dict[str, Any] = {}
    issues: list[ValidationIssue] = Field(default=[], description="Capped sample of issues")
    ok: bool

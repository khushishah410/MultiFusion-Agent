"""Schemas for symptom, image, retrieval and fusion outputs."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- symptoms
class SymptomMention(BaseModel):
    text: str = Field(description="Exact span from the user's text")
    start: int
    end: int
    canonical: str = Field(description="Lexicon concept the span matched")
    match_type: Literal["exact", "synonym", "fuzzy"]
    match_similarity: float | None = Field(
        None, description="String similarity for fuzzy matches (0-1). Not a clinical confidence."
    )
    negated: bool = False
    negation_cue: str | None = None
    uncertain: bool = False
    uncertainty_cue: str | None = None
    red_flag: bool = False


class DurationMention(BaseModel):
    text: str
    start: int
    end: int


class ContextMention(BaseModel):
    text: str
    start: int
    end: int
    kind: Literal["severity", "onset", "pattern", "trigger", "history", "user_context"]


class UnrecognizedTerm(BaseModel):
    text: str
    start: int
    end: int
    in_lexicon: bool = False
    in_dataset_text: bool | None = Field(
        None,
        description="True/False if the active dataset's report text was searched; "
        "None when no dataset was available to check.",
    )


class SymptomAnalysis(BaseModel):
    original_text: str
    normalized_text: str
    symptom_mentions: list[SymptomMention] = []
    negated_mentions: list[SymptomMention] = []
    duration_mentions: list[DurationMention] = []
    context_mentions: list[ContextMention] = []
    unrecognized_terms: list[UnrecognizedTerm] = []
    contradictions: list[str] = []
    method: str
    lexicon_version: str
    confidence: None = Field(None, description="Rule-based extraction has no defensible confidence score.")
    safety_notice: str | None = None
    limitations: list[str] = []


# ---------------------------------------------------------------- image
class ModelStatus(str, Enum):
    loaded = "loaded"
    unavailable = "unavailable"
    error = "error"


class ImageValidation(BaseModel):
    valid: bool
    detected_format: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    size_bytes: int
    errors: list[str] = []
    warnings: list[str] = []


class LabelScore(BaseModel):
    label: str
    score: float
    above_operating_point: bool | None = None


class ModelInfo(BaseModel):
    backend: str
    name: str
    version: str | None = None
    status: ModelStatus
    reason: str | None = None
    labels: list[str] = []
    provenance: dict[str, str] = {}
    training_datasets: list[str] = []


class ImageAnalysis(BaseModel):
    validation: ImageValidation
    model: ModelInfo
    preprocessing: list[str] = []
    scores: list[LabelScore] = []
    score_type: str | None = None
    score_limitations: list[str] = []
    latency_ms: float
    error: str | None = None


# ---------------------------------------------------------------- retrieval
class RetrievedPassage(BaseModel):
    doc_id: str
    title: str
    text: str
    source: str
    source_url: str | None = None
    source_date: str | None = None
    kind: Literal["reference_case", "document"]
    synthetic: bool = False
    similarity: float = Field(description="BM25 lexical similarity. Not a clinical probability.")


class RetrievalResults(BaseModel):
    status: Literal["active", "not_configured", "error"]
    query: str = ""
    method: str = "bm25_lexical"
    passages: list[RetrievedPassage] = []
    index_info: dict = {}
    note: str | None = None


# ---------------------------------------------------------------- diagnosis
class DiagnosisCandidate(BaseModel):
    key: str
    condition: str
    points: float
    max_points: float
    match_percent: float = Field(description="points / max available points. Not a probability.")
    strength: Literal["strong", "moderate", "weak"]
    image_support: bool
    symptom_support: bool
    supporting: list[str]
    against: list[str]
    confirm_tests: list[str]
    source_name: str
    source_url: str


class DiagnosisResult(BaseModel):
    status: Literal["suggested", "insufficient_evidence"]
    primary: DiagnosisCandidate | None = None
    differentials: list[DiagnosisCandidate] = []
    headline: str
    missing_information: list[str] = []
    method: str
    scoring_note: str
    disclaimer: str


# ---------------------------------------------------------------- fusion
class AnalysisMode(str, Enum):
    demo = "demo"
    text_only = "text_only"
    image_validation_only = "image_validation_only"
    image_model = "image_model"
    multimodal = "multimodal"


class EvidenceItem(BaseModel):
    id: str
    component: Literal["symptom_agent", "image_agent", "retrieval_agent", "dataset_case"]
    kind: Literal["user_reported", "model_output", "retrieved_reference", "dataset_record"]
    content: str
    score: float | None = None
    score_type: str | None = None
    derived_from: list[str] = []
    synthetic: bool = False


class FusionItem(BaseModel):
    statement: str
    category: Literal["diagnosis", "observation", "model_output", "reference", "disagreement", "missing", "safety"]
    supported_by: list[str] = Field(default=[], description="EvidenceItem ids")


class FusionResult(BaseModel):
    case_id: str
    mode: AnalysisMode
    symptom_analysis: SymptomAnalysis | None = None
    image_analysis: ImageAnalysis | None = None
    retrieval_results: RetrievalResults | None = None
    reference_case: dict | None = None
    diagnosis: DiagnosisResult | None = None
    fusion_summary: list[FusionItem] = []
    evidence: list[EvidenceItem] = []
    uncertainties: list[str] = []
    missing_components: list[str] = []
    warnings: list[str] = []
    processing_metadata: dict = {}
    model_versions: dict[str, str] = {}
    dataset_provenance: dict | None = None
    disclaimer: str


class SymptomRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    context: str | None = Field(None, max_length=2000)
    duration: str | None = Field(None, max_length=200)

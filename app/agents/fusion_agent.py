"""Deterministic, provenance-preserving combination of component outputs.

The suggested diagnosis (from DiagnosisAgent) is placed first. The rest of the
summary lists what each component said, which evidence supports each statement,
where components disagree, and what is missing.
"""
from __future__ import annotations

import time
import uuid

from app.schemas.analysis import (
    AnalysisMode,
    DiagnosisResult,
    EvidenceItem,
    FusionItem,
    FusionResult,
    ImageAnalysis,
    ModelStatus,
    RetrievalResults,
    SymptomAnalysis,
)

DISCLAIMER = (
    "Educational research prototype, not a medical device and not clinically validated. The suggested "
    "diagnosis comes from transparent rules over the inputs and must be confirmed by a qualified doctor."
)
TOP_SCORES = 5


class FusionAgent:
    def __init__(self, label_maps: dict[str, dict[str, str]] | None = None):
        # dataset_source -> {dataset label: model label}
        self.label_maps = label_maps or {}

    def fuse(
        self,
        symptoms: SymptomAnalysis | None,
        image: ImageAnalysis | None,
        retrieval: RetrievalResults | None,
        reference_case: dict | None = None,
        dataset_provenance: dict | None = None,
        component_latency_ms: dict[str, float] | None = None,
        case_id: str | None = None,
        image_is_reference_image: bool = False,
        diagnosis: DiagnosisResult | None = None,
    ) -> FusionResult:
        t0 = time.perf_counter()
        evidence: list[EvidenceItem] = []
        summary: list[FusionItem] = []
        uncertainties: list[str] = []
        warnings: list[str] = []
        missing: list[str] = []

        def add(component, kind, content, prefix, **kw) -> str:
            eid = f"{prefix}{sum(1 for e in evidence if e.id.startswith(prefix)) + 1}"
            evidence.append(EvidenceItem(id=eid, component=component, kind=kind, content=content, **kw))
            return eid

        # ---- suggested diagnosis (first item of the summary)
        if diagnosis is not None:
            if diagnosis.primary:
                d = diagnosis.primary
                alts = ", ".join(c.condition for c in diagnosis.differentials[:3])
                summary.append(FusionItem(
                    statement=f"Suggested diagnosis: {d.condition} ({d.strength} evidence, {d.points:g} points, "
                              f"{d.match_percent:g}% of available). " + (f"Also consider: {alts}. " if alts else "")
                              + "Confirm with: " + "; ".join(d.confirm_tests) + ".",
                    category="diagnosis"))
            else:
                summary.append(FusionItem(statement=diagnosis.headline, category="diagnosis"))

        # ---- symptoms
        has_text = symptoms is not None and bool(symptoms.normalized_text)
        if has_text:
            self._symptom_items(symptoms, add, summary, uncertainties)
        else:
            missing.append("symptom_text")
            summary.append(FusionItem(statement="No symptom text was provided.", category="missing"))

        # ---- image
        image_scored = False
        if image is None:
            missing.append("image")
            summary.append(FusionItem(statement="No image was uploaded.", category="missing"))
        else:
            image_scored = self._image_items(image, add, summary, uncertainties, missing)

        # ---- reference case vs image model
        if reference_case:
            # Labels are only comparable with model output when the analysed image
            # is the reference case's own image, never for a new upload.
            compare = image if (image_scored and image_is_reference_image) else None
            self._reference_items(reference_case, compare, add, summary, uncertainties, evidence)

        # ---- retrieval
        if retrieval is None or retrieval.status != "active":
            missing.append("retrieval")
            note = retrieval.note if retrieval else "Retrieval was not requested."
            summary.append(FusionItem(statement=f"Retrieval unavailable: {note}", category="missing"))
        elif retrieval.passages:
            derived = [e.id for e in evidence if e.component == "symptom_agent"]
            ids = [add("retrieval_agent", "retrieved_reference",
                       f"{p.title} [{p.source}{', synthetic' if p.synthetic else ''}]: {p.text[:200]}",
                       "R", score=p.similarity, score_type="bm25_similarity", derived_from=derived,
                       synthetic=p.synthetic) for p in retrieval.passages]
            summary.append(FusionItem(
                statement=f"{len(ids)} indexed passage(s) share wording with the symptom text. They are "
                          "reference material, not evidence that this person has the same condition. "
                          "Their query was built from the symptom text, so they are not independent of it.",
                category="reference", supported_by=ids))
        else:
            summary.append(FusionItem(statement="Retrieval returned no passages for this query.", category="missing"))

        # ---- safety
        if has_text and symptoms.safety_notice:
            ids = [e.id for e in evidence if e.component == "symptom_agent" and "red flag" in e.content]
            summary.append(FusionItem(statement=symptoms.safety_notice, category="safety", supported_by=ids))

        synthetic_used = any(e.synthetic for e in evidence) or bool(reference_case and reference_case.get("synthetic"))
        if synthetic_used:
            warnings.append("Synthetic demo data contributed to this output. Synthetic records contain no clinical information.")

        if synthetic_used:
            mode = AnalysisMode.demo
        elif image_scored:
            mode = AnalysisMode.multimodal if has_text else AnalysisMode.image_model
        elif has_text:
            mode = AnalysisMode.text_only
        else:
            mode = AnalysisMode.image_validation_only

        model_versions = {}
        if symptoms is not None:
            model_versions["symptom_agent"] = f"{symptoms.method} (lexicon {symptoms.lexicon_version})"
        if image is not None:
            model_versions["image_model"] = (
                f"{image.model.name} {image.model.version or ''} [{image.model.status.value}]".strip())
        latency = dict(component_latency_ms or {})
        latency["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return FusionResult(
            case_id=case_id or f"new-{uuid.uuid4().hex[:12]}",
            mode=mode,
            symptom_analysis=symptoms,
            image_analysis=image,
            retrieval_results=retrieval,
            reference_case=reference_case,
            diagnosis=diagnosis,
            fusion_summary=summary,
            evidence=evidence,
            uncertainties=uncertainties,
            missing_components=missing,
            warnings=warnings,
            processing_metadata={"latency_ms": latency, "synthetic_data_used": synthetic_used},
            model_versions=model_versions,
            dataset_provenance=dataset_provenance,
            disclaimer=DISCLAIMER,
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _symptom_items(s: SymptomAnalysis, add, summary, uncertainties):
        if s.symptom_mentions:
            ids, parts = [], []
            for m in s.symptom_mentions:
                note = []
                if m.match_type == "fuzzy":
                    note.append(f"spelling match {m.match_similarity}")
                if m.uncertain:
                    note.append(f"uncertain: '{m.uncertainty_cue}'")
                if m.red_flag:
                    note.append("red flag")
                label = m.canonical.replace("_", " ")
                content = f"'{m.text}' -> {label}" + (f" ({'; '.join(note)})" if note else "")
                ids.append(add("symptom_agent", "user_reported", content, "S"))
                parts.append(label)
            summary.append(FusionItem(statement="Reported in the text: " + ", ".join(dict.fromkeys(parts)) + ".",
                                      category="observation", supported_by=ids))
        if s.negated_mentions:
            ids = [add("symptom_agent", "user_reported", f"'{m.text}' denied (cue '{m.negation_cue}')", "N")
                   for m in s.negated_mentions]
            summary.append(FusionItem(
                statement="Stated as absent: " + ", ".join(dict.fromkeys(m.canonical.replace("_", " ")
                                                                         for m in s.negated_mentions)) + ".",
                category="observation", supported_by=ids))
        if s.duration_mentions or s.context_mentions:
            ids = [add("symptom_agent", "user_reported", f"duration: '{d.text}'", "D") for d in s.duration_mentions]
            ids += [add("symptom_agent", "user_reported", f"{c.kind}: '{c.text}'", "C") for c in s.context_mentions]
            texts = [d.text for d in s.duration_mentions] + [c.text for c in s.context_mentions]
            summary.append(FusionItem(statement="Timing and context given: " + "; ".join(texts) + ".",
                                      category="observation", supported_by=ids))
        if s.unrecognized_terms:
            ids = [add("symptom_agent", "user_reported", f"unrecognized: '{u.text}'", "U") for u in s.unrecognized_terms]
            summary.append(FusionItem(
                statement="Not in the symptom lexicon, kept as written and not interpreted: "
                          + "; ".join(f"'{u.text}'" for u in s.unrecognized_terms) + ".",
                category="observation", supported_by=ids))
            uncertainties.append("Unrecognized terms may describe relevant symptoms that this rule-based "
                                 "extractor cannot interpret.")
        for c in s.contradictions:
            summary.append(FusionItem(statement=c, category="disagreement"))
        if any(m.uncertain for m in s.symptom_mentions):
            uncertainties.append("Some symptoms were described with hedging words (e.g. 'maybe').")
        if any(m.match_type == "fuzzy" for m in s.symptom_mentions + s.negated_mentions):
            uncertainties.append("Some mentions were matched by spelling similarity and may be misreadings.")

    @staticmethod
    def _image_items(image: ImageAnalysis, add, summary, uncertainties, missing) -> bool:
        v = image.validation
        if not v.valid:
            summary.append(FusionItem(statement="Uploaded image failed validation: " + "; ".join(v.errors),
                                      category="missing"))
            missing.append("valid_image")
            return False
        if image.model.status != ModelStatus.loaded or not image.scores:
            missing.append("image_model_output")
            summary.append(FusionItem(
                statement=f"Image passed validation ({v.detected_format}, {v.width}x{v.height}) but no model "
                          f"output is available: {image.error or image.model.reason}. No image findings are reported.",
                category="missing"))
            return False
        top = sorted(image.scores, key=lambda s: s.score, reverse=True)[:TOP_SCORES]
        ids = [add("image_agent", "model_output",
                   f"{s.label}: {s.score:.3f}" + (" (above operating point)" if s.above_operating_point else ""),
                   "I", score=s.score, score_type=image.score_type) for s in image.scores]
        flagged = [s.label for s in image.scores if s.above_operating_point]
        statement = (f"{image.model.name} output, highest raw scores: "
                     + ", ".join(f"{s.label} {s.score:.2f}" for s in top) + ". ")
        statement += (f"Above the model's published operating points: {', '.join(flagged)}."
                      if flagged else "No label is above the model's published operating point.")
        summary.append(FusionItem(statement=statement, category="model_output", supported_by=ids))
        uncertainties.extend(image.score_limitations)
        return True

    def _reference_items(self, case: dict, image: ImageAnalysis | None, add, summary, uncertainties, evidence):
        text = " | ".join(x for x in [case.get("radiology_findings"), case.get("radiology_impression")] if x)
        kid = add("dataset_case", "dataset_record",
                  f"{case['case_id']} ({case['dataset_source']}): labels {sorted(k for k, v in case.get('labels', {}).items() if v == 1)}; {text[:200]}",
                  "K", synthetic=bool(case.get("synthetic")))
        summary.append(FusionItem(
            statement=f"Dataset case {case['case_id']} ({case['dataset_source']}) is attached as a reference record. "
                      "Its labels describe that record only and are not a finding for any new patient.",
            category="reference", supported_by=[kid]))
        if image is None:
            return
        mapping = self.label_maps.get(case["dataset_source"], {})
        case_pos = {mapping.get(k, k) for k, v in case.get("labels", {}).items() if v == 1}
        case_known = {mapping.get(k, k) for k in case.get("labels", {})}
        model_labels = {s.label for s in image.scores}
        flagged = {s.label for s in image.scores if s.above_operating_point}
        comparable = model_labels & (case_known | set(mapping.values()))
        if not comparable:
            uncertainties.append("Reference-case labels do not map to the image model's labels, so they were not compared.")
            return
        only_model = sorted((flagged - case_pos) & comparable)
        only_case = sorted((case_pos - flagged) & comparable)
        if only_model or only_case:
            image_ids = {e.content.split(":")[0]: e.id for e in evidence if e.component == "image_agent"}
            ids = [kid] + [image_ids[label] for label in only_model + only_case if label in image_ids]
            parts = []
            if only_model:
                parts.append(f"model flags {', '.join(only_model)} but the case labels do not")
            if only_case:
                parts.append(f"case labels include {', '.join(only_case)} but the model does not flag it")
            summary.append(FusionItem(
                statement="Image model and reference-case labels disagree: " + "; ".join(parts)
                          + ". This compares the uploaded image with a different case and is informational only.",
                category="disagreement", supported_by=ids))
        if case.get("label_source"):
            uncertainties.append(f"Reference-case labels come from {case['label_source']} and may be noisy.")
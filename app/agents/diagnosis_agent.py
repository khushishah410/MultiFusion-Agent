"""Suggested diagnosis from symptom mentions and image-model flags.

Transparent point scoring over config/diagnosis_rules.yaml. Every point is traceable to a
reported symptom, a denied symptom, a context phrase or an image label above the model's
operating point. The output is an educational suggestion, not a clinical diagnosis.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from app.schemas.analysis import (
    DiagnosisCandidate,
    DiagnosisResult,
    ImageAnalysis,
    ModelStatus,
    SymptomAnalysis,
)

DISCLAIMER = ("AI-suggested diagnosis from a rule-based educational prototype. It is not a clinical "
              "diagnosis; a qualified doctor must examine the patient and confirm it with the listed tests.")
SCORING_NOTE = ("Points: reported symptom = rule weight, hedged symptom = half, denied symptom = minus the "
                "weight, image label above the model's operating point = rule weight, matching context phrase "
                "= rule weight. Match % = points / maximum points available from the inputs given. Not a probability.")


class DiagnosisAgent:
    method = "rule_based_points_v1"

    def __init__(self, rules_path: Path):
        data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
        self.version = str(data.get("version", "unknown"))
        self.min_points = float(data.get("min_points_for_suggestion", 2.0))
        self.conditions = data["conditions"]

    def suggest(self, symptoms: SymptomAnalysis | None, image: ImageAnalysis | None) -> DiagnosisResult:
        has_text = symptoms is not None and bool(symptoms.normalized_text)
        image_used = (image is not None and image.model.status == ModelStatus.loaded and bool(image.scores))
        flagged = {s.label: s.score for s in image.scores if s.above_operating_point} if image_used else {}
        reported = {}
        hedged = set()
        denied = {}
        if has_text:
            for m in symptoms.symptom_mentions:
                reported.setdefault(m.canonical, m.text)
                if m.uncertain:
                    hedged.add(m.canonical)
            hedged -= {m.canonical for m in symptoms.symptom_mentions if not m.uncertain}
            for m in symptoms.negated_mentions:
                if m.canonical not in reported:
                    denied.setdefault(m.canonical, m.text)
        text = symptoms.original_text.lower() if has_text else ""

        candidates = []
        for key, rule in self.conditions.items():
            points, max_points, sup, img_sup, sym_sup, against = 0.0, 0.0, [], False, False, []
            if image_used:
                for label, w in rule.get("image", {}).items():
                    max_points += w
                    if label in flagged:
                        points += w
                        img_sup = True
                        sup.append(f"X-ray model flags {label} (score {flagged[label]:.2f}) +{w:g}")
            if has_text:
                for canon, w in rule.get("symptoms", {}).items():
                    max_points += w
                    name = canon.replace("_", " ")
                    if canon in reported:
                        gain = w / 2 if canon in hedged else w
                        points += gain
                        sym_sup = True
                        sup.append(f"Reported {name} ('{reported[canon]}')"
                                   + (" - hedged" if canon in hedged else "") + f" +{gain:g}")
                    elif canon in denied:
                        points -= w
                        against.append(f"Denies {name} -{w:g}")
                for phrase, w in rule.get("context", {}).items():
                    max_points += w
                    if re.search(rf"(?<![a-z-]){re.escape(phrase)}(?![a-z-])", text):
                        points += w
                        sup.append(f"Context '{phrase}' +{w:g}")
            if image_used and rule.get("image") and not img_sup:
                against.append("X-ray model does not flag the findings usually seen in this condition")
            if points <= 0 or not sup:
                continue
            strength = ("strong" if points >= 4 and img_sup and sym_sup else
                        "moderate" if points >= 2.5 else "weak")
            candidates.append(DiagnosisCandidate(
                key=key, condition=rule["name"], points=round(points, 2), max_points=round(max_points, 2),
                match_percent=round(100 * points / max_points, 1) if max_points else 0.0, strength=strength,
                image_support=img_sup, symptom_support=sym_sup, supporting=sup, against=against,
                confirm_tests=rule.get("confirm_tests", []), source_name=rule["source"]["name"],
                source_url=rule["source"]["url"]))
        candidates.sort(key=lambda c: (c.points, c.match_percent), reverse=True)

        missing = []
        if not has_text:
            missing.append("No symptom text, so only X-ray findings were used.")
        if not image_used:
            missing.append("No X-ray model output, so only symptoms were used; X-ray-based conditions cannot be assessed.")
        missing.append("Age, vital signs (temperature, heart rate, oxygen saturation) and examination findings "
                       "were not provided and would change the assessment.")

        primary = candidates[0] if candidates and candidates[0].points >= self.min_points else None
        return DiagnosisResult(
            status="suggested" if primary else "insufficient_evidence",
            primary=primary,
            differentials=candidates[1:5] if primary else candidates[:5],
            headline=(f"Most likely: {primary.condition} ({primary.strength} evidence)" if primary else
                      "No condition is supported strongly enough to suggest; see the differentials."),
            missing_information=missing,
            method=f"{self.method} (rules {self.version})",
            scoring_note=SCORING_NOTE,
            disclaimer=DISCLAIMER,
        )

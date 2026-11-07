"""Rule-based extraction of symptoms from arbitrary free text.

The agent does not look up the training dataset to decide what a symptom is.
It matches a configurable lexicon (exact, then fuzzy for spelling variants),
applies NegEx-style negation and uncertainty cues, extracts duration and
context phrases, and keeps every leftover content phrase as an
"unrecognized term" instead of discarding it.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from app.schemas.analysis import (
    ContextMention,
    DurationMention,
    SymptomAnalysis,
    SymptomMention,
    UnrecognizedTerm,
)

FUZZY_THRESHOLD = 0.85
FUZZY_MIN_CHARS = 5
CUE_WINDOW = 6

SAFETY_NOTICE = (
    "Your description includes {items}. These can accompany conditions that need prompt "
    "assessment by a clinician. If symptoms are severe, sudden, or getting worse, contact "
    "your local emergency number or go to the nearest emergency department. This tool "
    "cannot assess urgency or identify a cause."
)

LIMITATIONS = [
    "Rule-based keyword and pattern matching against a hand-written lexicon; not a clinical NLP model.",
    "Negation and uncertainty use simple cue words within the same clause and can be wrong "
    "(e.g. 'no improvement in my cough' is read as a negated cough).",
    "Extracted mentions describe what the text says. They are not findings, diagnoses, or risk estimates.",
    "Terms outside the lexicon are reported as unrecognized, not as absent or harmless.",
]

STOPWORDS = set(
    """a about above after again against all also am an and any are as at be because been before
    being below between both by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself of off on once only or other our ours ourselves out over own same she
    should so some such than that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves im i'm ive i've id i'd its it's dont don't cant can't
    feel feels feeling felt get gets getting got lot lots really since day days week weeks month months
    year years hour hours today yesterday tonight morning evening night last past ago now still again
    left right side little bit thing things like also well much many time times start started starting
    keep keeps kept going go goes went come comes came noticed notice having has seem seems patient
    reports report complains complaint complaining presents presenting experiencing experience
    symptom symptoms context duration one two three four five six seven eight nine ten first""".split()
)
BRIDGE_WORDS = {"of", "in", "on", "at", "my", "the", "around", "under", "behind", "near"}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
CLAUSE_BREAK_RE = re.compile(r"[.;!?\n]")
NUM = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|few|a few|couple of|a couple of|several)"
UNIT = r"(?:hours?|hrs?|days?|weeks?|wks?|months?|years?|yrs?)"
DURATION_PATTERNS = [
    re.compile(rf"\b(?:for|since|over|in|from)\s+(?:the\s+)?(?:last|past)?\s*{NUM}\s+{UNIT}\b"),
    re.compile(rf"\b{NUM}\s*-?\s*{UNIT}(?:\s+ago)?\b"),
    re.compile(r"\bsince\s+(?:yesterday|last\s+(?:night|week|month|year)|this\s+(?:morning|week)|"
               r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b"),
    re.compile(r"\b(?:yesterday|last night|this morning|today)\b"),
]


@dataclass
class _Token:
    text: str
    start: int
    end: int
    clause: int


class SymptomExtractor(Protocol):
    """Interface for alternative extractors (e.g. a medical NER model).

    An implementation must return a SymptomAnalysis and set `method` to say
    what produced the result.
    """

    def analyze(self, text: str, context: str | None = None, duration: str | None = None) -> SymptomAnalysis: ...


class DatasetTextIndex:
    """Lower-cased report text of the active dataset, for 'is this term in the dataset?' checks."""

    def __init__(self, texts: list[str]):
        self._blob = "\n".join(t.lower() for t in texts if t)

    def contains(self, phrase: str) -> bool:
        return re.search(rf"\b{re.escape(phrase.lower())}\b", self._blob) is not None

    @property
    def empty(self) -> bool:
        return not self._blob


def _lower_same_length(text: str) -> str:
    out = []
    for ch in text:
        if ch in "‘’ʼ":
            ch = "'"
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
    return "".join(out)


class RuleBasedSymptomAgent:
    method = "rule_based_lexicon_v1"

    def __init__(self, lexicon_path: Path, dataset_index: DatasetTextIndex | None = None):
        lex = yaml.safe_load(Path(lexicon_path).read_text(encoding="utf-8"))
        self.lexicon_version = str(lex.get("version", "unknown"))
        self.dataset_index = dataset_index
        self.red_flags: set[str] = set()
        self._terms: dict[tuple[str, ...], tuple[str, str]] = {}  # tokens -> (canonical, match_type)
        for canonical, spec in lex["symptoms"].items():
            if spec.get("red_flag"):
                self.red_flags.add(canonical)
            canonical_surface = canonical.replace("_", " ")
            for term in spec["terms"]:
                key = tuple(TOKEN_RE.findall(_lower_same_length(term)))
                if key and key not in self._terms:
                    kind = "exact" if " ".join(key) == canonical_surface else "synonym"
                    self._terms[key] = (canonical, kind)
        self._max_len = max(len(k) for k in self._terms)
        self._neg_pre = self._phrases(lex["negation_pre"])
        self._neg_post = self._phrases(lex["negation_post"])
        self._terminators = self._phrases(lex["negation_terminators"])
        self._uncertain = self._phrases(lex["uncertainty"])
        self._context = {kind: self._phrases(items) for kind, items in lex["context"].items()}

    @staticmethod
    def _phrases(items: list[str]) -> list[tuple[str, ...]]:
        phrases = {tuple(TOKEN_RE.findall(_lower_same_length(i))) for i in items}
        return sorted((p for p in phrases if p), key=len, reverse=True)

    # ------------------------------------------------------------ public
    def analyze(self, text: str, context: str | None = None, duration: str | None = None) -> SymptomAnalysis:
        parts = [text or ""]
        if context:
            parts.append(f"Context: {context}")
        if duration:
            parts.append(f"Duration: {duration}")
        original = "\n".join(parts)
        work = _lower_same_length(original)
        normalized = " ".join(unicodedata.normalize("NFKC", original).lower().split())

        if not normalized.strip():
            return SymptomAnalysis(
                original_text=original, normalized_text="", method=self.method,
                lexicon_version=self.lexicon_version,
                limitations=["No text was provided, so nothing was extracted. "
                             "An empty result is not evidence that symptoms are absent."] + LIMITATIONS,
            )

        tokens = self._tokenize(work)
        used = [False] * len(tokens)

        mentions = self._match_symptoms(original, tokens, used)
        for mention, span in mentions:
            self._apply_cues(mention, tokens, span)
        all_mentions = [m for m, _ in mentions]

        durations = self._match_durations(original, work, tokens, used)
        contexts = self._match_context(original, tokens, used)
        self._mark_cue_tokens(tokens, used)
        unrecognized = self._unrecognized(original, tokens, used)

        affirmed = [m for m in all_mentions if not m.negated]
        negated = [m for m in all_mentions if m.negated]
        contradictions = sorted(
            {m.canonical for m in affirmed} & {m.canonical for m in negated}
        )
        contradiction_msgs = [
            f"'{c.replace('_', ' ')}' is both reported and denied in the text; both mentions are kept."
            for c in contradictions
        ]
        red = sorted({m.canonical for m in affirmed if m.red_flag})
        notice = None
        if red:
            notice = SAFETY_NOTICE.format(items=", ".join(r.replace("_", " ") for r in red))

        return SymptomAnalysis(
            original_text=original,
            normalized_text=normalized,
            symptom_mentions=affirmed,
            negated_mentions=negated,
            duration_mentions=durations,
            context_mentions=contexts,
            unrecognized_terms=unrecognized,
            contradictions=contradiction_msgs,
            method=self.method,
            lexicon_version=self.lexicon_version,
            safety_notice=notice,
            limitations=LIMITATIONS,
        )

    # ------------------------------------------------------------ steps
    @staticmethod
    def _tokenize(work: str) -> list[_Token]:
        breaks = [m.start() for m in CLAUSE_BREAK_RE.finditer(work)]
        tokens = []
        for m in TOKEN_RE.finditer(work):
            clause = sum(1 for b in breaks if b < m.start())
            tokens.append(_Token(m.group(), m.start(), m.end(), clause))
        return tokens

    def _match_symptoms(self, original, tokens, used):
        found = []
        i = 0
        while i < len(tokens):
            hit = None
            for n in range(min(self._max_len, len(tokens) - i), 0, -1):
                window = tokens[i:i + n]
                if len({t.clause for t in window}) > 1:
                    continue
                key = tuple(t.text for t in window)
                if key in self._terms:
                    canonical, kind = self._terms[key]
                    hit = (n, canonical, kind, None)
                    break
            if hit is None:
                hit = self._fuzzy(tokens, i)
            if hit:
                n, canonical, kind, sim = hit
                start, end = tokens[i].start, tokens[i + n - 1].end
                found.append((SymptomMention(
                    text=original[start:end], start=start, end=end, canonical=canonical,
                    match_type=kind, match_similarity=sim, red_flag=canonical in self.red_flags,
                ), (i, i + n)))
                for k in range(i, i + n):
                    used[k] = True
                i += n
            else:
                i += 1
        return found

    def _fuzzy(self, tokens, i):
        for n in range(min(3, len(tokens) - i), 0, -1):
            window = tokens[i:i + n]
            if len({t.clause for t in window}) > 1:
                continue
            if n == 1 and window[0].text in STOPWORDS:
                continue
            candidate = " ".join(t.text for t in window)
            if len(candidate) < FUZZY_MIN_CHARS:
                continue
            best, best_key = 0.0, None
            for key in self._terms:
                if len(key) != n:
                    continue
                ratio = difflib.SequenceMatcher(None, candidate, " ".join(key)).ratio()
                if ratio > best:
                    best, best_key = ratio, key
            if best_key and best >= FUZZY_THRESHOLD:
                canonical, _ = self._terms[best_key]
                return n, canonical, "fuzzy", round(best, 3)
        return None

    def _find_phrase(self, tokens, lo, hi, phrases, clause):
        """Return (phrase_text, index) of the cue closest to the mention in tokens[lo:hi]."""
        best = None
        for start in range(lo, hi):
            for phrase in phrases:
                end = start + len(phrase)
                if end > hi:
                    continue
                if all(tokens[start + k].text == phrase[k] and tokens[start + k].clause == clause
                       for k in range(len(phrase))):
                    best = (" ".join(phrase), start, end)
                    break
        return best

    def _apply_cues(self, mention: SymptomMention, tokens, span):
        s, e = span
        clause = tokens[s].clause
        lo = max(0, s - CUE_WINDOW)
        # scope stops at a terminator ("but", "however") between cue and mention
        term = None
        for k in range(s - 1, lo - 1, -1):
            if tokens[k].clause != clause:
                lo = k + 1
                break
            if any(tuple(t.text for t in tokens[k:k + len(p)]) == p for p in self._terminators):
                term = k
                break
        if term is not None:
            lo = term + 1
        unc = self._find_phrase(tokens, lo, s, self._uncertain, clause)
        # a hedge between the cue and the mention ends negation scope:
        # "no chest pain, maybe some cough" -> cough is hedged, not denied
        neg_lo = unc[2] if unc else lo
        neg = self._find_phrase(tokens, neg_lo, s, self._neg_pre, clause)
        if neg is None:
            neg = self._find_phrase(tokens, e, min(len(tokens), e + 3), self._neg_post, clause)
        if neg:
            mention.negated, mention.negation_cue = True, neg[0]
        if unc:
            mention.uncertain, mention.uncertainty_cue = True, unc[0]

    @staticmethod
    def _match_durations(original, work, tokens, used):
        spans = []
        for pattern in DURATION_PATTERNS:
            for m in pattern.finditer(work):
                spans.append((m.start(), m.end()))
        spans.sort()
        merged: list[list[int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        for idx, t in enumerate(tokens):
            if any(s <= t.start and t.end <= e for s, e in merged):
                used[idx] = True
        return [DurationMention(text=original[s:e], start=s, end=e) for s, e in merged]

    def _match_context(self, original, tokens, used):
        out = []
        i = 0
        while i < len(tokens):
            matched = False
            for kind, phrases in self._context.items():
                for phrase in phrases:
                    n = len(phrase)
                    window = tokens[i:i + n]
                    if len(window) == n and tuple(t.text for t in window) == phrase and not any(used[i:i + n]):
                        s, e = window[0].start, window[-1].end
                        out.append(ContextMention(text=original[s:e], start=s, end=e, kind=kind))
                        for k in range(i, i + n):
                            used[k] = True
                        i += n
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                i += 1
        return out

    def _mark_cue_tokens(self, tokens, used):
        cue_phrases = self._neg_pre + self._neg_post + self._uncertain + self._terminators
        for i in range(len(tokens)):
            for phrase in cue_phrases:
                n = len(phrase)
                if tuple(t.text for t in tokens[i:i + n]) == phrase:
                    for k in range(i, i + n):
                        used[k] = True

    def _unrecognized(self, original, tokens, used):
        def content(idx):
            t = tokens[idx]
            return (not used[idx] and t.text not in STOPWORDS and not t.text.isdigit()
                    and len(t.text) > 2 and t.text not in ("context", "duration"))

        groups: list[list[int]] = []
        i = 0
        while i < len(tokens):
            if not content(i):
                i += 1
                continue
            group = [i]
            j = i + 1
            while j < len(tokens) and tokens[j].clause == tokens[i].clause:
                if content(j):
                    group.append(j)
                    j += 1
                    continue
                # bridge "pain in my left arm"
                k = j
                while k < len(tokens) and k - j < 2 and not used[k] and tokens[k].text in BRIDGE_WORDS:
                    k += 1
                if k > j and k < len(tokens) and tokens[k].clause == tokens[i].clause and content(k):
                    group.extend(range(j, k + 1))
                    j = k + 1
                    continue
                break
            groups.append(group)
            i = j
        out = []
        for g in groups:
            s, e = tokens[g[0]].start, tokens[g[-1]].end
            phrase = original[s:e]
            in_dataset = None
            if self.dataset_index is not None and not self.dataset_index.empty:
                in_dataset = self.dataset_index.contains(phrase)
            out.append(UnrecognizedTerm(text=phrase, start=s, end=e, in_lexicon=False, in_dataset_text=in_dataset))
        return out

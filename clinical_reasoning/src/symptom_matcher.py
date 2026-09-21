"""Deterministic, rule-based matching of free-text patient symptoms against
knowledge base symptom/warning-sign text.

This is intentionally simple keyword/stem overlap, NOT NLP or an LLM. It is
fully traceable (every match can be explained by which stem overlapped) and
produces the same output every time for the same input, which is required
for a clinical decision-support prototype. It is a heuristic, not a
medically validated symptom-matching algorithm.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "a", "an", "the", "of", "or", "and", "in", "on", "to", "is", "are",
    "with", "for", "at", "by", "as", "may", "can", "be", "it", "this",
    "that", "than", "over", "into", "not", "has", "have", "will",
}

_STEM_LENGTH = 4
_MIN_TOKEN_LENGTH = 3

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _stems(text: str) -> set[str]:
    """Return a set of crude word stems (lowercased prefixes) for a piece of text.

    This is truncation stemming (e.g. "bleeding" and "bleeds" both stem to
    "blee"), a cheap heuristic that tolerates simple plural/tense variation
    without any external NLP dependency.
    """
    words = _WORD_RE.findall(text.lower())
    stems = set()
    for word in words:
        if len(word) < _MIN_TOKEN_LENGTH or word in _STOPWORDS:
            continue
        stems.add(word[:_STEM_LENGTH])
    return stems


def match_patient_symptoms(
    patient_symptoms: list[str],
    kb_symptoms: list[str],
    kb_warning_signs: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Classify each patient-reported symptom against a disease's KB text.

    A symptom that overlaps the disease's warning_signs text is classified
    as a warning sign (warning signs take priority, since they are the more
    clinically significant category). Otherwise, if it overlaps the general
    symptoms text, it is classified as a matched symptom. Symptoms matching
    neither are returned as unmatched, for transparency rather than being
    silently dropped.

    Returns (matched_symptoms, matched_warning_signs, unmatched_symptoms),
    each a list preserving the caller's original wording and order.
    """
    warning_stems = set()
    for sign in kb_warning_signs:
        warning_stems |= _stems(sign)

    symptom_stems = set()
    for symptom in kb_symptoms:
        symptom_stems |= _stems(symptom)

    matched_symptoms: list[str] = []
    matched_warning_signs: list[str] = []
    unmatched_symptoms: list[str] = []

    for reported in patient_symptoms:
        reported_stems = _stems(reported)
        if reported_stems & warning_stems:
            matched_warning_signs.append(reported)
        elif reported_stems & symptom_stems:
            matched_symptoms.append(reported)
        else:
            unmatched_symptoms.append(reported)

    return matched_symptoms, matched_warning_signs, unmatched_symptoms

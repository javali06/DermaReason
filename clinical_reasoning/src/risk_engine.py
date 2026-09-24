"""Transparent, rule-based risk assessment.

This module answers "how concerning is this case", which is NOT the same
question as "how confident is the model in its prediction" (that is a
separate input, handled explicitly and never silently conflated with risk).

Every contribution to a risk score is recorded as a human-readable entry in
risk_factors, so a risk_level can always be explained by pointing at the
specific rules that produced it. There are no hidden rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import RiskConfig
from .knowledge_base import KnowledgeBase
from .symptom_matcher import match_patient_symptoms


class RiskEngineError(Exception):
    """Raised for invalid inputs to the risk engine (e.g. out-of-range confidence)."""


@dataclass
class RiskAssessment:
    disease_code: str
    confidence: float
    confidence_band: str
    risk_level: str
    risk_score: int
    risk_factors: list[str]
    matched_symptoms: list[str]
    matched_warning_signs: list[str]
    unmatched_symptoms: list[str] = field(default_factory=list)


class RiskEngine:
    """Combines KnowledgeBase facts, model confidence, and patient symptoms
    into a deterministic, explainable risk assessment.
    """

    def __init__(self, knowledge_base: KnowledgeBase, config: RiskConfig | None = None):
        self._kb = knowledge_base
        self._config = config or RiskConfig()

    def assess(self, disease_code: str, confidence: float, symptoms: list[str]) -> RiskAssessment:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise RiskEngineError(f"confidence must be a number between 0 and 1, got {confidence!r}")
        if not (0.0 <= confidence <= 1.0):
            raise RiskEngineError(f"confidence must be between 0 and 1, got {confidence}")

        entry = self._kb.get_disease(disease_code)  # raises UnknownDiseaseError if not found
        symptoms = symptoms or []

        matched_symptoms, matched_warning_signs, unmatched_symptoms = match_patient_symptoms(
            symptoms, entry["symptoms"], entry["warning_signs"]
        )

        risk_factors: list[str] = []
        score = 0

        concern_level = entry["general_concern_level"]
        concern_weight = self._config.concern_level_weight(concern_level)
        score += concern_weight
        if concern_weight > 0:
            risk_factors.append(
                f"Predicted class '{entry['display_name']}' has a baseline clinical "
                f"concern level of '{concern_level}' (contributes {concern_weight} to the score)."
            )
        else:
            risk_factors.append(
                f"Predicted class '{entry['display_name']}' has a baseline clinical "
                f"concern level of '{concern_level}' (contributes 0 to the score)."
            )

        capped_warning_count = min(len(matched_warning_signs), self._config.max_warning_sign_score)
        warning_score = capped_warning_count * self._config.warning_sign_weight
        score += warning_score
        for sign in matched_warning_signs:
            risk_factors.append(
                f"Reported symptom '{sign}' matches a documented warning sign for "
                f"{entry['display_name']}."
            )
        if len(matched_warning_signs) > self._config.max_warning_sign_score:
            risk_factors.append(
                f"{len(matched_warning_signs)} warning signs were reported; "
                f"score contribution capped at {self._config.max_warning_sign_score} "
                f"per configured max_warning_sign_score."
            )

        confidence_band = self._config.confidence_band(confidence)
        if confidence_band == "low":
            score += self._config.low_confidence_caution_weight
            risk_factors.append(
                f"Model confidence is low ({confidence:.2f}); the underlying prediction "
                f"is less reliable, so this is added as a separate caution factor "
                f"(this reflects prediction reliability, not clinical severity)."
            )
        else:
            risk_factors.append(
                f"Model confidence band is '{confidence_band}' ({confidence:.2f}); "
                f"no caution factor applied at this band per current configuration."
            )

        risk_level = self._config.score_to_risk_level(score)

        return RiskAssessment(
            disease_code=entry["disease_code"],
            confidence=confidence,
            confidence_band=confidence_band,
            risk_level=risk_level,
            risk_score=score,
            risk_factors=risk_factors,
            matched_symptoms=matched_symptoms,
            matched_warning_signs=matched_warning_signs,
            unmatched_symptoms=unmatched_symptoms,
        )

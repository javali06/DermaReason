"""Maps a risk level to a decision-support recommendation category.

These categories (MONITOR / CONSULT_DERMATOLOGIST / URGENT_REFERRAL) are
decision-support signals, not treatment prescriptions. This module never
produces medication or treatment plans - only a category and general next
step, both drawn from configuration rather than generated freely.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RecommendationConfig


class RecommendationEngineError(Exception):
    """Raised when a recommendation cannot be produced for the given input."""


@dataclass
class Recommendation:
    risk_level: str
    category: str
    text: str
    basis: str


class RecommendationEngine:
    """Turns a RiskEngine's risk_level into a fixed-category recommendation."""

    def __init__(self, config: RecommendationConfig | None = None):
        self._config = config or RecommendationConfig()

    def recommend(self, risk_level: str) -> Recommendation:
        mapping = self._config.risk_level_to_recommendation
        if risk_level not in mapping:
            raise RecommendationEngineError(
                f"Unknown risk_level '{risk_level}'. Known risk levels: {sorted(mapping)}"
            )

        category = mapping[risk_level]
        text_map = self._config.recommendation_text
        if category not in text_map:
            raise RecommendationEngineError(
                f"No recommendation text configured for category '{category}'"
            )

        basis = (
            f"Risk level '{risk_level}' maps to recommendation category '{category}' "
            f"per this prototype's configured mapping (config/recommendation_config.json). "
            f"This mapping is an engineering rule for the DermaReason prototype, "
            f"not a clinically validated triage protocol."
        )

        return Recommendation(
            risk_level=risk_level,
            category=category,
            text=text_map[category],
            basis=basis,
        )

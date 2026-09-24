"""Configurable, non-clinically-validated parameters for the risk engine.

Every threshold and weight here is a prototype engineering choice for the
DermaReason project, not a clinically validated value. They live in a JSON
file (config/risk_config.json) instead of being hardcoded so they can be
tuned, reviewed, or replaced without touching the risk engine's logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_config.json"
DEFAULT_RECOMMENDATION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "recommendation_config.json"
)

REQUIRED_TOP_LEVEL_KEYS = (
    "confidence_thresholds",
    "concern_level_weights",
    "warning_sign_weight",
    "max_warning_sign_score",
    "low_confidence_caution_weight",
    "risk_score_bands",
)

VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "URGENT"}
VALID_RECOMMENDATION_CATEGORIES = {"MONITOR", "CONSULT_DERMATOLOGIST", "URGENT_REFERRAL"}


class RiskConfigError(Exception):
    """Raised when the risk configuration file is missing or malformed."""


class RiskConfig:
    """Loads and exposes prototype risk-scoring configuration.

    IMPORTANT: values loaded here (confidence thresholds, weights, score
    bands) are engineering parameters for this prototype. Nothing about
    them should be presented to a user as clinically validated.
    """

    def __init__(self, config_path: str | Path | None = None):
        self._path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
        self._data = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RiskConfigError(f"Risk config file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RiskConfigError(f"Risk config file is not valid JSON ({path}): {exc}") from exc

        missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in data]
        if missing:
            raise RiskConfigError(f"Risk config file {path} is missing required keys: {missing}")

        thresholds = data["confidence_thresholds"]
        for level in ("low", "medium", "high"):
            if level not in thresholds:
                raise RiskConfigError(f"confidence_thresholds is missing '{level}'")

        if not (0 <= thresholds["low"] <= thresholds["medium"] <= thresholds["high"] <= 1):
            raise RiskConfigError(
                "confidence_thresholds must satisfy 0 <= low <= medium <= high <= 1, "
                f"got {thresholds}"
            )

        return data

    @property
    def confidence_thresholds(self) -> dict[str, float]:
        return dict(self._data["confidence_thresholds"])

    @property
    def concern_level_weights(self) -> dict[str, int]:
        return dict(self._data["concern_level_weights"])

    @property
    def warning_sign_weight(self) -> int:
        return self._data["warning_sign_weight"]

    @property
    def max_warning_sign_score(self) -> int:
        return self._data["max_warning_sign_score"]

    @property
    def low_confidence_caution_weight(self) -> int:
        return self._data["low_confidence_caution_weight"]

    @property
    def risk_score_bands(self) -> dict[str, list[int | None]]:
        return dict(self._data["risk_score_bands"])

    def confidence_band(self, confidence: float) -> str:
        """Classify a model confidence value into a band using configured thresholds.

        This describes how confident the MODEL is in its own prediction. It is
        deliberately kept separate from clinical risk (see risk_engine.py).
        """
        t = self.confidence_thresholds
        if confidence < t["low"]:
            return "low"
        if confidence < t["medium"]:
            return "medium"
        if confidence < t["high"]:
            return "high"
        return "very_high"

    def concern_level_weight(self, concern_level: str) -> int:
        weights = self.concern_level_weights
        if concern_level not in weights:
            raise RiskConfigError(
                f"Unknown general_concern_level '{concern_level}'. "
                f"Configured levels: {sorted(weights)}"
            )
        return weights[concern_level]

    def score_to_risk_level(self, score: int) -> str:
        """Map a raw rule-based score to a risk level using configured bands.

        Bands are defined as [min, max] pairs; max may be null/None to mean
        'and above'. Raises RiskConfigError if no band covers the score,
        which should only happen if the config itself is malformed.
        """
        for level, (low, high) in self.risk_score_bands.items():
            if score >= low and (high is None or score <= high):
                return level
        raise RiskConfigError(
            f"No configured risk_score_bands entry covers score={score}. "
            f"Bands: {self.risk_score_bands}"
        )


class RecommendationConfigError(Exception):
    """Raised when the recommendation configuration file is missing or malformed."""


class RecommendationConfig:
    """Loads and exposes the prototype risk-level -> recommendation-category mapping.

    IMPORTANT: this mapping is an engineering simplification for the
    DermaReason prototype, not a clinically validated triage protocol.
    """

    def __init__(self, config_path: str | Path | None = None):
        self._path = Path(config_path) if config_path is not None else DEFAULT_RECOMMENDATION_CONFIG_PATH
        self._data = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise RecommendationConfigError(f"Recommendation config file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RecommendationConfigError(
                f"Recommendation config file is not valid JSON ({path}): {exc}"
            ) from exc

        for key in ("risk_level_to_recommendation", "recommendation_text"):
            if key not in data:
                raise RecommendationConfigError(f"Recommendation config file {path} is missing key '{key}'")

        mapping = data["risk_level_to_recommendation"]
        missing_levels = VALID_RISK_LEVELS - set(mapping)
        if missing_levels:
            raise RecommendationConfigError(
                f"risk_level_to_recommendation is missing entries for: {sorted(missing_levels)}"
            )

        invalid_categories = set(mapping.values()) - VALID_RECOMMENDATION_CATEGORIES
        if invalid_categories:
            raise RecommendationConfigError(
                f"risk_level_to_recommendation contains unknown categories: {sorted(invalid_categories)}. "
                f"Valid categories: {sorted(VALID_RECOMMENDATION_CATEGORIES)}"
            )

        text_map = data["recommendation_text"]
        missing_text = set(mapping.values()) - set(text_map)
        if missing_text:
            raise RecommendationConfigError(
                f"recommendation_text is missing entries for categories: {sorted(missing_text)}"
            )

        return data

    @property
    def risk_level_to_recommendation(self) -> dict[str, str]:
        return dict(self._data["risk_level_to_recommendation"])

    @property
    def recommendation_text(self) -> dict[str, str]:
        return dict(self._data["recommendation_text"])

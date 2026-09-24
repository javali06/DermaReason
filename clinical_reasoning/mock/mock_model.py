"""Stand-in for the not-yet-built ML Diagnostic Agent.

This mock exposes the same interface a real diagnostic model would:

    prediction = model.predict(image_path)
    # -> {"disease": "MELANOMA", "confidence": 0.87}

Swapping this mock for the real model later should require changing only
which model object gets constructed and handed to ClinicalReasoningAgent -
nothing in clinical_reasoning/src/ needs to change, since the agent only
ever depends on this dict shape, never on how it was produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MOCK_PREDICTION_PATH = Path(__file__).resolve().parent / "mock_prediction.json"
DEFAULT_MOCK_SCENARIOS_PATH = Path(__file__).resolve().parent / "mock_scenarios.json"


class MockModelError(Exception):
    """Raised when mock prediction/scenario data is missing or malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MockModelError(f"Mock data file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MockModelError(f"Mock data file is not valid JSON ({path}): {exc}") from exc


class MockDiagnosticModel:
    """A mock ML Diagnostic Agent that always returns a fixed prediction
    loaded from mock_prediction.json.
    """

    def __init__(self, mock_prediction_path: str | Path | None = None):
        self._path = Path(mock_prediction_path) if mock_prediction_path else DEFAULT_MOCK_PREDICTION_PATH
        self._data = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        data = _load_json(path)
        if "disease" not in data or "confidence" not in data:
            raise MockModelError(
                f"Mock prediction file {path} must contain 'disease' and 'confidence' keys."
            )
        return data

    def predict(self, image_path: str | None = None) -> dict[str, Any]:
        """Return the mock prediction.

        image_path is accepted (to match the interface a real model would
        need) but ignored - this mock does not process images.
        """
        return {"disease": self._data["disease"], "confidence": self._data["confidence"]}


def load_mock_scenarios(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load a small set of named demo scenarios (disease + confidence + symptoms)
    for exercising the full pipeline end-to-end without a real model.

    This is a development/testing convenience, separate from the single
    mock_prediction.json the project spec asks for.
    """
    resolved_path = Path(path) if path else DEFAULT_MOCK_SCENARIOS_PATH
    data = _load_json(resolved_path)
    scenarios = {key: value for key, value in data.items() if not key.startswith("_")}
    for name, scenario in scenarios.items():
        if "disease" not in scenario or "confidence" not in scenario:
            raise MockModelError(
                f"Scenario '{name}' in {resolved_path} must contain 'disease' and 'confidence'."
            )
    return scenarios

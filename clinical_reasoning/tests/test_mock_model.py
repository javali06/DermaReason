import json

import pytest

from mock.mock_model import (
    MockDiagnosticModel,
    MockModelError,
    load_mock_scenarios,
)
from src.clinical_reasoning_agent import ClinicalReasoningAgent


def test_default_mock_prediction_loads():
    model = MockDiagnosticModel()
    prediction = model.predict()
    assert prediction == {"disease": "melanoma", "confidence": 0.87}


def test_predict_ignores_image_path_argument():
    model = MockDiagnosticModel()
    # A real model would use image_path; the mock accepts but ignores it.
    assert model.predict(image_path="some/path.jpg") == model.predict()


def test_missing_mock_file_raises(tmp_path):
    with pytest.raises(MockModelError):
        MockDiagnosticModel(tmp_path / "does_not_exist.json")


def test_malformed_mock_file_raises(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(MockModelError):
        MockDiagnosticModel(path)


def test_mock_file_missing_required_keys_raises(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"disease": "melanoma"}), encoding="utf-8")
    with pytest.raises(MockModelError):
        MockDiagnosticModel(path)


def test_load_mock_scenarios_returns_expected_names():
    scenarios = load_mock_scenarios()
    expected = {
        "melanoma_high_confidence_concerning",
        "melanoma_low_confidence",
        "bcc_with_symptoms",
        "benign_nv_no_symptoms",
    }
    assert expected.issubset(scenarios.keys())
    assert "_note" not in scenarios


# --- End-to-end: mock model output feeding the full reasoning pipeline ---


def test_full_pipeline_with_default_mock_prediction():
    model = MockDiagnosticModel()
    prediction = model.predict()
    agent = ClinicalReasoningAgent()
    result = agent.analyze(prediction["disease"], prediction["confidence"], [])
    assert result.disease_code == "MELANOMA"


@pytest.mark.parametrize(
    "scenario_name",
    [
        "melanoma_high_confidence_concerning",
        "melanoma_low_confidence",
        "bcc_with_symptoms",
        "benign_nv_no_symptoms",
    ],
)
def test_full_pipeline_with_each_named_scenario(scenario_name):
    scenarios = load_mock_scenarios()
    scenario = scenarios[scenario_name]
    agent = ClinicalReasoningAgent()
    result = agent.analyze_request(
        {
            "prediction": {"disease": scenario["disease"], "confidence": scenario["confidence"]},
            "symptoms": scenario.get("symptoms", []),
        }
    )
    assert result.risk_level in ("LOW", "MODERATE", "HIGH", "URGENT")
    assert result.recommendation_category in ("MONITOR", "CONSULT_DERMATOLOGIST", "URGENT_REFERRAL")

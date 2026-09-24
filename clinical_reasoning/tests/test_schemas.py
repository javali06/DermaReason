import json

import pytest
from pydantic import ValidationError

from src.clinical_reasoning_agent import ClinicalReasoningAgent
from src.report_generator import ReportGenerator
from src.schemas import (
    ClinicalReasoningInput,
    ClinicalReasoningOutput,
    PredictionInput,
    ReportOutput,
)


def test_prediction_input_valid():
    p = PredictionInput(disease="melanoma", confidence=0.87)
    assert p.disease == "melanoma"
    assert p.confidence == 0.87


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2])
def test_prediction_input_rejects_out_of_range_confidence(bad_confidence):
    with pytest.raises(ValidationError):
        PredictionInput(disease="melanoma", confidence=bad_confidence)


def test_prediction_input_rejects_empty_disease():
    with pytest.raises(ValidationError):
        PredictionInput(disease="", confidence=0.5)


def test_clinical_reasoning_input_defaults_symptoms_to_empty_list():
    request = ClinicalReasoningInput(prediction=PredictionInput(disease="nv", confidence=0.9))
    assert request.symptoms == []


def test_clinical_reasoning_input_parses_from_raw_dict():
    raw = {
        "prediction": {"disease": "melanoma", "confidence": 0.87},
        "symptoms": ["bleeding", "itching", "rapid growth"],
    }
    request = ClinicalReasoningInput.model_validate(raw)
    assert request.prediction.disease == "melanoma"
    assert request.symptoms == ["bleeding", "itching", "rapid growth"]


# --- Round-tripping the real dataclass output into the schema ---


def test_output_schema_builds_directly_from_agent_dataclass():
    agent = ClinicalReasoningAgent()
    result = agent.analyze("melanoma", 0.87, ["bleeding", "itching", "rapid growth"])

    schema = ClinicalReasoningOutput.model_validate(result)

    assert schema.disease_code == result.disease_code
    assert schema.risk_level == result.risk_level
    assert schema.sources[0].title == result.sources[0]["title"]
    assert schema.taxonomy_note is None


def test_output_schema_serializes_to_json():
    agent = ClinicalReasoningAgent()
    result = agent.analyze("akiec", 0.7, ["ulceration"])
    schema = ClinicalReasoningOutput.model_validate(result)

    payload = json.loads(schema.model_dump_json())
    assert payload["disease_code"] == "AKIEC"
    assert payload["taxonomy_note"] is not None


def test_report_output_schema_bundles_text_and_structured_result():
    agent = ClinicalReasoningAgent()
    result = agent.analyze("bcc", 0.75, ["bleeding"])
    report_text = ReportGenerator.generate(result)

    report_output = ReportOutput(
        report_text=report_text,
        result=ClinicalReasoningOutput.model_validate(result),
    )
    assert "DERMAREASON ANALYSIS REPORT" in report_output.report_text
    assert report_output.result.disease_code == "BCC"

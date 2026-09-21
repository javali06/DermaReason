import pytest

from src.clinical_reasoning_agent import ClinicalReasoningAgent, ClinicalReasoningAgentError
from src.knowledge_base import UnknownDiseaseError
from src.risk_engine import RiskEngineError


@pytest.fixture
def agent():
    return ClinicalReasoningAgent()


# Scenario 1: Melanoma + high confidence + concerning symptoms


def test_melanoma_high_confidence_concerning_symptoms(agent):
    result = agent.analyze("melanoma", 0.87, ["bleeding", "itching", "rapid growth"])
    assert result.disease_code == "MELANOMA"
    assert result.risk_level in ("HIGH", "URGENT")
    assert result.recommendation_category == "URGENT_REFERRAL"
    assert result.disclaimer  # medical disclaimer always present
    assert len(result.sources) > 0
    assert "Melanoma" in result.reasoning


# Scenario 2: Melanoma + low confidence


def test_melanoma_low_confidence(agent):
    result = agent.analyze("melanoma", 0.20, [])
    assert result.confidence_band == "low"
    assert any("low" in f.lower() for f in result.risk_factors)


# Scenario 3: BCC + relevant symptoms


def test_bcc_with_symptoms(agent):
    result = agent.analyze("bcc", 0.75, ["bleeding", "crusting"])
    assert result.disease_code == "BCC"
    assert result.risk_level in ("LOW", "MODERATE", "HIGH", "URGENT")


# Scenario 4: Benign class + no concerning symptoms


def test_benign_class_no_symptoms_results_in_monitor(agent):
    result = agent.analyze("nv", 0.9, [])
    assert result.risk_level == "LOW"
    assert result.recommendation_category == "MONITOR"


# Scenario 5: Unknown disease


def test_unknown_disease_raises(agent):
    with pytest.raises(UnknownDiseaseError):
        agent.analyze("SCURVY", 0.5, [])


# Scenario 6: Missing symptoms


def test_missing_symptoms_treated_as_empty(agent):
    result = agent.analyze("df", 0.6, None)
    assert result.matched_symptoms == []
    assert result.warning_signs == []


# Scenario 7: Invalid confidence value


def test_invalid_confidence_raises(agent):
    with pytest.raises(RiskEngineError):
        agent.analyze("melanoma", 1.5, [])


# Scenario: request-shape entry point (analyze_request)


def test_analyze_request_valid_shape(agent):
    result = agent.analyze_request(
        {
            "prediction": {"disease": "melanoma", "confidence": 0.87},
            "symptoms": ["bleeding", "itching", "rapid growth"],
        }
    )
    assert result.disease_code == "MELANOMA"


def test_analyze_request_missing_prediction_key_raises(agent):
    with pytest.raises(ClinicalReasoningAgentError):
        agent.analyze_request({"symptoms": ["bleeding"]})


def test_analyze_request_missing_confidence_raises(agent):
    with pytest.raises(ClinicalReasoningAgentError):
        agent.analyze_request({"prediction": {"disease": "melanoma"}})


def test_analyze_request_symptoms_not_a_list_raises(agent):
    with pytest.raises(ClinicalReasoningAgentError):
        agent.analyze_request(
            {"prediction": {"disease": "melanoma", "confidence": 0.8}, "symptoms": "bleeding"}
        )


def test_analyze_request_defaults_symptoms_to_empty_when_absent(agent):
    result = agent.analyze_request({"prediction": {"disease": "nv", "confidence": 0.9}})
    assert result.matched_symptoms == []


def test_reasoning_output_is_fully_traceable(agent):
    result = agent.analyze("akiec", 0.65, ["ulceration"])
    assert result.taxonomy_note is not None
    assert result.taxonomy_note in result.reasoning
    for factor in result.risk_factors:
        assert factor in result.reasoning

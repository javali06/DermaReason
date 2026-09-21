import pytest

from src.config import RiskConfig
from src.knowledge_base import KnowledgeBase, UnknownDiseaseError
from src.risk_engine import RiskEngine, RiskEngineError


@pytest.fixture
def engine():
    return RiskEngine(KnowledgeBase(), RiskConfig())


# Scenario 1: Melanoma + high confidence + concerning symptoms -> should be high risk


def test_melanoma_high_confidence_with_warning_signs_is_high_or_urgent(engine):
    result = engine.assess("melanoma", 0.95, ["bleeding", "itching"])
    assert result.risk_level in ("HIGH", "URGENT")
    assert result.matched_warning_signs  # bleeding/itching match melanoma's warning signs
    assert any("baseline clinical concern level" in f for f in result.risk_factors)
    assert all(isinstance(f, str) for f in result.risk_factors)


# Scenario 2: Melanoma + low confidence -> caution factor applied, still reflects concern level


def test_melanoma_low_confidence_adds_caution_factor(engine):
    result = engine.assess("melanoma", 0.30, [])
    assert result.confidence_band == "low"
    assert any("Model confidence is low" in f for f in result.risk_factors)


# Scenario 3: BCC + relevant symptoms


def test_bcc_with_matching_warning_sign(engine):
    result = engine.assess("bcc", 0.80, ["bleeding"])
    assert result.disease_code == "BCC"
    assert "bleeding" in result.matched_warning_signs or "bleeding" in result.matched_symptoms


# Scenario 4: Benign class + no concerning symptoms -> LOW risk


def test_benign_nv_no_symptoms_is_low_risk(engine):
    result = engine.assess("nv", 0.85, [])
    assert result.risk_level == "LOW"
    assert result.risk_score == 0


# Scenario 5: Unknown disease


def test_unknown_disease_raises(engine):
    with pytest.raises(UnknownDiseaseError):
        engine.assess("NOT_A_DISEASE", 0.9, [])


# Scenario 6: Missing symptoms (None) is treated as empty, not an error


def test_missing_symptoms_defaults_to_empty(engine):
    result = engine.assess("nv", 0.85, None)
    assert result.matched_symptoms == []
    assert result.matched_warning_signs == []
    assert result.unmatched_symptoms == []


# Scenario 7: Invalid confidence value


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1, 2, "0.9", None, True])
def test_invalid_confidence_raises(engine, bad_confidence):
    with pytest.raises(RiskEngineError):
        engine.assess("melanoma", bad_confidence, [])


def test_warning_sign_score_is_capped(engine):
    # melanoma has 9 warning signs; report many overlapping symptoms to exceed the cap
    many_symptoms = ["bleeding", "itching", "asymmetry", "border", "evolving", "crusty", "unusual mark"]
    result = engine.assess("melanoma", 0.6, many_symptoms)
    config = RiskConfig()
    assert len(result.matched_warning_signs) >= config.max_warning_sign_score
    # the capped contribution should show up in the factors text
    assert any("capped" in f for f in result.risk_factors)


def test_risk_factors_are_fully_traceable_strings(engine):
    result = engine.assess("akiec", 0.7, ["ulceration"])
    assert len(result.risk_factors) >= 2
    for factor in result.risk_factors:
        assert isinstance(factor, str) and len(factor) > 0

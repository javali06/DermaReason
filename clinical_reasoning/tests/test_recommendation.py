import pytest

from src.config import RecommendationConfig, RecommendationConfigError
from src.recommendation_engine import RecommendationEngine, RecommendationEngineError


@pytest.fixture
def engine():
    return RecommendationEngine(RecommendationConfig())


@pytest.mark.parametrize(
    "risk_level, expected_category",
    [
        ("LOW", "MONITOR"),
        ("MODERATE", "CONSULT_DERMATOLOGIST"),
        ("HIGH", "URGENT_REFERRAL"),
        ("URGENT", "URGENT_REFERRAL"),
    ],
)
def test_risk_level_maps_to_expected_category(engine, risk_level, expected_category):
    result = engine.recommend(risk_level)
    assert result.category == expected_category
    assert result.risk_level == risk_level
    assert len(result.text) > 0
    assert "prototype" in result.basis.lower()


def test_unknown_risk_level_raises(engine):
    with pytest.raises(RecommendationEngineError):
        engine.recommend("NOT_A_RISK_LEVEL")


def test_no_treatment_language_in_recommendation_text(engine):
    banned_terms = ["mg", "dose", "prescribe", "medication"]
    for risk_level in ("LOW", "MODERATE", "HIGH", "URGENT"):
        text = engine.recommend(risk_level).text.lower()
        for term in banned_terms:
            assert term not in text


def test_config_missing_risk_level_raises(tmp_path):
    import json

    bad_config = {
        "risk_level_to_recommendation": {
            "LOW": "MONITOR",
            "MODERATE": "CONSULT_DERMATOLOGIST",
            "HIGH": "URGENT_REFERRAL",
            # URGENT missing on purpose
        },
        "recommendation_text": {
            "MONITOR": "x",
            "CONSULT_DERMATOLOGIST": "x",
            "URGENT_REFERRAL": "x",
        },
    }
    path = tmp_path / "bad_recommendation_config.json"
    path.write_text(json.dumps(bad_config), encoding="utf-8")
    with pytest.raises(RecommendationConfigError, match="URGENT"):
        RecommendationConfig(path)


def test_config_invalid_category_raises(tmp_path):
    import json

    bad_config = {
        "risk_level_to_recommendation": {
            "LOW": "MONITOR",
            "MODERATE": "CONSULT_DERMATOLOGIST",
            "HIGH": "URGENT_REFERRAL",
            "URGENT": "CALL_AMBULANCE",  # not a valid category
        },
        "recommendation_text": {
            "MONITOR": "x",
            "CONSULT_DERMATOLOGIST": "x",
            "URGENT_REFERRAL": "x",
        },
    }
    path = tmp_path / "bad_recommendation_config.json"
    path.write_text(json.dumps(bad_config), encoding="utf-8")
    with pytest.raises(RecommendationConfigError, match="CALL_AMBULANCE"):
        RecommendationConfig(path)

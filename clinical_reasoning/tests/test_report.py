import pytest

from src.clinical_reasoning_agent import ClinicalReasoningAgent, MEDICAL_DISCLAIMER
from src.report_generator import ReportGenerator, SEPARATOR


@pytest.fixture
def agent():
    return ClinicalReasoningAgent()


def test_report_has_all_required_sections(agent):
    result = agent.analyze("melanoma", 0.87, ["bleeding", "itching", "rapid growth"])
    report = ReportGenerator.generate(result)

    assert report.startswith(SEPARATOR)
    assert report.rstrip().endswith(SEPARATOR)
    for header in (
        "DERMAREASON ANALYSIS REPORT",
        "Predicted Disease:",
        "Model Confidence:",
        "Risk Level:",
        "Clinical Context:",
        "Risk Factors:",
        "Reasoning:",
        "Recommendation:",
        "Medical Disclaimer:",
        "Sources:",
    ):
        assert header in report


def test_report_contains_disclaimer_verbatim(agent):
    result = agent.analyze("melanoma", 0.87, [])
    report = ReportGenerator.generate(result)
    assert MEDICAL_DISCLAIMER in report


def test_report_contains_disease_and_risk_level(agent):
    result = agent.analyze("melanoma", 0.87, ["bleeding", "itching", "rapid growth"])
    report = ReportGenerator.generate(result)
    assert "Melanoma (MELANOMA)" in report
    assert result.risk_level in report


def test_report_lists_all_sources(agent):
    result = agent.analyze("bcc", 0.7, [])
    report = ReportGenerator.generate(result)
    for source in result.sources:
        assert source["title"] in report
        assert source["url"] in report


def test_report_handles_no_matched_symptoms(agent):
    result = agent.analyze("nv", 0.9, [])
    report = ReportGenerator.generate(result)
    assert "No reported symptoms matched the knowledge base" in report


def test_report_shows_unmatched_symptoms_section(agent):
    result = agent.analyze("nv", 0.9, ["headache"])
    report = ReportGenerator.generate(result)
    assert "Unrecognized reported symptoms" in report
    assert "headache" in report


def test_report_includes_taxonomy_note_for_combined_categories(agent):
    result = agent.analyze("akiec", 0.7, [])
    report = ReportGenerator.generate(result)
    assert "Dataset Note:" in report
    assert result.taxonomy_note in report


def test_report_omits_taxonomy_note_for_non_combined_categories(agent):
    result = agent.analyze("melanoma", 0.7, [])
    report = ReportGenerator.generate(result)
    assert "Dataset Note:" not in report

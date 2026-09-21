"""Pydantic schemas for the future API boundary.

These mirror the plain dataclasses used internally (RiskAssessment,
Recommendation, ClinicalReasoningOutput) but add request-time validation
(e.g. confidence must be in [0, 1]) and JSON (de)serialization, which is
what a FastAPI layer needs. Nothing in clinical_reasoning_agent.py,
risk_engine.py, or recommendation_engine.py imports this module or
pydantic - the reasoning engine stays usable as plain Python objects in,
plain Python objects out, independent of any web framework.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionInput(BaseModel):
    """The (disease, confidence) pair a diagnostic model - mock or real - produces."""

    disease: str = Field(..., min_length=1, description="Predicted disease code, e.g. 'MELANOMA'.")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in [0, 1]. This is NOT clinical risk."
    )


class SymptomsInput(BaseModel):
    """Patient-reported symptoms as free-text strings."""

    symptoms: list[str] = Field(default_factory=list)


class ClinicalReasoningInput(BaseModel):
    """Full request payload for the Clinical Reasoning Agent."""

    prediction: PredictionInput
    symptoms: list[str] = Field(default_factory=list)


class SourceSchema(BaseModel):
    title: str
    organization: str
    url: str


class RiskAssessment(BaseModel):
    """API-facing view of a risk assessment result."""

    model_config = ConfigDict(from_attributes=True)

    risk_level: str
    risk_score: int
    risk_factors: list[str]
    confidence_band: str
    matched_symptoms: list[str]
    matched_warning_signs: list[str]
    unmatched_symptoms: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """API-facing view of a recommendation result."""

    model_config = ConfigDict(from_attributes=True)

    risk_level: str
    category: str
    text: str
    basis: str


class ClinicalReasoningOutput(BaseModel):
    """API-facing view of the full Clinical Reasoning Agent output.

    Field names and shape mirror
    src.clinical_reasoning_agent.ClinicalReasoningOutput exactly, so it can
    be built directly from that dataclass via .model_validate(result).
    """

    model_config = ConfigDict(from_attributes=True)

    disease_code: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_band: str
    matched_symptoms: list[str]
    warning_signs: list[str]
    unmatched_symptoms: list[str]
    risk_level: str
    risk_score: int
    risk_factors: list[str]
    recommendation_category: str
    recommendation_text: str
    reasoning: str
    sources: list[SourceSchema]
    taxonomy_note: str | None = None
    disclaimer: str


class ReportOutput(BaseModel):
    """The final formatted report, alongside the structured data it was built from."""

    report_text: str
    result: ClinicalReasoningOutput

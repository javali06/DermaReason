"""Formats a ClinicalReasoningOutput into a human-readable report.

This module makes no medical decisions and computes nothing - it only
formats structured results already produced by the Clinical Reasoning
Agent (which itself only assembles results from the KnowledgeBase,
RiskEngine, and RecommendationEngine).
"""

from __future__ import annotations

from .clinical_reasoning_agent import ClinicalReasoningOutput

SEPARATOR = "-" * 38


class ReportGenerator:
    """Formats a ClinicalReasoningOutput as a plain-text report."""

    @staticmethod
    def generate(result: ClinicalReasoningOutput) -> str:
        lines: list[str] = []

        lines.append(SEPARATOR)
        lines.append("DERMAREASON ANALYSIS REPORT")
        lines.append(SEPARATOR)
        lines.append("")

        lines.append("Predicted Disease:")
        lines.append(f"{result.display_name} ({result.disease_code})")
        lines.append("")

        lines.append("Model Confidence:")
        lines.append(f"{result.confidence:.2f} ({result.confidence_band} confidence band)")
        lines.append("")

        lines.append("Risk Level:")
        lines.append(result.risk_level)
        lines.append("")

        lines.append("Clinical Context:")
        if result.matched_symptoms or result.warning_signs:
            for symptom in result.matched_symptoms:
                lines.append(f"- {symptom} (general symptom)")
            for sign in result.warning_signs:
                lines.append(f"- {sign} (warning sign)")
        else:
            lines.append("No reported symptoms matched the knowledge base for this disease.")

        if result.unmatched_symptoms:
            lines.append("")
            lines.append("Unrecognized reported symptoms (not matched to any knowledge base entry):")
            for symptom in result.unmatched_symptoms:
                lines.append(f"- {symptom}")
        lines.append("")

        lines.append("Risk Factors:")
        for factor in result.risk_factors:
            lines.append(f"- {factor}")
        lines.append("")

        if result.taxonomy_note:
            lines.append("Dataset Note:")
            lines.append(result.taxonomy_note)
            lines.append("")

        lines.append("Reasoning:")
        lines.append(result.reasoning)
        lines.append("")

        lines.append("Recommendation:")
        lines.append(f"[{result.recommendation_category}] {result.recommendation_text}")
        lines.append("")

        lines.append("Medical Disclaimer:")
        lines.append(result.disclaimer)
        lines.append("")

        lines.append("Sources:")
        for source in result.sources:
            lines.append(f"- {source['title']} ({source['organization']}) - {source['url']}")

        lines.append(SEPARATOR)

        return "\n".join(lines)

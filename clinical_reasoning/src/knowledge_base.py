"""Loader and accessor for the DermaReason dermatology knowledge base.

This module is the only code allowed to read knowledge_base/*.json directly.
Everything else in the project must go through the KnowledgeBase class so
that malformed or missing data fails loudly instead of silently producing
incorrect medical information.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

VALID_CONCERN_LEVELS = {
    "malignant_high",
    "locally_invasive_moderate",
    "precancerous_moderate",
    "benign_low",
}

REQUIRED_STRING_FIELDS = (
    "disease_code",
    "display_name",
    "description",
    "general_concern_level",
    "recommendation",
    "last_reviewed",
)
REQUIRED_LIST_FIELDS = ("symptoms", "warning_signs", "sources")


class KnowledgeBaseError(Exception):
    """Base class for all knowledge base errors."""


class KnowledgeBaseLoadError(KnowledgeBaseError):
    """Raised when the knowledge base directory or one of its files cannot be loaded or is invalid."""


class UnknownDiseaseError(KnowledgeBaseError):
    """Raised when a disease code has no matching knowledge base entry."""


def _normalize(disease_name: str) -> str:
    if not isinstance(disease_name, str) or not disease_name.strip():
        raise UnknownDiseaseError(f"Disease name must be a non-empty string, got: {disease_name!r}")
    return disease_name.strip().upper()


class KnowledgeBase:
    """Loads and serves the curated dermatology knowledge base.

    Usage:
        kb = KnowledgeBase()
        kb.load_all()
        entry = kb.get_disease("melanoma")
    """

    def __init__(self, knowledge_base_dir: str | Path | None = None):
        self._dir = Path(knowledge_base_dir) if knowledge_base_dir is not None else DEFAULT_KNOWLEDGE_BASE_DIR
        self._diseases: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Load and validate every JSON file in the knowledge base directory.

        Raises KnowledgeBaseLoadError with an aggregated, human-readable list
        of every problem found, rather than stopping at the first one.
        """
        if not self._dir.exists() or not self._dir.is_dir():
            raise KnowledgeBaseLoadError(f"Knowledge base directory not found: {self._dir}")

        json_files = sorted(self._dir.glob("*.json"))
        if not json_files:
            raise KnowledgeBaseLoadError(f"No knowledge base JSON files found in: {self._dir}")

        diseases: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for path in json_files:
            try:
                raw_text = path.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"{path.name}: could not read file ({exc})")
                continue

            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}: invalid JSON ({exc})")
                continue

            entry_errors = self._entry_problems(data)
            if entry_errors:
                errors.append(f"{path.name}: " + "; ".join(entry_errors))
                continue

            code = _normalize(data["disease_code"])
            if code in diseases:
                errors.append(
                    f"{path.name}: duplicate disease_code '{code}' "
                    f"(already loaded from another file)"
                )
                continue

            diseases[code] = data

        if errors:
            raise KnowledgeBaseLoadError(
                "Knowledge base failed validation:\n- " + "\n- ".join(errors)
            )

        self._diseases = diseases
        self._loaded = True

    @staticmethod
    def _entry_problems(data: Any) -> list[str]:
        """Return a list of validation problems for a single parsed entry (empty list if valid)."""
        problems: list[str] = []

        if not isinstance(data, dict):
            return ["top-level JSON must be an object"]

        for field in REQUIRED_STRING_FIELDS:
            if field not in data:
                problems.append(f"missing required field '{field}'")
            elif not isinstance(data[field], str) or not data[field].strip():
                problems.append(f"field '{field}' must be a non-empty string")

        for field in REQUIRED_LIST_FIELDS:
            if field not in data:
                problems.append(f"missing required field '{field}'")
            elif not isinstance(data[field], list):
                problems.append(f"field '{field}' must be a list")

        if "general_concern_level" in data and data["general_concern_level"] not in VALID_CONCERN_LEVELS:
            problems.append(
                f"general_concern_level '{data.get('general_concern_level')}' "
                f"is not one of {sorted(VALID_CONCERN_LEVELS)}"
            )

        if "sources" in data and isinstance(data["sources"], list):
            if len(data["sources"]) == 0:
                problems.append("'sources' must contain at least one source")
            for i, source in enumerate(data["sources"]):
                if not isinstance(source, dict) or not all(
                    key in source and isinstance(source[key], str) and source[key].strip()
                    for key in ("title", "organization", "url")
                ):
                    problems.append(
                        f"sources[{i}] must be an object with non-empty 'title', "
                        f"'organization', and 'url' string fields"
                    )

        if "taxonomy_note" in data and not isinstance(data["taxonomy_note"], str):
            problems.append("field 'taxonomy_note' must be a string when present")

        return problems

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_all()

    def validate(self) -> bool:
        """Force (re)loading and validation of the knowledge base.

        Returns True if the knowledge base is valid. Raises KnowledgeBaseLoadError
        otherwise — validation failures are never silently swallowed.
        """
        self.load_all()
        return True

    def get_all_diseases(self) -> list[str]:
        """Return the sorted list of known, normalized disease codes."""
        self._ensure_loaded()
        return sorted(self._diseases.keys())

    def get_disease(self, disease_name: str) -> dict[str, Any]:
        """Return the full knowledge base entry for a disease code (case/whitespace insensitive).

        Raises UnknownDiseaseError if the disease code has no entry.
        Returns a deep copy so callers cannot mutate the loaded knowledge base in place.
        """
        self._ensure_loaded()
        code = _normalize(disease_name)
        if code not in self._diseases:
            raise UnknownDiseaseError(
                f"Unknown disease code '{disease_name}'. Known codes: {self.get_all_diseases()}"
            )
        return copy.deepcopy(self._diseases[code])

    def get_symptoms(self, disease_name: str) -> list[str]:
        return list(self.get_disease(disease_name)["symptoms"])

    def get_warning_signs(self, disease_name: str) -> list[str]:
        return list(self.get_disease(disease_name)["warning_signs"])

    def get_recommendation(self, disease_name: str) -> str:
        return self.get_disease(disease_name)["recommendation"]

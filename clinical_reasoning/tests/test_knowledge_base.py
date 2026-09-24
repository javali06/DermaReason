import json

import pytest

from src.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseLoadError,
    UnknownDiseaseError,
)

EXPECTED_DISEASE_CODES = ["AKIEC", "BCC", "BKL", "DF", "MELANOMA", "NV", "VASC"]


def make_valid_entry(disease_code="MELANOMA"):
    return {
        "disease_code": disease_code,
        "display_name": "Melanoma",
        "description": "A test description.",
        "general_concern_level": "malignant_high",
        "symptoms": ["itching"],
        "warning_signs": ["rapid growth"],
        "recommendation": "See a dermatologist.",
        "sources": [
            {"title": "Test Source", "organization": "Test Org", "url": "https://example.com"}
        ],
        "last_reviewed": "2026-09-21",
    }


# --- Tests against the real bundled knowledge base ---


def test_load_all_real_knowledge_base():
    kb = KnowledgeBase()
    kb.load_all()
    assert kb.get_all_diseases() == EXPECTED_DISEASE_CODES


def test_get_disease_is_case_and_whitespace_insensitive():
    kb = KnowledgeBase()
    entry_lower = kb.get_disease("melanoma")
    entry_upper = kb.get_disease("MELANOMA")
    entry_padded = kb.get_disease("  Melanoma  ")
    assert entry_lower == entry_upper == entry_padded
    assert entry_lower["disease_code"] == "MELANOMA"


def test_get_disease_returns_a_copy_not_a_live_reference():
    kb = KnowledgeBase()
    entry = kb.get_disease("nv")
    entry["symptoms"].append("tampered")
    fresh_entry = kb.get_disease("nv")
    assert "tampered" not in fresh_entry["symptoms"]


def test_get_symptoms_warning_signs_recommendation():
    kb = KnowledgeBase()
    assert "bleeding" in " ".join(kb.get_symptoms("melanoma")).lower() or len(kb.get_symptoms("melanoma")) > 0
    assert len(kb.get_warning_signs("melanoma")) > 0
    assert isinstance(kb.get_recommendation("melanoma"), str)
    assert len(kb.get_recommendation("melanoma")) > 0


def test_unknown_disease_raises():
    kb = KnowledgeBase()
    with pytest.raises(UnknownDiseaseError):
        kb.get_disease("NOT_A_REAL_DISEASE")


def test_lazy_loading_without_explicit_load_all():
    kb = KnowledgeBase()
    # get_disease should trigger load_all() implicitly
    entry = kb.get_disease("bcc")
    assert entry["disease_code"] == "BCC"


def test_validate_real_knowledge_base_succeeds():
    kb = KnowledgeBase()
    assert kb.validate() is True


def test_taxonomy_note_present_only_on_combined_categories():
    kb = KnowledgeBase()
    for code in ("AKIEC", "BKL", "VASC"):
        assert "taxonomy_note" in kb.get_disease(code)
    for code in ("MELANOMA", "BCC", "NV", "DF"):
        assert "taxonomy_note" not in kb.get_disease(code)


# --- Edge case / error handling tests, using temporary directories ---


def test_missing_directory_raises(tmp_path):
    kb = KnowledgeBase(tmp_path / "does_not_exist")
    with pytest.raises(KnowledgeBaseLoadError):
        kb.load_all()


def test_empty_directory_raises(tmp_path):
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError):
        kb.load_all()


def test_malformed_json_raises(tmp_path):
    (tmp_path / "broken.json").write_text("{ this is not valid json ", encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError, match="invalid JSON"):
        kb.load_all()


def test_missing_required_field_raises(tmp_path):
    entry = make_valid_entry()
    del entry["symptoms"]
    (tmp_path / "melanoma.json").write_text(json.dumps(entry), encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError, match="symptoms"):
        kb.load_all()


def test_invalid_concern_level_raises(tmp_path):
    entry = make_valid_entry()
    entry["general_concern_level"] = "super_duper_dangerous"
    (tmp_path / "melanoma.json").write_text(json.dumps(entry), encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError, match="general_concern_level"):
        kb.load_all()


def test_source_missing_url_raises(tmp_path):
    entry = make_valid_entry()
    entry["sources"] = [{"title": "Test", "organization": "Test Org"}]
    (tmp_path / "melanoma.json").write_text(json.dumps(entry), encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError, match="sources"):
        kb.load_all()


def test_duplicate_disease_code_raises(tmp_path):
    entry_a = make_valid_entry(disease_code="MELANOMA")
    entry_b = make_valid_entry(disease_code="melanoma")
    (tmp_path / "a.json").write_text(json.dumps(entry_a), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(entry_b), encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError, match="duplicate"):
        kb.load_all()


def test_load_all_aggregates_multiple_errors(tmp_path):
    bad_entry = make_valid_entry()
    del bad_entry["recommendation"]
    (tmp_path / "bad.json").write_text(json.dumps(bad_entry), encoding="utf-8")
    (tmp_path / "broken.json").write_text("not json at all", encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    with pytest.raises(KnowledgeBaseLoadError) as exc_info:
        kb.load_all()
    message = str(exc_info.value)
    assert "bad.json" in message
    assert "broken.json" in message

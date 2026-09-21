from src.symptom_matcher import match_patient_symptoms

KB_SYMPTOMS = ["May itch or bleed", "Nodular lesions may be elevated, firm to touch, and growing"]
KB_WARNING_SIGNS = ["Evolving - changing in size, shape, or color over time (ABCDE rule)", "Itching or bleeding"]


def test_matches_warning_sign_over_symptom_when_both_overlap():
    # "bleeding" overlaps both the symptoms text ("bleed") and the warning
    # signs text ("bleeding") - warning signs should take priority.
    matched, warnings, unmatched = match_patient_symptoms(["bleeding"], KB_SYMPTOMS, KB_WARNING_SIGNS)
    assert matched == []
    assert warnings == ["bleeding"]
    assert unmatched == []


def test_matches_growth_variant_via_stemming():
    matched, warnings, unmatched = match_patient_symptoms(["growth"], KB_SYMPTOMS, KB_WARNING_SIGNS)
    assert matched == ["growth"]
    assert warnings == []


def test_unmatched_symptom_is_reported_not_dropped():
    matched, warnings, unmatched = match_patient_symptoms(["headache"], KB_SYMPTOMS, KB_WARNING_SIGNS)
    assert matched == []
    assert warnings == []
    assert unmatched == ["headache"]


def test_preserves_original_wording_and_order():
    matched, warnings, unmatched = match_patient_symptoms(
        ["Bleeding", "headache", "growing"], KB_SYMPTOMS, KB_WARNING_SIGNS
    )
    assert warnings == ["Bleeding"]
    assert matched == ["growing"]
    assert unmatched == ["headache"]


def test_empty_patient_symptoms_returns_empty_lists():
    matched, warnings, unmatched = match_patient_symptoms([], KB_SYMPTOMS, KB_WARNING_SIGNS)
    assert matched == warnings == unmatched == []

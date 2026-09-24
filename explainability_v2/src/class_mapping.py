"""
Class mapping shared across the DermaReason pipeline.

CRITICAL: this must stay identical to the mapping used by the
Diagnostic Agent (Person 1) and the Clinical Reasoning Agent
(Person 2). Do not reorder or rename these without updating both.
"""

CLASS_MAPPING = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc",
}

# Human-readable names, used only for display / report text.
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

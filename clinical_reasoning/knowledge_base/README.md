# DermaReason Dermatology Knowledge Base

## Purpose

This directory is the single source of dermatological facts used by the Clinical
Reasoning Agent. It exists so that the reasoning agent never has to invent or
recall medical information on its own — it only looks up pre-curated,
source-cited entries. If a fact is not in one of these JSON files, the
reasoning agent must not claim it.

Each file corresponds to one of the seven disease classes the (separately
developed) diagnostic ML model is expected to output:

| `disease_code` | File          | Display name                                              |
|----------------|----------------|-------------------------------------------------------------|
| `MELANOMA`     | `melanoma.json`| Melanoma                                                     |
| `BCC`          | `bcc.json`     | Basal Cell Carcinoma                                         |
| `AKIEC`        | `akiec.json`   | Actinic Keratosis / Intraepithelial Carcinoma (Bowen's disease) |
| `BKL`          | `bkl.json`     | Benign Keratosis-like Lesions                                |
| `NV`           | `nv.json`      | Melanocytic Nevi                                             |
| `DF`           | `df.json`      | Dermatofibroma                                               |
| `VASC`         | `vasc.json`    | Vascular Lesions                                             |

These `disease_code` values must match exactly what the diagnostic model
emits. They are never renamed or reinterpreted downstream.

## Schema

```json
{
    "disease_code": "AKIEC",
    "display_name": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen's Disease)",
    "description": "Established medical description, 2-4 sentences.",
    "taxonomy_note": "Optional. Only present on AKIEC, BKL, VASC — see 'Combined dataset categories' below.",
    "general_concern_level": "precancerous_moderate",
    "symptoms": ["..."],
    "warning_signs": ["..."],
    "recommendation": "General next-step guidance, not a treatment plan.",
    "sources": [
        { "title": "...", "organization": "...", "url": "..." }
    ],
    "last_reviewed": "2026-09-21"
}
```

| Field                  | Meaning |
|------------------------|---------|
| `disease_code`         | Short code matching the ML model's output label. Used as the lookup key. |
| `display_name`         | Human-readable clinical name for reports/UI. |
| `description`          | Plain-language summary of the condition, drawn from cited sources. |
| `taxonomy_note`        | Present only where the dataset label groups multiple distinct clinical entities together (see below). Explains the grouping so downstream code/readers don't mistake the entry for a single clean diagnosis. |
| `general_concern_level`| One of `malignant_high`, `locally_invasive_moderate`, `precancerous_moderate`, `benign_low`. **This is our own engineering categorization for the prototype**, informed by how sources describe the condition's behavior — it is not a formally validated clinical severity scale. |
| `symptoms`             | Signs a patient/clinician might observe, taken only from cited sources. |
| `warning_signs`        | Features sources associate with malignancy, progression, or the need for evaluation. |
| `recommendation`       | General next-step guidance *as stated by sources* (e.g. "see a dermatologist"). Never a treatment/medication plan. |
| `sources`              | Every source used to populate this entry. |
| `last_reviewed`        | Date this entry's content was last checked against its sources. |

## Sources used

All entries draw only from: American Academy of Dermatology (aad.org),
National Cancer Institute (cancer.gov), NHS (nhs.uk), DermNet NZ
(dermnetnz.org), the National Organization for Rare Disorders (rarediseases.org),
Cleveland Clinic, and one peer-reviewed article via PMC/NCBI. Every fact in
every JSON file is traceable to one of the sources listed in that file's
`sources` array. Full citations with direct links are inside each file.

**Known sourcing gaps** (documented here for transparency, not hidden):
- `df.json` and `vasc.json` have weaker cross-verification than the others —
  AAD/NHS do not appear to publish dedicated pages for dermatofibroma or for
  the individual vascular lesion types, so DermNet NZ and (for one claim)
  PMC/NCBI are the primary sources.
- `akiec.json`: reported progression-risk figures for actinic keratosis
  vary substantially across sources depending on whether they're per-lesion,
  per-patient, or lifetime risk (from <0.1% to 10-15%). No single number is
  presented as authoritative in the entry; only qualitative warning signs are used.
- `nv.json`: sources disagree on the exact age threshold for "a new mole is
  more concerning" (30 vs. 40). Both figures are mentioned in the entry
  rather than resolved into a single number.

## Combined dataset categories (`AKIEC`, `BKL`, `VASC`)

Three of the seven classes are not single clinical diagnoses — they are
grouped labels from the HAM10000/ISIC dataset taxonomy the diagnostic model
is trained against:

- **AKIEC** = actinic keratosis (precancerous) + Bowen's disease / SCC-in-situ
  (already carcinoma in situ). These carry different concern levels.
- **BKL** = seborrheic keratosis + solar lentigo + lichen-planus-like
  keratosis. Mostly benign, documented primarily via seborrheic keratosis.
- **VASC** = cherry angioma + angiokeratoma + pyogenic granuloma +
  hemorrhage/hematoma. Mostly incidental, except pyogenic granuloma and
  hematoma both have documented melanoma-mimicry.

Each affected file has a `taxonomy_note` field explaining this. Any code
that consumes these entries (recommendation logic, report text) should treat
the entry as an approximation across the grouped entities, not a single
precise diagnosis — and should surface the `warning_signs` for the
higher-concern sub-entity rather than defaulting to the most benign one.

## Medical facts vs. project-specific rules — a hard boundary

Everything in these JSON files (`description`, `symptoms`, `warning_signs`,
source-stated parts of `recommendation`) is an **established medical fact**,
traceable to a cited source.

`general_concern_level` is the one field that is **our own engineering
judgment**, used only to give the downstream risk engine a starting
categorical signal. It is explicitly *not* claimed to be a clinically
validated severity score.

Everything else that the wider Clinical Reasoning module does — confidence
thresholds, risk-level rules (LOW/MODERATE/HIGH/URGENT), and
recommendation-category mappings (MONITOR/CONSULT_DERMATOLOGIST/URGENT_REFERRAL)
— lives in separate config/engine files outside this knowledge base
(`src/config.py`, `src/risk_engine.py`, `src/recommendation_engine.py`, to be
built in later stages) and is documented there as **prototype engineering
rules, not clinical guidelines**. This directory only ever answers "what do
authoritative sources say about this disease" — it never answers "what
should the system recommend."

## Updating an entry

1. Find or fetch a current, authoritative source (prefer AAD, NCI, NHS, or
   DermNet NZ; peer-reviewed literature is acceptable for gaps).
2. Only add facts the source actually states — do not paraphrase into a
   stronger or more specific claim than the source supports.
3. Add the citation to `sources` with `title`, `organization`, and `url`.
4. Update `last_reviewed` to the date of the change.
5. If a change affects `general_concern_level`, explain the reasoning in the
   pull request / commit message, since this field is our own judgment call,
   not a direct citation.
6. Never delete a `warning_sign` or `symptom` without checking whether
   removing it changes the clinical safety posture of the entry.

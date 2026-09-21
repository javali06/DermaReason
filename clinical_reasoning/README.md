# DermaReason — Clinical Reasoning & Knowledge Base Module

This module implements the Clinical Reasoning Agent, Dermatology Knowledge
Base, Risk Assessment, Recommendation Engine, and Report Generation stages
of the DermaReason architecture:

```
Image + Symptoms
       |
Preprocessing Agent            )
Diagnostic Agent               )  NOT implemented here - mocked (see mock/)
Confidence Estimation Agent    )
       |
Clinical Reasoning Agent       )
Explainability Agent           )  <- this module
Recommendation Agent           )
Report Generation Agent        )
```

This module is completely independent of the ML model. It only ever
consumes `{"disease": <code>, "confidence": <float>}` plus a list of
patient-reported symptom strings - it does not know or care whether that
prediction came from a real diagnostic model or `mock/mock_model.py`.

## Module map

```
clinical_reasoning/
├── config/
│   ├── risk_config.json            prototype confidence thresholds & risk weights
│   └── recommendation_config.json  prototype risk-level -> recommendation mapping
├── knowledge_base/
│   ├── *.json                      one sourced entry per disease class
│   └── README.md                   schema, sourcing, update process
├── src/
│   ├── knowledge_base.py           loads/validates knowledge_base/*.json
│   ├── config.py                   loads/validates the two config files above
│   ├── symptom_matcher.py          deterministic keyword matching (patient symptoms <-> KB text)
│   ├── risk_engine.py              rule-based, explainable risk scoring
│   ├── recommendation_engine.py    risk_level -> MONITOR/CONSULT/URGENT_REFERRAL
│   ├── clinical_reasoning_agent.py orchestrates the above into one result
│   ├── report_generator.py         formats a result as a text report
│   └── schemas.py                  Pydantic models for a future API layer
├── mock/
│   ├── mock_prediction.json        stand-in for the real diagnostic model's output
│   ├── mock_scenarios.json         a few named scenarios for demos/tests
│   └── mock_model.py               MockDiagnosticModel (same interface a real model needs)
├── tests/                          one test file per module above
├── conftest.py                     makes `src`/`mock` importable for pytest
└── requirements.txt
```

## Data flow

```
MockDiagnosticModel.predict()  (or, later, the real Diagnostic Agent)
        -> {"disease": str, "confidence": float}
                |
                v
ClinicalReasoningAgent.analyze(disease, confidence, symptoms)
        -> looks up KnowledgeBase.get_disease(disease)
        -> RiskEngine.assess(...)          (confidence band + rule-based risk_level)
        -> RecommendationEngine.recommend(risk_level)
        -> ClinicalReasoningOutput (dataclass)
                |
                v
ReportGenerator.generate(result) -> plain-text report

Optionally: schemas.ClinicalReasoningOutput.model_validate(result)
            -> Pydantic model, ready for a FastAPI response
```

## Design principles this module follows

- **Model confidence != clinical risk.** `RiskEngine` treats them as separate
  inputs; a low-confidence prediction adds an explicit *caution* factor, it
  never gets relabeled as clinical severity.
- **No hidden rules.** Every contribution to a risk score appears as a
  plain-English entry in `risk_factors`, and the final `reasoning` text is
  assembled only from those factors plus fixed templates - never freely
  generated.
- **Configurable, not hardcoded.** Confidence thresholds, concern-level
  weights, and the risk-to-recommendation mapping all live in
  `config/*.json`, each carrying an explicit `_disclaimer` that the values
  are prototype engineering choices, not clinically validated thresholds.
- **Fail loudly.** `KnowledgeBase`, `RiskConfig`, `RecommendationConfig`, and
  `MockDiagnosticModel` all raise clear, typed errors on missing/malformed
  data instead of silently returning something plausible-looking but wrong.
- **Framework-independent core.** `schemas.py` is the only file that imports
  `pydantic`; nothing in `risk_engine.py`, `recommendation_engine.py`, or
  `clinical_reasoning_agent.py` depends on it, so the reasoning engine works
  as plain Python regardless of what (if any) API framework wraps it later.

## Installation

```bash
cd clinical_reasoning
pip install -r requirements.txt
```

## Running the pipeline

```bash
python -c "
from mock.mock_model import MockDiagnosticModel
from src.clinical_reasoning_agent import ClinicalReasoningAgent
from src.report_generator import ReportGenerator

model = MockDiagnosticModel()
prediction = model.predict()
agent = ClinicalReasoningAgent()
result = agent.analyze(prediction['disease'], prediction['confidence'], ['bleeding', 'itching'])
print(ReportGenerator.generate(result))
"
```

## Running the tests

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (85 at time of writing). Each test file targets one
module: `test_knowledge_base.py`, `test_symptom_matcher.py`,
`test_risk_engine.py`, `test_recommendation.py`, `test_reasoning.py`
(end-to-end agent), `test_report.py`, `test_mock_model.py`, `test_schemas.py`.

## Important limitations (read before presenting this project)

- This is a decision-support prototype, **not** a diagnostic tool and **not**
  a replacement for a dermatologist. Every generated report carries an
  explicit medical disclaimer.
- Three disease classes (`AKIEC`, `BKL`, `VASC`) are combined dataset
  categories covering multiple distinct clinical entities - see each JSON
  file's `taxonomy_note` and `knowledge_base/README.md` for details.
- Risk score weights, confidence-band thresholds, and the risk-to-
  recommendation mapping are this project's own engineering choices,
  informed by general patterns in the cited sources but not themselves
  clinically validated - see `config/risk_config.json` and
  `config/recommendation_config.json`.
- Symptom matching (`symptom_matcher.py`) uses simple keyword/stem overlap,
  not clinical NLP - it is deterministic and explainable, but will miss
  paraphrased or synonymous symptom descriptions it wasn't designed to catch.

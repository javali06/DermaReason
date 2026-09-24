# DermaReason Demo UI

This project provides a lightweight Gradio app for the DermaReason pipeline.

## What the app does

1. Accepts a skin image upload
2. Runs a diagnostic prediction model or a mock fallback when a model weight file is not available
3. Sends the predicted disease and confidence to the clinical reasoning engine
4. Produces a risk level and recommendation
5. Generates a Grad-CAM-style heatmap and overlay image for explainability
6. Displays the final clinical report in the UI

## Folder structure

- `app.py` — Gradio UI and main pipeline entry point
- `clinical_reasoning/` — risk assessment, knowledge base, recommendation logic, and report generation
- `explainability_v2/` — model explainability and Grad-CAM logic
- `sample_outputs/` — generated original, heatmap, and overlay images

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

From the project root:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

## Optional model weights

If you have a trained `best_efficientnet_b0.pth` model file, you can upload it in the UI through the optional file selector. If no weights file is present, the app automatically falls back to the mock prediction flow so the interface still works.

## Expected workflow

Upload Skin Image
↓
Predicted Disease + Confidence
↓
Risk Level + Recommendation
↓
Heatmap + Overlay
↓
Clinical Report

## Notes

- This is a prototype decision-support UI and should not be treated as a clinical diagnostic tool.
- The reasoning engine is explainable and rule-based, not a validated medical decision engine.
- The UI is meant to demo the pipeline end-to-end in a user-friendly way.

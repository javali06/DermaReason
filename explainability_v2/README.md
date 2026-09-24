# Explainability Agent (DermaReason)

Visual explainability for the Diagnostic Agent's predictions, via
Grad-CAM on the trained EfficientNet-B0. This is the piece flagged as
missing in the Clinical Reasoning writeup:

> "our reasoning text covers textual explainability for the clinical
> decision, but not visual/image-level explainability, which depends
> on the (unbuilt) diagnostic model."

The diagnostic model now exists (`best_efficientnet_b0.pth`), so this
module implements both the real prediction step and the Grad-CAM
explanation on top of it.

## Contract compatibility

`ExplainabilityAgent.predict(image_path)` returns exactly the same
shape as `mock.mock_model.MockDiagnosticModel.predict()`:

```python
{"disease": "mel", "confidence": 0.91}
```

This was intentional — the Clinical Reasoning Agent, `schemas.py`,
and the 85 existing tests built against the mock model can point at
this class instead with no changes anywhere else in the pipeline.

`ExplainabilityAgent.explain(image_path)` returns the same two keys,
extended with the Grad-CAM outputs:

```python
{
    "disease": "mel",
    "confidence": 0.91,
    "original_path": "sample_outputs/xxx_original.png",
    "heatmap_path": "sample_outputs/xxx_heatmap.png",
    "overlay_path": "sample_outputs/xxx_overlay.png",
}
```

## Files

```
explainability/
├── src/
│   ├── class_mapping.py       # shared CLASS_MAPPING (must match all agents)
│   ├── gradcam.py              # Grad-CAM core: hooks + CAM computation
│   └── explainability_agent.py # ExplainabilityAgent class (predict / explain)
├── demo.py                     # runnable CLI demo
├── sample_outputs/             # generated original/heatmap/overlay images
└── README.md
```

## Setup

```bash
pip install torch timm opencv-python-headless numpy
```

Place `best_efficientnet_b0.pth` in this folder (or pass its path to
`demo.py` / `ExplainabilityAgent(...)`).

## Usage

```python
from src.explainability_agent import ExplainabilityAgent

agent = ExplainabilityAgent("best_efficientnet_b0.pth", device="cpu")

# Drop-in replacement for MockDiagnosticModel.predict()
result = agent.predict("path/to/image.jpg")
# {"disease": "mel", "confidence": 0.91}

# Same contract + Grad-CAM visuals
explained = agent.explain("path/to/image.jpg")
```

Or from the command line:

```bash
python demo.py best_efficientnet_b0.pth sample_image.jpg
```

## Preprocessing note (important)

Preprocessing exactly mirrors `HAMDataset` from the training
notebook: `BGR->RGB`, scale to `[0, 1]`. **No ImageNet mean/std
normalization is applied**, because none was applied during training
either. Diverging from this silently degrades both prediction
accuracy and Grad-CAM heatmap quality.

## Class mapping

```python
{0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
```

Kept identical to the mapping used by the Diagnostic Agent and the
Clinical Reasoning Agent's knowledge base disease codes.

## Verified

Tested end-to-end against the real trained weights and 3 sample HAM10000
images. `predict()`'s output shape was checked against the documented
`MockDiagnosticModel.predict()` contract (`{"disease": str, "confidence": float}`)
and confirmed to match key-for-key.

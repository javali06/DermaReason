from __future__ import annotations

import socket
import sys
import uuid
from pathlib import Path

from typing import Optional

import cv2
import numpy as np
import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLINICAL_REASONING_DIR = ROOT / "clinical_reasoning"
EXPLAINABILITY_DIR = ROOT / "explainability_v2"

sys.path.insert(0, str(CLINICAL_REASONING_DIR))
sys.path.insert(0, str(EXPLAINABILITY_DIR))
sys.path.insert(0, str(EXPLAINABILITY_DIR / "src"))

from clinical_reasoning.mock.mock_model import MockDiagnosticModel
from clinical_reasoning.src.clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning.src.report_generator import ReportGenerator
from explainability_v2.src.explainability_agent import ExplainabilityAgent

MODEL_CANDIDATES = [
    ROOT / "best_efficientnet_b0.pth",
    ROOT / "efficientnet_b0.pth",
    ROOT / "explainability_v2" / "best_efficientnet_b0.pth",
    ROOT / "explainability_v2" / "effective_model.pth",
    ROOT / "explainability_v2" / "src" / "best_efficientnet_b0.pth",
]
OUTPUT_DIR = ROOT / "sample_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _find_available_port(start_port=7860, end_port=7900):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port available in range {start_port}-{end_port}")


def _resolve_weights(weights_file=None):
    if weights_file is not None:
        p = Path(weights_file)
        if p.exists():
            return str(p)

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    found = []
    for path in sorted(ROOT.rglob("*.pth")):
        if "sample_outputs" in str(path):
            continue
        found.append(str(path))

    if found:
        return found[0]

    return None


def _make_fallback_heatmap_and_overlay(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    yy, xx = np.mgrid[0:h, 0:w]
    center_y = h / 2
    center_x = w / 2
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    norm = np.clip(1 - dist / max(h, w) * 1.6, 0, 1)
    heatmap = np.zeros_like(rgb, dtype=np.uint8)
    heatmap[..., 2] = (255 * norm).astype(np.uint8)
    heatmap[..., 1] = (100 * norm).astype(np.uint8)

    overlay = cv2.addWeighted(rgb, 0.7, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR), 0.3, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    base = Path(image_path)
    base_name = base.stem
    original_path = OUTPUT_DIR / f"{base_name}_original.png"
    heatmap_path = OUTPUT_DIR / f"{base_name}_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{base_name}_overlay.png"

    cv2.imwrite(str(original_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "disease": "mel",
        "confidence": 0.5,
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }


def _read_uploaded_image(image):
    if image is None:
        raise ValueError("Please upload a skin image first.")

    if isinstance(image, Image.Image):
        pil_img = image
    else:
        pil_img = Image.fromarray(np.asarray(image))

    temp_dir = ROOT / "tmp_ui"
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / f"upload_{uuid.uuid4().hex}.png"
    pil_img.save(file_path)
    return str(file_path)


def process_image(image):
    if image is None:
        raise gr.Error("Please upload a skin image first.")

    image_path = _read_uploaded_image(image)
    weights_path = _resolve_weights()

    if weights_path is not None:
        explainer = ExplainabilityAgent(weights_path, device="cpu", output_dir=str(OUTPUT_DIR))
        prediction = explainer.predict(image_path)
        explained = explainer.explain(image_path)
        model_status = f"Using trained model: {Path(weights_path).name}"
    else:
        prediction = MockDiagnosticModel().predict(image_path)
        explained = _make_fallback_heatmap_and_overlay(image_path)
        prediction = {
            "disease": explained["disease"],
            "confidence": explained["confidence"],
        }
        model_status = "No .pth model found. Using mock model fallback."

    reasoning_agent = ClinicalReasoningAgent()
    result = reasoning_agent.analyze(prediction["disease"], prediction["confidence"], [])
    report = ReportGenerator.generate(result)

    if not hasattr(explained, "keys"):
        explained = dict(explained)

    original_image = Image.open(explained.get("original_path", image_path))
    heatmap_image = Image.open(explained.get("heatmap_path", explained.get("original_path", image_path)))
    overlay_image = Image.open(explained.get("overlay_path", explained.get("original_path", image_path)))

    return (
        model_status,
        result.display_name,
        f"{result.confidence:.4f}",
        result.risk_level,
        result.recommendation_text,
        original_image,
        heatmap_image,
        overlay_image,
        report,
    )


with gr.Blocks(title="DermaReason Demo") as demo:
    gr.Markdown("# DermaReason Demo")
    gr.Markdown(
        "Upload a skin image to get disease prediction, clinical reasoning, recommendation, and Grad-CAM explainability."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Skin Image")

    run_button = gr.Button("Run Diagnosis", variant="primary")

    status_output = gr.Textbox(label="Model Status")

    with gr.Row():
        disease_output = gr.Textbox(label="Predicted Disease")
        confidence_output = gr.Textbox(label="Confidence")

    with gr.Row():
        risk_output = gr.Textbox(label="Risk Level")
        recommendation_output = gr.Textbox(label="Recommendation")

    with gr.Row():
        original_output = gr.Image(label="Original Image")
        heatmap_output = gr.Image(label="Heatmap")
        overlay_output = gr.Image(label="Overlay")

    report_output = gr.Textbox(label="Final Clinical Report", lines=18)

    run_button.click(
        fn=process_image,
        inputs=[image_input],
        outputs=[
            status_output,
            disease_output,
            confidence_output,
            risk_output,
            recommendation_output,
            original_output,
            heatmap_output,
            overlay_output,
            report_output,
        ],
    )


if __name__ == "__main__":
    port = _find_available_port()
    demo.launch(server_name="127.0.0.1", server_port=port, theme=gr.themes.Soft())

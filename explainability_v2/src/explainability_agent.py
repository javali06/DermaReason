"""
explainability_agent.py

Real Diagnostic + Explainability Agent for DermaReason.

IMPORTANT — CONTRACT COMPATIBILITY:
ExplainabilityAgent.predict(image_path) returns exactly the same
shape as mock.mock_model.MockDiagnosticModel.predict():

    {"disease": str, "confidence": float}

This is intentional. The Clinical Reasoning Agent, schemas.py, and
every test built against the mock model should be able to swap the
mock for this class with zero changes elsewhere in the pipeline.

.explain(image_path) is additive: same two keys, plus the Grad-CAM
heatmap/overlay paths, for callers that want the visual explanation
(report generator, UI) without breaking anyone relying on the base
predict() contract.

Preprocessing mirrors HAMDataset from the training notebook exactly:
BGR->RGB, scale to [0,1]. No ImageNet mean/std normalization is
applied, because none was applied at training time either.
"""

import os

import cv2
import numpy as np
import timm
import torch

from .class_mapping import CLASS_MAPPING
from .gradcam import GradCAM

IMG_SIZE = 224


class ExplainabilityAgent:

    def __init__(self, weights_path, device="cpu", output_dir="sample_outputs"):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=7,
        )
        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    # -- internal helpers -----------------------------------------------

    def _preprocess(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        original_rgb = image.copy()

        image = image.astype("float32") / 255.0
        tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # (1,3,224,224)

        return tensor, original_rgb

    @staticmethod
    def _make_heatmap(cam):
        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _make_overlay(original_rgb, heatmap, alpha=0.4):
        return cv2.addWeighted(heatmap, alpha, original_rgb, 1 - alpha, 0)

    # -- public API -------------------------------------------------------

    def predict(self, image_path):
        """
        Matches MockDiagnosticModel.predict() exactly:
            {"disease": str, "confidence": float}
        """
        tensor, _ = self._preprocess(image_path)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        return {
            "disease": CLASS_MAPPING[pred_idx],
            "confidence": round(confidence, 4),
        }

    def explain(self, image_path):
        """
        Same two keys as predict(), extended with Grad-CAM outputs:
            {
                "disease": str,
                "confidence": float,
                "original_path": str,
                "heatmap_path": str,
                "overlay_path": str,
            }
        """
        tensor, original_rgb = self._preprocess(image_path)
        tensor = tensor.to(self.device)

        target_layer = self.model.conv_head  # last conv block before pooling

        cam_engine = GradCAM(self.model, target_layer)
        cam, logits, pred_idx = cam_engine.generate(tensor)
        cam_engine.remove_hooks()

        probs = torch.softmax(logits, dim=1)
        confidence = float(probs[0, pred_idx].item())
        disease = CLASS_MAPPING[pred_idx]

        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = self._make_heatmap(cam_resized)
        overlay = self._make_overlay(original_rgb, heatmap)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_path = os.path.join(self.output_dir, f"{base_name}_original.png")
        heatmap_path = os.path.join(self.output_dir, f"{base_name}_heatmap.png")
        overlay_path = os.path.join(self.output_dir, f"{base_name}_overlay.png")

        cv2.imwrite(original_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return {
            "disease": disease,
            "confidence": round(confidence, 4),
            "original_path": original_path,
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
        }

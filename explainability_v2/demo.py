"""
Runnable demo for the Explainability Agent.

Usage:
    python demo.py <weights.pth> <image1> [image2 ...]

Prints predict() (the mock-model-compatible contract) and explain()
(the same contract + Grad-CAM paths) for each image.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from explainability_agent import ExplainabilityAgent


def main():
    if len(sys.argv) < 3:
        print("Usage: python demo.py <weights.pth> <image1> [image2 ...]")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_paths = sys.argv[2:]

    agent = ExplainabilityAgent(weights_path, device="cpu", output_dir="sample_outputs")

    for image_path in image_paths:
        print(f"\n=== {image_path} ===")
        print("predict():", json.dumps(agent.predict(image_path)))
        print("explain():", json.dumps(agent.explain(image_path), indent=2))


if __name__ == "__main__":
    main()

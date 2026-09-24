import sys

# Add project roots
sys.path.insert(0, "clinical_reasoning")
sys.path.insert(0, "explainability_v2")

# Clinical reasoning imports
from clinical_reasoning.src.clinical_reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning.src.report_generator import ReportGenerator

# Explainability imports
from explainability_v2.src.explainability_agent import ExplainabilityAgent


IMAGE_PATH = r"C:\Projects\DermaReason\dataset\raw\images\ISIC_0024306.jpg"

MODEL_PATH = r"C:\Projects\DermaReason\best_efficientnet_b0.pth"


print("=" * 50)
print("LOADING MODEL")
print("=" * 50)

explainer = ExplainabilityAgent(
    MODEL_PATH,
    device="cpu"
)

print("Model Loaded")


print("\n" + "=" * 50)
print("STEP 1: PREDICTION")
print("=" * 50)

prediction = explainer.predict(IMAGE_PATH)

print(prediction)


print("\n" + "=" * 50)
print("STEP 2: CLINICAL REASONING")
print("=" * 50)

reasoning_agent = ClinicalReasoningAgent()

result = reasoning_agent.analyze(
    prediction["disease"],
    prediction["confidence"],
    []
)


print("Risk Level:", result.risk_level)
print("Recommendation:", result.recommendation_category)


print("\n" + "=" * 50)
print("STEP 3: REPORT")
print("=" * 50)

report = ReportGenerator.generate(result)

print(report)


print("\n" + "=" * 50)
print("STEP 4: GRADCAM")
print("=" * 50)

explanation = explainer.explain(IMAGE_PATH)

print(explanation)

print("\nPipeline Complete")
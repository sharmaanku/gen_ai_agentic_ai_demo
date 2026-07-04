"""
===========================================================
LLM OBSERVABILITY USING MLFLOW
===========================================================

This demo tracks everything that a production AI application
usually monitors.

Metrics Logged
--------------
✔ Prompt Version
✔ Model Version
✔ Temperature
✔ Latency
✔ Token Usage
✔ Estimated Cost
✔ Quality Score
✔ Hallucination Detection
✔ Prompt
✔ Response

Run:
    python observability_demo.py

Open Dashboard:
    mlflow ui

Browser:
    http://127.0.0.1:5000

===========================================================
"""
import time
import random
import mlflow
from openai import OpenAI

# ---------------------------------------------------
# OpenAI Client
# ---------------------------------------------------

client = OpenAI(
    api_key="")
print("\n===== GOLD STANDARD EVALUATION DEMO ====="
)

# ---------------------------------------------------
# MLflow Experiment
# ---------------------------------------------------

mlflow.set_experiment("LLM_Observability_Demo")
# ---------------------------------------------------
# MLflow Experiment
# ---------------------------------------------------

mlflow.set_experiment("LLM_Observability_Demo")

# ---------------------------------------------------
# Sample Input
# ---------------------------------------------------

resume = """
John has 6 years of experience.

Skills:
Python
SQL
Azure
Docker
Kubernetes
Git
Terraform

Soft Skills:
Leadership
Communication
"""

# ---------------------------------------------------
# Prompt Versions
# ---------------------------------------------------

PROMPTS = {

    "v1":
"""
Extract skills from the resume.
""",

    "v2":
"""
Extract only technical skills.

Return a Python List.
""",

    "v3":
"""
You are an HR Recruiter.

Extract only technical skills.

Ignore soft skills.

Return JSON.

Example

{
   "skills":[]
}
"""
}

# ---------------------------------------------------
# Model Information
# ---------------------------------------------------

MODEL = "gpt-4.1-mini"

TEMPERATURE = 0.2

# Approximate cost per 1K tokens (example values only)
INPUT_COST = 0.0005
OUTPUT_COST = 0.0015

# ---------------------------------------------------
# Run Every Prompt Version
# ---------------------------------------------------

for version, prompt in PROMPTS.items():

    print("=" * 70)
    print("Running Prompt Version:", version)

    start = time.time()

    response = client.responses.create(
        model=MODEL,
        temperature=TEMPERATURE,
        input=f"{prompt}\n\nResume:\n{resume}"
    )

    latency = time.time() - start

    answer = response.output_text

    # ---------------------------------------------------
    # Token Usage
    # ---------------------------------------------------

    usage = response.usage

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    total_tokens = usage.total_tokens

    # ---------------------------------------------------
    # Estimated Cost
    # ---------------------------------------------------

    estimated_cost = (
        (input_tokens / 1000) * INPUT_COST +
        (output_tokens / 1000) * OUTPUT_COST
    )

    # ---------------------------------------------------
    # Dummy Quality Score
    # Normally calculated using evaluation framework
    # ---------------------------------------------------

    quality_score = random.randint(85, 99)

    # ---------------------------------------------------
    # Dummy Hallucination Detection
    # 0 = No
    # 1 = Yes
    # ---------------------------------------------------

    hallucination = 0

    if "Java" in answer and "Java" not in resume:
        hallucination = 1

    # ---------------------------------------------------
    # MLflow Logging
    # ---------------------------------------------------

    with mlflow.start_run(run_name=version):

        # ---------------- Parameters ----------------

        mlflow.log_param("Prompt Version", version)
        mlflow.log_param("Model", MODEL)
        mlflow.log_param("Temperature", TEMPERATURE)

        # ---------------- Metrics ----------------

        mlflow.log_metric("Latency (sec)", latency)
        mlflow.log_metric("Input Tokens", input_tokens)
        mlflow.log_metric("Output Tokens", output_tokens)
        mlflow.log_metric("Total Tokens", total_tokens)
        mlflow.log_metric("Estimated Cost ($)", estimated_cost)
        mlflow.log_metric("Quality Score", quality_score)
        mlflow.log_metric("Hallucination", hallucination)

        # ---------------- Artifacts ----------------

        mlflow.log_text(prompt, "Prompt.txt")
        mlflow.log_text(answer, "Response.txt")

    # ---------------------------------------------------
    # Console Output
    # ---------------------------------------------------

    print("\nModel :", MODEL)
    print("Latency :", round(latency, 2), "seconds")
    print("Input Tokens :", input_tokens)
    print("Output Tokens :", output_tokens)
    print("Total Tokens :", total_tokens)
    print("Estimated Cost : $", round(estimated_cost, 6))
    print("Quality Score :", quality_score)
    print("Hallucination :", hallucination)

    print("\nGenerated Response\n")
    print(answer)

print("\nAll prompt versions have been logged to MLflow.")


"""
gold_standard_evaluation_demo.py

Simple Trainer Demo:
1. What is a Gold Standard?
2. Generation Evaluation
3. Similarity Score
4. GPT Evaluation
5. Classification Evaluation
6. Accuracy
7. Confusion Matrix
8. Classification Report

Install:
pip install openai scikit-learn python-dotenv
"""

import os
from difflib import SequenceMatcher
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from openai import OpenAI

# -------------------------------------------------------
# STEP 1 : Load OpenAI API Key
# -------------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("\n===== GOLD STANDARD EVALUATION DEMO =====")

# =======================================================
# PART 1 : GENERATION EVALUATION
# =======================================================

print("\nPART 1 : GENERATION EVALUATION")

# Resume used as input
resume = """
John Smith
Experience: 5 Years

Skills:
Python
SQL
Machine Learning
Azure

Worked on predictive analytics projects.
"""

# Gold Standard = Human expert answer
gold_standard = """
John Smith is a Data Scientist with 5 years of experience in Python, SQL,
Azure and Machine Learning. He has worked on predictive analytics projects.
"""

print("\nGold Standard (Human Expert)")
print(gold_standard)

# Ask GPT to generate a summary
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
Generate a professional resume summary.

Resume:
{resume}

Return only the summary.
"""
        }
    ]
)

ai_summary = response.choices[0].message.content.strip()

print("\nAI Generated Summary")
print(ai_summary)

# Compare Gold Standard with AI summary
similarity = SequenceMatcher(None, gold_standard, ai_summary).ratio()

print("\nSimilarity Score")
print(round(similarity * 100, 2), "%")

# Ask GPT to evaluate itself against the Gold Standard
evaluation = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
Compare these two summaries.

Gold Standard:
{gold_standard}

AI Summary:
{ai_summary}

Evaluate on:
1. Accuracy (/10)
2. Professionalism (/10)
3. Missing Information

Keep the answer short.
"""
        }
    ]
)

print("\nGPT Evaluation")
print(evaluation.choices[0].message.content)

# =======================================================
# PART 2 : CLASSIFICATION EVALUATION
# =======================================================

print("\n=================================================")
print("PART 2 : CLASSIFICATION EVALUATION")
print("=================================================")

# Gold Standard dataset
tickets = [
    ("Payment failed during checkout", "Billing"),
    ("Resume alignment is incorrect", "Resume"),
    ("Application fee deducted twice", "Billing"),
    ("Resume font size is too small", "Resume"),
    ("Unable to complete payment", "Billing"),
    ("Resume spacing issue", "Resume"),
]

actual = []
predicted = []

for text, label in tickets:

    actual.append(label)

    result = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the support ticket.

Categories:
Billing
Resume

Ticket:
{text}

Return only one word.
"""
            }
        ]
    )

    prediction = result.choices[0].message.content.strip()

    predicted.append(prediction)

print("\nActual Labels")
print(actual)

print("\nPredicted Labels")
print(predicted)

print("\nAccuracy")
print(accuracy_score(actual, predicted))

print("\nConfusion Matrix")
print(confusion_matrix(actual, predicted))

print("\nClassification Report")
print(classification_report(actual, predicted))

print("""
================ TRAINER SUMMARY ================

Gold Standard
-------------
Human-created correct answer or correct label.

Generation
----------
AI creates text.
Compare AI text with Gold Standard.

Classification
--------------
AI predicts a category.
Compare predicted labels with Gold Standard labels.

Similarity Score
----------------
Measures how similar AI output is to the Gold Standard.

Accuracy
--------
Percentage of correct predictions.

Confusion Matrix
----------------
Shows where AI predicted correctly and where it made mistakes.
""")

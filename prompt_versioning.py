"""
Prompt Versioning Demo
----------------------
Save multiple prompt versions and compare their outputs.

Install:
pip install openai
"""

from openai import OpenAI

client = OpenAI()

# Same input for all prompts
resume = """
John worked on Python, SQL, Azure,
Docker, Kubernetes and GitHub.
"""

# Multiple Prompt Versions
prompts = {
    "v1": "Extract skills from the resume.",

    "v2": """
Extract only technical skills.
Return Python list.
""",

    "v3": """
You are an HR recruiter.

Extract only technical skills.

Ignore soft skills.

Return JSON.

Example:
{
   "skills":[]
}
"""
}

results = []

# Test every prompt version
for version, prompt in prompts.items():
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"{prompt}\n\nResume:\n{resume}"
    )

    output = response.output_text

    results.append({
        "Version": version,
        "Prompt": prompt,
        "Output": output
    })

# Print comparison
print("=" * 70)

for item in results:
    print(f"Prompt Version : {item['Version']}")
    print("-" * 70)
    print(item["Output"])
    print("=" * 70)

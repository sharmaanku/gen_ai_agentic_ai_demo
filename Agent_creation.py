Background
BOLD is an online career platform that helps job seekers create resumes, search for jobs, and improve their chances of getting hired.
Every day, thousands of users upload resumes and ask different career-related questions. Currently, each request is handled by separate services, resulting in multiple manual steps and slower response times.
BOLD wants to build a single AI Agent that can understand a user's request, decide which internal services (tools) are required, execute them in the correct order, and provide a single, well-structured response.


Problem Statement
As an AI Developer, your task is to build a lightweight AI Agent that can autonomously decide which tool(s) to use based on the user's request.
The agent should not follow a fixed sequence of steps. Instead, it should analyze the user's intent, plan the required actions, invoke the appropriate tools, observe their outputs, and continue until the user's request is fully satisfied.


Available Internal Tools
The AI Agent can use the following internal tools:
•	Job Search Tool – Finds suitable jobs based on skills.
•	ATS Resume Checker – Calculates the resume's ATS compatibility score.
•	Resume Formatter – Formats the resume into a professional layout.
•	Skill Recommendation Tool – Suggests missing skills based on the desired job role.
•	Resume Summary Generator – Generates a professional profile summary.


Sample User Requests
•	"Find Python jobs for me."
•	"Check my resume's ATS score."
•	"Format my resume and suggest missing skills."
•	"Find Data Engineer jobs, check my ATS score, and generate a professional summary."


Developer Objectives
As a developer, you need to:
•	Design an AI Agent that understands user intent.
•	Build reusable Python functions for each business capability.
•	Expose these functions as tools to the LLM.
•	Implement a ReAct (Thought → Action → Observation) workflow so the agent can decide which tool to call next.
•	Optionally orchestrate the workflow using LangGraph instead of manually managing the execution logic.
•	Generate a final consolidated response after all required tools have been executed.


Expected Outcome
The completed solution should demonstrate that the AI Agent can:
•	Understand natural language requests.
•	Choose the correct tool or combination of tools.
•	Execute tools autonomously.
•	Chain multiple tool calls when required.
•	Produce a single, user-friendly final response.


# AI Resume Screener: OpenEnv Environment

## Overview
The **AI Resume Screener** is a high-fidelity recruitment automation environment built on the [OpenEnv](https://github.com/openenv/openenv) specification. It simulates the real-world task of a technical recruiter screening candidate resumes against complex Job Descriptions (JDs). 

Unlike toy environments, this benchmark uses the **NetSol Technical Recruitment Dataset**, featuring real-world resumes and technical JDs. Agents must perform structured reasoning across micro-skills and macro-experience requirements to make accurate hiring decisions.

## Motivation
Manual resume screening is a high-volume, high-stakes bottleneck in recruitment. Evaluating LLM agents on this task requires:
1.  **Multi-step Reasoning**: Moving beyond keyword matching to weighted criteria analysis.
2.  **Adversarial Robustness**: Identifying "junk" or "prompt-injection" styled resumes.
3.  **Strict Compliance**: Adhering to organizational thresholds for Shortlisting and Rejection.

## Environment Specification

### Observation Space (`ResumeObservation`)
The agent receives a rich state representing the recruiter's desk:
- `candidate_id`: A unique trace for the current candidate.
- `resume_text`: The full, raw text extracted from the candidate's CV.
- `job_title` & `job_description`: The formal requirements for the role.
- `macro_criteria`: JSON-weighted strategic requirements (e.g., "7+ years of Java").
- `micro_criteria`: JSON-weighted tactical skill requirements (e.g., "Experience with Spring Boot").

### Action Space (`ResumeAction`)
The agent must provide a structured response:
- **`decision`**: One of `shortlist`, `reject`, or `flag_for_review`.
- **`reasoning`**: A justification (Chain-of-Thought) for the chosen decision.

## Task Definitions
The environment provides three distinct difficulty levels:

| Task | Description | Goal |
| :--- | :--- | :--- |
| **Easy** | Binary Screening | Clear-cut cases where candidates are either 100% matches or completely unrelated. |
| **Medium** | Full Screening | Realistic mix including borderline cases (~60-70% match) requiring nuanced judgment. |
| **Hard** | Adversarial | Includes "junk" resumes, gibberish text, and candidates who try to game the criteria. |

## Reward Function
The environment uses a programmatic grader that evaluates accuracy against GPT-4 verified ground-truth:
- **Correct Decision**: +1.0 reward.
- **Incorrect Decision**: 0.0 reward.
- **Adversarial Failure**: -0.5 penalty (only in `hard` mode) if the agent shortlists a gibberish resume.

## Getting Started

### Prerequisites
- Python 3.10+
- Docker (for containerized execution)
- OpenEnv Core: `pip install openenv-core`

### Configuration
Create a `.env` file in the root directory:
```bash
HF_TOKEN=your_huggingface_token
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

### Local Development
To run the server locally for development:
```bash
uv run --project . server --port 8000
```

### Running Inference
To run the baseline evaluation agent:
```bash
python inference.py
```

## Deployment
This environment is designed to be deployed as a containerized Hugging Face Space.
```bash
openenv push
```

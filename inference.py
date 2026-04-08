"""
Inference Script — AI Resume Screener (resume_env)
===================================================

MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM (default: HuggingFace router)
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your HuggingFace / API key
    IMAGE_NAME     Docker image name for the environment (from_docker_image)

STDOUT FORMAT (strictly followed for automated evaluation):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] per episode.
    - One [STEP] per step, immediately after env.step() returns.
    - One [END] after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string, or null if none.
    - All fields on one line, no newlines within a line.
    - Each task returns score in [0, 1].

Example:
    [START] task=easy env=resume_env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=shortlist reward=1.00 done=false error=null
    [STEP] step=2 action=reject reward=0.00 done=false error=null
    [END] success=true steps=2 score=0.50 rewards=1.00,0.00
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

# Load .env file automatically when running locally.
# In Docker / HF Space the vars come from the container environment instead.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed — rely on shell environment vars

from openai import OpenAI

# Direct imports — inference.py lives inside resume_env/, so we import siblings directly.
# (When run as a standalone script, Python's cwd is resume_env/ itself)
try:
    from resume_env import ResumeAction, ResumeEnv   # from parent dir (e.g. during docker eval)
except ModuleNotFoundError:
    from models import ResumeAction                   # local sibling (running: python inference.py)
    from client import ResumeEnv

# --- MANDATORY CONFIGURATION ---
# --- MANDATORY CONFIGURATION ---
IMAGE_NAME   = os.getenv("IMAGE_NAME", "resume_env")
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "ollama"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1" # Let it be None if not set, we'll handle fallback below
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK    = "resume_env"

# --- EPISODE SETTINGS ---
MAX_STEPS = 5
TEMPERATURE = 0.3  
MAX_TOKENS  = 1000 

# Max possible reward per episode (1.0 per correct step)
MAX_TOTAL_REWARD = MAX_STEPS * 1.0

# Score threshold to consider an episode successful
SUCCESS_SCORE_THRESHOLD = 0.4

# Tasks to run — easy, medium, hard
TASKS = ["easy", "medium", "hard"]

# --- RESUME TEXT CLEANER ---
def clean_resume_text(text: str) -> str:
    """Replace common OCR ligatures and artifacts that confuse LLMs."""
    if not text:
        return ""
    replacements = {
        '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl',
        '\u2013': '-', '\u2014': '-', '\u2022': '*', '\u27a2': '*',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = textwrap.dedent("""
    You are a Senior HR Expert and AI Resume Screening Specialist.

    === KNOWLEDGE BASE (critical rules — always apply these) ===
    - "MS Office" proficiency ALWAYS includes MS Word, MS Excel, and MS PowerPoint.
      If you see 'MS Office', you MUST score any micro-criteria for Word/Excel/PowerPoint as a match.
    - Be extremely careful with abbreviations (e.g., 'BD' = 'Business Development',
      'PM' = 'Project Management', 'BI' = 'Business Intelligence').
    - Gibberish resumes (e.g. "blorph snizzle wackadoo") must be immediately REJECTED.

    === YOUR 5-STEP EVALUATION PROCESS ===
    1. JUNK CHECK:      Is this a real professional resume? If not, REJECT immediately.
    2. MICRO SCORING:   Rate each DETAILED skill criteria (0-10). Apply the MS Office rule.
    3. MACRO SCORING:   Rate each HIGH-LEVEL criteria (0-10) based on micro scores.
    4. WEIGHTED MATH:   Calculate: sum(MacroScore × Weight) / 100 = weighted_average.
    5. DECISION:
       - "shortlist"       if weighted_average > 6.5  (strong match)
       - "flag_for_review" if weighted_average 4.0-6.5 (borderline)
       - "reject"          if weighted_average < 4.0  (poor match or junk)

    RESPOND ONLY with valid JSON — no markdown, no extra text:
    {
      "analysis": {
        "macro_scores": {"criteria_name": score, ...},
        "weighted_average": total
      },
      "decision": "shortlist|reject|flag_for_review",
      "reasoning": "Step-by-step: [Micro] -> [Macro] -> [Math] -> [Decision]"
    }
""").strip()


# --- STDOUT LOGGING (exact format required by OpenEnv evaluation) ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- PROMPT BUILDER ---

def build_user_prompt(obs) -> str:
    """Build the per-step screening prompt from a ResumeObservation."""
    try:
        macro = json.loads(obs.macro_criteria or "{}")
        micro = json.loads(obs.micro_criteria or "{}")
    except Exception:
        macro, micro = {}, {}

    macro_rubric = "\n".join(f"  - {k} (weight: {v}%)" for k, v in macro.items()) or "  - Not specified"
    micro_rubric = "\n".join(f"  - {k} (weight: {v}%)" for k, v in micro.items()) or "  - Not specified"

    cleaned_resume = clean_resume_text(obs.resume_text or "")

    return textwrap.dedent(f"""
        === JOB DESCRIPTION ===
        {obs.job_description}

        === HIGH-LEVEL SCORING CRITERIA (macro) ===
        {macro_rubric}

        === DETAILED SKILL CRITERIA (micro) ===
        {micro_rubric}

        === CANDIDATE RESUME ===
        {cleaned_resume}

        Follow the 5-step process. Respond with JSON only.
    """).strip()


# --- LLM CALL ---

def get_screening_decision(
    client: OpenAI,
    obs,
    history: List[str],
) -> ResumeAction:
    """Call the LLM and parse its output into a ResumeAction."""
    user_prompt = build_user_prompt(obs)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Robustly extract JSON — strip markdown fences first
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        # Use regex to find first complete JSON object
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in model response")
        parsed = json.loads(json_match.group())
        return ResumeAction(
            decision=parsed.get("decision", "flag_for_review"),
            reasoning=parsed.get("reasoning", "No reasoning provided."),
        )
    except Exception as exc:
        return ResumeAction(
            decision="flag_for_review",
            reasoning=f"Parse error: {exc}",
        )


# --- TESTING SETTINGS ---
# Set to True to test against your locally running server (on port 8000).
# Set to False for the final Docker/OpenEnv submission validation.
USE_LOCAL_SERVER = False
LOCAL_SERVER_URL = "http://localhost:8000"

async def run_episode(client: OpenAI, task: str) -> None:
    """Run one full episode for a given task difficulty."""
    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    try:
        if USE_LOCAL_SERVER:
            env = ResumeEnv(base_url=LOCAL_SERVER_URL)
        else:
            env = await ResumeEnv.from_docker_image(IMAGE_NAME)

        result = await env.reset(task=task)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_screening_decision(client, obs, history)

            result = await env.step(action)
            obs    = result.observation

            reward = result.reward or 0.0
            done   = result.done
            error  = None

            rewards.append(reward)
            steps_taken = step

            history.append(
                f"Step {step}: candidate={obs.candidate_id} decision={action.decision.value} reward={reward:+.2f}"
            )

            log_step(
                step=step,
                action=action.decision.value,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)       # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        log_step(
            step=steps_taken + 1,
            action="flag_for_review",
            reward=0.0,
            done=True,
            error=str(exc),
        )

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# --- MAIN ---

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        await run_episode(client, task)


if __name__ == "__main__":
    asyncio.run(main())

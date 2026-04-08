"""
Data Loader for the Resume Env Environment.

Loads resume screening scenarios from the netsol/resume-score-details dataset,
stored as individual JSON files in the netsol_raw/ directory.

File naming convention:
  - match_*.json               : Resume genuinely fits the JD → high score → shortlist
  - mismatch_*.json            : Resume intentionally wrong JD → low score → reject
  - invalid_*.json             : Gibberish/fake resume → automatic reject
  - empty_additional_info_*.json : Valid match but no extra hiring context

Each file structure:
  input.resume                 : Full resume text
  input.job_description        : Full job description
  input.minimum_requirements   : List of hard requirements
  input.additional_info        : Extra recruiter context
  input.macro_dict             : High-level weighted criteria (e.g. {"experience": 70})
  input.micro_dict             : Detailed skill criteria (e.g. {"python": 30})
  output.valid_resume_and_jd   : False for gibberish files
  output.scores.aggregated_scores.macro_scores : GPT-4 computed score out of 10
"""

import glob
import json
import os
import random


def load_data(limit: int = 5, split: str = "train", task: str = None) -> list:
    """
    Loads screening scenarios from netsol_raw/*.json files.

    Args:
        limit: Maximum number of scenarios to return per episode.
        split: Dataset split (unused but kept for OpenEnv interface compatibility).
        task:  Difficulty filter: "easy", "medium", "hard", or None for all.

    Returns:
        List of scenario dicts ready for the environment queue.
    """
    print(f"Loading randomized dataset from netsol_raw/ (limit={limit}, task={task})...")

    local_dir = os.path.join(os.path.dirname(__file__), "netsol_raw")
    files = glob.glob(os.path.join(local_dir, "*.json"))

    if not files:
        print("WARNING: netsol_raw/ directory not found or empty.")
        return []

    random.shuffle(files)  # Different candidate order every episode

    scenarios = []

    for filepath in files:
        if len(scenarios) >= limit:
            break

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                row = json.load(f)
        except Exception:
            continue

        input_data = row.get("input", {})
        output_data = row.get("output", {})

        # --- Handle invalid/gibberish files as automatic Reject test cases ---
        is_valid = output_data.get("valid_resume_and_jd", True)

        if not is_valid:
            score = 0.0
            expected_decision = "reject"
            rationale = "Candidate submission contains invalid or nonsensical text (gibberish/spam)."
        else:
            scores = output_data.get("scores", {})
            aggregated = scores.get("aggregated_scores", {})
            try:
                score = aggregated.get("macro_scores", 0.0) / 10.0
            except Exception:
                score = 0.0

            # --- Task-based filtering ---
            if task == "easy":
                # Easy mode: only very strong matches or very clear rejections
                if 0.3 < score < 0.7:
                    continue
            elif task == "hard":
                # Hard mode: include adversarial/junk cases
                # (already covered by is_valid logic above)
                pass
            # "medium" task (the default) includes the full spectrum

            # --- Map score to expected decision ---
            if score > 0.65:
                expected_decision = "shortlist"
                rationale = f"High compatibility (Score: {score * 10:.1f}/10.0)."
            elif score > 0.40:
                expected_decision = "flag_for_review"
                rationale = f"Partial match (Score: {score * 10:.1f}/10.0) requiring manual review."
            else:
                expected_decision = "reject"
                rationale = f"Low factor compatibility (Score: {score * 10:.1f}/10.0)."

        # --- Build job description with all available context ---
        try:
            jd_text = input_data.get("job_description", "N/A")
            job_title = jd_text.split("\n")[0][:60] + "..." if "\n" in jd_text else "Target Role"
        except Exception:
            jd_text = "N/A"
            job_title = "Target Role"

        min_reqs = input_data.get("minimum_requirements", [])
        if min_reqs:
            jd_text += "\n\nMinimum Requirements:\n- " + "\n- ".join(min_reqs)

        add_info = input_data.get("additional_info", "")
        if add_info:
            jd_text += f"\n\nAdditional Info:\n{add_info}"

        macro_dict = input_data.get("macro_dict", {})
        micro_dict = input_data.get("micro_dict", {})

        # --- Build enriched, structured resume from the details block ---
        # The 'details' block has been pre-parsed by GPT-4, giving clean structured fields.
        # We build a labeled text block the AI can reason against instead of raw OCR.
        raw_resume = input_data.get("resume", "N/A")
        details = row.get("details", {})

        enriched_parts = []

        if details.get("name"):
            enriched_parts.append(f"Name: {details['name']}")
        if details.get("email_id"):
            enriched_parts.append(f"Email: {details['email_id']}")
        if details.get("location"):
            enriched_parts.append(f"Location: {details['location']}")

        if details.get("executive_summary"):
            enriched_parts.append(f"\nSUMMARY:\n{details['executive_summary']}")

        if details.get("employment_history"):
            enriched_parts.append("\nEXPERIENCE:")
            for job in details["employment_history"]:
                title = job.get("job_title", "")
                company = job.get("company_name", "")
                start = job.get("start_date", "")
                end = job.get("end_date", "Present")
                job_details = job.get("details", "")
                enriched_parts.append(f"  - {title} at {company} ({start} - {end}): {job_details}")

        if details.get("education"):
            enriched_parts.append("\nEDUCATION:")
            for edu in details["education"]:
                enriched_parts.append(
                    f"  - {edu.get('degree_title', '')} from {edu.get('university', '')} (Graduated: {edu.get('end_date', '')})"
                )

        if details.get("skills"):
            skill_items = details["skills"]
            if skill_items and isinstance(skill_items[0], dict):
                skills_str = ", ".join(s.get("skill", s.get("name", str(s))) for s in skill_items)
            else:
                skills_str = ", ".join(str(s) for s in skill_items)
            enriched_parts.append(f"\nSKILLS: {skills_str}")

        if details.get("certifications"):
            cert_items = details["certifications"]
            if cert_items and isinstance(cert_items[0], dict):
                certs_str = ", ".join(c.get("certification_name", c.get("name", str(c))) for c in cert_items)
            else:
                certs_str = ", ".join(str(c) for c in cert_items)
            enriched_parts.append(f"CERTIFICATIONS: {certs_str}")

        # Always append raw OCR text as additional context fallback
        if raw_resume and raw_resume != "N/A":
            enriched_parts.append(f"\n--- Original Resume Text ---\n{raw_resume}")

        resume_text = "\n".join(enriched_parts) if enriched_parts else raw_resume

        scenarios.append({
            "id": os.path.basename(filepath),
            "difficulty": task or "mixed",
            "job_title": job_title,
            "job_description": jd_text,
            "resume_text": resume_text,
            "macro_criteria": json.dumps(macro_dict),
            "micro_criteria": json.dumps(micro_dict),
            "expected_decision": expected_decision,
            "rationale": rationale,
        })

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios

"""
Graders for the Resume Env Environment.

Three distinct tasks with different objectives and grader behaviors:

  Task 1: binary_screen
    Objective: Make a decisive shortlist OR reject. No hedging allowed.
    Grader:    +1.0 correct, -0.5 for flag_for_review, 0.0 wrong.

  Task 2: full_screen
    Objective: Make the correct 3-way decision across the full distribution.
    Grader:    +1.0 correct, -0.1 for flag_for_review when wrong, 0.0 wrong.

  Task 3: adversarial_screen
    Objective: Screen real candidates AND detect gibberish/fake submissions.
    Grader:    +1.0 correct, -0.5 for not rejecting a gibberish resume,
               -0.1 flag wrong on real resumes, 0.0 other wrong.
"""


def _is_gibberish(scenario: dict) -> bool:
    """Check if this scenario is a gibberish/fake submission."""
    return "gibberish" in scenario.get("rationale", "").lower() or \
           "nonsensical" in scenario.get("rationale", "").lower() or \
           "invalid" in scenario.get("rationale", "").lower()


# -------------------------------------------------------------------
# Task 1: Binary Screening (Easy)
# Agent must commit to shortlist OR reject — flag_for_review is penalized
# -------------------------------------------------------------------
def grade_binary_screen(scenario: dict, candidate_decision: str, candidate_reasoning: str) -> dict:
    """
    Binary screening grader.
    Objective: Confidently shortlist or reject. No hedging.
    """
    expected = scenario["expected_decision"].lower().strip()
    agent    = candidate_decision.lower().strip()

    # Normalize expected: if GPT-4 said flag_for_review, we accept either shortlist or reject
    # based on which side of the threshold the score falls
    if expected == "flag_for_review":
        if agent == "flag_for_review":
            reward   = 0.1  # Low reward instead of negative penalty
            feedback = f"Hedged with flag_for_review. Commit to a decision in binary_screen."
        else:
            reward   = 0.5  # Partial credit
            feedback = f"Acceptable. Borderline case — committed to '{agent}'."
        return {"reward": float(reward), "is_correct": False, "feedback": feedback}

    is_correct = (agent == expected)

    if is_correct:
        reward   = 0.95  # Slightly less than 1.0
        feedback = f"Correct identifying {expected}. Ground Truth: {scenario['rationale']}"
    elif agent == "flag_for_review":
        reward   = 0.1  # Low reward for hedging
        feedback = f"Hedged with flag_for_review in binary_screen. Ground Truth: {scenario['rationale']}"
    else:
        reward   = 0.05  # Slightly more than 0.0
        feedback = f"Incorrect. Expected '{expected}', chose '{agent}'. Ground Truth: {scenario['rationale']}"

    return {"reward": float(reward), "is_correct": is_correct, "feedback": feedback}


# -------------------------------------------------------------------
# Task 2: Full Screening (Medium)
# Agent makes the correct 3-way decision across full candidate distribution
# -------------------------------------------------------------------
def grade_full_screen(scenario: dict, candidate_decision: str, candidate_reasoning: str) -> dict:
    """
    Full 3-way screening grader.
    Objective: Match GPT-4's decision (shortlist / flag_for_review / reject).
    """
    expected = scenario["expected_decision"].lower().strip()
    agent    = candidate_decision.lower().strip()

    is_correct = (agent == expected)

    if is_correct:
        reward   = 0.95
        feedback = f"Correct decision: {expected}. Ground Truth: {scenario['rationale']}"
    elif agent == "flag_for_review":
        reward   = 0.1
        feedback = f"Incorrect. You hedged with flag_for_review. Ground Truth: {scenario['rationale']}"
    else:
        reward   = 0.05
        feedback = f"Incorrect decision. Expected '{expected}', chose '{agent}'. Ground Truth: {scenario['rationale']}"

    return {"reward": float(reward), "is_correct": is_correct, "feedback": feedback}


# -------------------------------------------------------------------
# Task 3: Adversarial Screening (Hard)
# Agent screens real candidates + must detect and reject gibberish/fake resumes
# Failing to reject a fake resume incurs a severe penalty
# -------------------------------------------------------------------
def grade_adversarial_screen(scenario: dict, candidate_decision: str, candidate_reasoning: str) -> dict:
    """
    Adversarial screening grader.
    Objective: Screen real candidates correctly AND detect fake/gibberish resumes.
    Not rejecting a gibberish resume is severely penalized (-0.5).
    """
    expected   = scenario["expected_decision"].lower().strip()
    agent      = candidate_decision.lower().strip()
    is_gibber  = _is_gibberish(scenario)

    is_correct = (agent == expected)

    if is_correct:
        reward   = 0.95
        feedback = f"Correct identifying {expected}. Ground Truth: {scenario['rationale']}"

    elif is_gibber and agent != "reject":
        reward   = 0.05
        feedback = f"Failed to detect gibberish/fake resume. Ground Truth: {scenario['rationale']}"

    elif agent == "flag_for_review":
        reward   = 0.1
        feedback = f"Incorrect hedging. Expected '{expected}'. Ground Truth: {scenario['rationale']}"

    else:
        reward   = 0.05
        feedback = f"Incorrect decision. Expected '{expected}', chose '{agent}'. Ground Truth: {scenario['rationale']}"

    return {"reward": float(reward), "is_correct": is_correct, "feedback": feedback}


# -------------------------------------------------------------------
# Router — selects the correct grader based on current task
# -------------------------------------------------------------------
def evaluate_action(
    scenario: dict,
    candidate_decision: str,
    candidate_reasoning: str,
    task: str = "full_screen",
) -> dict:
    """
    Routes to the correct grader based on the active task.
    Routes to the appropriate grader based on the mission difficulty.

    Args:
        scenario:            The current scenario dict from the queue.
        candidate_decision:  The agent's decision string.
        candidate_reasoning: The agent's reasoning text.
        task:                Active task difficulty. One of:
                               easy, medium, hard

    Returns:
        dict with keys: reward (float), is_correct (bool), feedback (str)
    """
    if task == "easy":
        return grade_binary_screen(scenario, candidate_decision, candidate_reasoning)
    elif task == "hard":
        return grade_adversarial_screen(scenario, candidate_decision, candidate_reasoning)
    else:  # Default: medium (full_screen)
        return grade_full_screen(scenario, candidate_decision, candidate_reasoning)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Resume Env Environment Implementation.

A real-world AI Resume Screener environment where an agent evaluates candidates
against job descriptions using GPT-4-verified scoring data from the
netsol/resume-score-details dataset.

The agent must decide: shortlist / flag_for_review / reject.
It is graded against GPT-4's own evaluation as ground truth.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ResumeAction, ResumeObservation
except (ImportError, ValueError, ModuleNotFoundError):
    from models import ResumeAction, ResumeObservation

from . import graders
from . import loader


class ResumeEnvironment(Environment):
    """
    AI Resume Screener Environment.

    The agent receives a candidate's resume and a job description with
    weighted scoring criteria (from GPT-4's grading rubric). It must
    classify each candidate as shortlist / flag_for_review / reject.

    Rewards:
        +1.0  for matching GPT-4's ground truth decision
        -0.1  for choosing flag_for_review when wrong (penalize hedging)
         0.0  for a confidently wrong answer

    Tasks:
        - easy:   Obvious matches and clear mismatches (score > 0.7 or < 0.35)
        - medium: Full distribution, no filtering
        - hard:   Borderline edge cases (score 0.20–0.85)

    Example:
        >>> env = ResumeEnvironment()
        >>> obs = env.reset(task="hard")
        >>> print(obs.candidate_id)
        >>> action = ResumeAction(decision="shortlist", reasoning="Strong match.")
        >>> obs = env.step(action)
        >>> print(obs.reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the resume screener environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.data_queue = []
        self._current_index = 0
        self._total_screened = 0
        self.max_steps = 100
        self.episode_rewards = []
        self.current_task = "medium"

    def _get_observation(self, done: bool = False, reward: float = 0.0, info: dict = None) -> ResumeObservation:
        """Build a ResumeObservation from the current queue position."""
        if info is None:
            info = {}

        metadata = {
            "step_count": self._state.step_count,
            "resumes_remaining": max(0, len(self.data_queue) - self._current_index),
            "total_screened": self._total_screened,
            "task": self.current_task,
        }
        metadata.update(info)

        # Queue exhausted — return empty observation
        if self._current_index >= len(self.data_queue):
            return ResumeObservation(
                candidate_id="none",
                resume_text="none",
                job_title="none",
                job_description="none",
                status=info.get("feedback", "waiting"),
                done=done,
                reward=reward,
                metadata=metadata,
            )

        scenario = self.data_queue[self._current_index]
        return ResumeObservation(
            candidate_id=scenario.get("id", "none"),
            resume_text=scenario.get("resume_text", "none"),
            job_title=scenario.get("job_title", "none"),
            job_description=scenario.get("job_description", "none"),
            macro_criteria=scenario.get("macro_criteria", "{}"),
            micro_criteria=scenario.get("micro_criteria", "{}"),
            status=info.get("feedback", "waiting" if self._state.step_count == 0 else "proceeding"),
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def reset(self, task: str = "medium", **kwargs) -> ResumeObservation:
        """
        Resets the environment for a new screening episode.

        Args:
            task: The difficulty level for the episode. One of:
                  "easy"   - Mostly clear-cut candidates.
                  "medium" - Full range of candidates.
                  "hard"   - Includes adversarial/gibberish cases.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = task
        self._current_index = 0
        self._total_screened = 0
        self.episode_rewards = []
        self.data_queue = loader.load_data(limit=100, task=task)
        return self._get_observation()

    def step(self, action: ResumeAction) -> ResumeObservation:  # type: ignore[override]
        """
        Execute a screening decision step.

        Args:
            action: ResumeAction with decision and reasoning.

        Returns:
            ResumeObservation with reward and feedback for the next candidate.
        """
        if self._current_index >= len(self.data_queue):
            return self._get_observation(done=True)

        self._state.step_count += 1
        current_scenario = self.data_queue[self._current_index]

        eval_result = graders.evaluate_action(
            scenario=current_scenario,
            candidate_decision=action.decision.value,
            candidate_reasoning=action.reasoning,
            task=self.current_task,
        )

        step_reward = eval_result["reward"]
        self.episode_rewards.append(step_reward)

        info = {
            "is_correct_decision": eval_result["is_correct"],
            "feedback": eval_result["feedback"],
        }

        self._current_index += 1
        self._total_screened += 1

        done = (
            self._current_index >= len(self.data_queue)
            or self._state.step_count >= self.max_steps
        )

        return self._get_observation(done=done, reward=step_reward, info=info)

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

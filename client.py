# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resume Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ResumeAction, ResumeObservation
except (ImportError, ValueError):
    from models import ResumeAction, ResumeObservation


class ResumeEnv(
    EnvClient[ResumeAction, ResumeObservation, State]
):
    """
    Client for the Resume Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with ResumeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.candidate_id)
        ...
        ...     action = ResumeAction(decision="shortlist", reasoning="Strong match.")
        ...     result = client.step(action)
        ...     print(result.observation.status)

    Example with Docker:
        >>> client = ResumeEnv.from_docker_image("resume_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     action = ResumeAction(decision="reject", reasoning="Missing required skills.")
        ...     result = client.step(action)
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ResumeAction) -> Dict:
        """
        Convert ResumeAction to JSON payload for step message.

        Args:
            action: ResumeAction instance with decision and reasoning.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "decision": action.decision.value,
            "reasoning": action.reasoning,
            "status": action.status,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ResumeObservation]:
        """
        Parse server response into StepResult[ResumeObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with ResumeObservation.
        """
        obs_data = payload.get("observation", {})
        observation = ResumeObservation(
            candidate_id=obs_data.get("candidate_id", "none"),
            resume_text=obs_data.get("resume_text", "none"),
            job_title=obs_data.get("job_title", "none"),
            job_description=obs_data.get("job_description", "none"),
            macro_criteria=obs_data.get("macro_criteria", "{}"),
            micro_criteria=obs_data.get("micro_criteria", "{}"),
            status=obs_data.get("status", "waiting"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request.

        Returns:
            State object with episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

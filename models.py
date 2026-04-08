# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Resume Env Environment.

The resume_env environment screens candidates against job descriptions
using GPT-4-verified resume scoring data from the netsol dataset.
"""

from enum import Enum
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


class ScreeningDecision(str, Enum):
    """Strictly controlled action space for the LLM screener."""
    SHORTLIST = "shortlist"
    REJECT = "reject"
    FLAG_FOR_REVIEW = "flag_for_review"


class ResumeAction(Action):
    """
    Action for the Resume Env environment.
    The agent must make a hiring decision and justify it.
    """
    decision: ScreeningDecision = Field(
        ...,
        description="The screening decision: shortlist, reject, or flag_for_review."
    )
    reasoning: str = Field(
        ...,
        description="A concise justification for the decision based on the resume vs. JD."
    )
    status: str = Field(
        default="active",
        description="Current screening status."
    )

    @field_validator("decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: str) -> str:
        """Map common synonyms to strict Enum values."""
        if not isinstance(v, str):
            return v
        mapping = {
            "rejected": "reject",
            "shortlisted": "shortlist",
            "flagged_for_review": "flag_for_review",
            "flag": "flag_for_review",
            "review": "flag_for_review",
        }
        val = v.lower().strip()
        return mapping.get(val, val)


class ResumeObservation(Observation):
    """
    Observation from the Resume Env environment.
    Contains the full resume, job description, and weighted scoring criteria
    so the agent can make a data-backed hiring decision.
    """
    # Candidate information
    candidate_id: str = Field(..., description="Unique identifier for the current resume.")
    resume_text: str = Field(..., description="The raw, full text of the candidate's resume.")

    # Job requirements
    job_title: str = Field(..., description="The title of the position being hired for.")
    job_description: str = Field(..., description="The full job description including minimum requirements.")

    # GPT-4 weighted scoring rubric (from netsol dataset)
    macro_criteria: str = Field(default="{}", description="JSON string of high-level criteria with weights, e.g. {\"experience\": 70}.")
    micro_criteria: str = Field(default="{}", description="JSON string of detailed skill criteria with weights, e.g. {\"python\": 30}.")

    # Feedback from last action
    status: str = Field(default="waiting", description="Feedback status from the previous step.")

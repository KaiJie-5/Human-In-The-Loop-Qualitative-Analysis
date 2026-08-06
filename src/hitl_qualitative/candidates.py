from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .categories import CATEGORY_BY_ID, CategoryId


class _StrictAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    @field_validator("*", mode="after")
    @classmethod
    def require_nonempty_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("Text fields must not be empty.")
        return value


class WrongCodeAssessment(_StrictAssessment):
    category_id: Literal["wrong_code"]
    why_plausible_for_wider_dataset: str
    why_unsupported_by_this_segment: str
    relation_to_research_questions: str
    category_boundary: str
    reflective_question: str


class DescriptiveAssessment(_StrictAssessment):
    category_id: Literal["descriptive_not_answering_rq"]
    surface_description: str
    why_true_of_segment: str
    why_not_useful_for_research_questions: str
    relation_to_research_questions: str
    category_boundary: str
    reflective_question: str


class TooBroadAssessment(_StrictAssessment):
    category_id: Literal["too_broad"]
    broad_relevance_to_research_questions: str
    specific_meaning_lost: str
    why_it_is_too_broad: str
    relation_to_research_questions: str
    category_boundary: str
    reflective_question: str


class UsefulAssessment(_StrictAssessment):
    category_id: Literal["useful_analytical_code"]
    specific_analytical_insight: str
    why_it_is_useful: str
    relation_to_research_questions: str
    category_boundary: str
    reflective_question: str


CandidateAssessment = Annotated[
    Union[
        WrongCodeAssessment,
        DescriptiveAssessment,
        TooBroadAssessment,
        UsefulAssessment,
    ],
    Field(discriminator="category_id"),
]
CANDIDATE_ADAPTER = TypeAdapter(CandidateAssessment)

_OPEN_QUESTION = re.compile(
    r"^(?:what|why|how|which|where|when|who|in\s+what\s+ways|to\s+what\s+extent)\b",
    re.IGNORECASE,
)


def candidate_json_schema() -> dict[str, Any]:
    return CANDIDATE_ADAPTER.json_schema()


def parse_candidate(raw_content: str) -> CandidateAssessment:
    return CANDIDATE_ADAPTER.validate_json(raw_content)


def validate_candidate(
    candidate: CandidateAssessment,
) -> tuple[CandidateAssessment | None, list[str]]:
    errors: list[str] = []
    payload = candidate.model_dump()
    question = str(payload["reflective_question"])
    if question.count("?") != 1 or not question.rstrip().endswith("?"):
        errors.append("reflective_question must contain one question mark at the end.")
    if not _OPEN_QUESTION.match(question.lstrip()):
        errors.append("reflective_question must begin with an open-ended question form.")
    if errors:
        return None, errors
    return CANDIDATE_ADAPTER.validate_python(payload), []


def render_candidate(candidate: CandidateAssessment) -> str:
    spec = CATEGORY_BY_ID[candidate.category_id]
    values = candidate.model_dump()
    lines = [f"Code category: {spec.display_label}"]
    lines.extend(f"{heading}: {values[field]}" for field, heading in spec.rendered_fields)
    return "\n".join(lines)


def without_redundant_response_fields(rendered_text: str) -> str:
    """Hide deprecated response lines while retaining historical audit records unchanged."""
    deprecated_prefixes = ("Code label:", "Evidence quote:", "Actual segment quote:")
    return "\n".join(
        line for line in rendered_text.splitlines()
        if not line.startswith(deprecated_prefixes)
    )


def candidate_to_json(candidate: CandidateAssessment) -> str:
    return json.dumps(candidate.model_dump(), ensure_ascii=False, sort_keys=True)

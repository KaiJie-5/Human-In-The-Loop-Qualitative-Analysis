from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .categories import CATEGORY_BY_ID


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
    evidence_quote: str
    why_plausible_for_wider_dataset: str
    why_unsupported_by_this_segment: str
    relation_to_research_questions: str
    reflective_question: str


class DescriptiveAssessment(_StrictAssessment):
    category_id: Literal["descriptive_not_answering_rq"]
    evidence_quote: str
    surface_description: str
    why_true_of_segment: str
    why_not_useful_for_research_questions: str
    relation_to_research_questions: str
    reflective_question: str


class TooBroadAssessment(_StrictAssessment):
    category_id: Literal["too_broad"]
    evidence_quote: str
    broad_relevance_to_research_questions: str
    specific_meaning_lost: str
    why_it_is_too_broad: str
    relation_to_research_questions: str
    reflective_question: str


class UsefulAssessment(_StrictAssessment):
    category_id: Literal["useful_analytical_code"]
    evidence_quote: str
    specific_analytical_insight: str
    why_it_is_useful: str
    relation_to_research_questions: str
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

@dataclass(frozen=True, slots=True)
class ResponseSection:
    label: str
    value: str
    is_evidence: bool = False


def candidate_json_schema() -> dict[str, Any]:
    return CANDIDATE_ADAPTER.json_schema()


def parse_candidate(raw_content: str) -> CandidateAssessment:
    return CANDIDATE_ADAPTER.validate_json(raw_content)


def render_candidate(candidate: CandidateAssessment) -> str:
    return "\n".join(
        f"{section.label}: {section.value}"
        for section in response_sections(candidate.model_dump())
    )


def response_sections(payload: Mapping[str, Any]) -> tuple[ResponseSection, ...]:
    """Return ordered display/export sections for current and historical candidates."""
    category_id = str(payload.get("category_id", ""))
    spec = CATEGORY_BY_ID.get(category_id)  # type: ignore[arg-type]
    if spec is None:
        return ()
    sections = [ResponseSection("Code category", spec.display_label)]
    evidence = payload.get("evidence_quote")
    if evidence is None:
        evidence = payload.get("actual_segment_quote")
    if evidence is not None and str(evidence).strip():
        sections.append(ResponseSection("Evidence quote", str(evidence), is_evidence=True))
    for field, heading in spec.rendered_fields:
        if field == "evidence_quote":
            continue
        value = payload.get(field)
        if value is not None and str(value).strip():
            sections.append(ResponseSection(heading, str(value)))
    return tuple(sections)


def normalize_response_text(rendered_text: str) -> str:
    """Normalize retired headings without rewriting immutable historical records."""
    normalized: list[str] = []
    for line in rendered_text.splitlines():
        if line.startswith(("Code label:", "Category boundary:")):
            continue
        if line.startswith("Actual segment quote:"):
            line = "Evidence quote:" + line.removeprefix("Actual segment quote:")
        normalized.append(line)
    return "\n".join(normalized)


def candidate_to_json(candidate: CandidateAssessment) -> str:
    return json.dumps(candidate.model_dump(), ensure_ascii=False, sort_keys=True)

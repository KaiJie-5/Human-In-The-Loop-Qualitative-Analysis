from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, field_validator

from .categories import CATEGORY_BY_ID, CategoryId


class CandidateAssessment(BaseModel):
    """Version-4 model-controlled response contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    category_id: CategoryId
    reflective_question: str

    @field_validator("reflective_question", mode="after")
    @classmethod
    def require_nonempty_reflection(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Reflective question must not be empty.")
        return value


@dataclass(frozen=True, slots=True)
class ResponseSection:
    label: str
    value: str


def candidate_json_schema() -> dict[str, Any]:
    return CandidateAssessment.model_json_schema()


def parse_candidate(raw_content: str) -> CandidateAssessment:
    return CandidateAssessment.model_validate_json(raw_content)


def render_response(category_id: str, reflective_question: str) -> str:
    spec = CATEGORY_BY_ID.get(category_id)  # type: ignore[arg-type]
    if spec is None:
        raise ValueError(f"Unsupported category ID {category_id!r}.")
    if not reflective_question.strip():
        raise ValueError("Reflective question must not be empty.")
    return (
        f"Code category: {spec.display_label}\n"
        f"Reflective question: {reflective_question}"
    )


def render_candidate(candidate: CandidateAssessment) -> str:
    return render_response(candidate.category_id, candidate.reflective_question)


def response_sections(
    payload: Mapping[str, Any],
    *,
    effective_category_id: str | None = None,
) -> tuple[ResponseSection, ...]:
    category_id = effective_category_id or str(payload.get("category_id", ""))
    spec = CATEGORY_BY_ID.get(category_id)  # type: ignore[arg-type]
    reflection = payload.get("reflective_question")
    if spec is None or reflection is None or not str(reflection).strip():
        return ()
    return (
        ResponseSection("Code category", spec.display_label),
        ResponseSection("Reflective question", str(reflection)),
    )


def historical_candidate_fields(
    parsed: Mapping[str, Any] | None,
    rendered_text: str | None,
) -> tuple[str, str] | None:
    """Extract the two v4 fields without modifying a historical audit record."""
    parsed = parsed or {}
    category_id = str(parsed.get("category_id") or "")
    reflection = str(parsed.get("reflective_question") or "")
    if category_id in CATEGORY_BY_ID and reflection.strip():
        return category_id, reflection

    category_label = ""
    reflection_lines: list[str] = []
    collecting_reflection = False
    for line in str(rendered_text or "").splitlines():
        if line.startswith("Code category:"):
            category_label = line.split(":", 1)[1].strip()
            collecting_reflection = False
        elif line.startswith("Reflective question:"):
            reflection_lines = [line.split(":", 1)[1].lstrip()]
            collecting_reflection = True
        elif collecting_reflection:
            reflection_lines.append(line)
    if category_id not in CATEGORY_BY_ID:
        category_id = next(
            (spec.id for spec in CATEGORY_BY_ID.values() if spec.display_label == category_label),
            "",
        )
    reflection = "\n".join(reflection_lines).strip()
    if category_id in CATEGORY_BY_ID and reflection:
        return category_id, reflection
    return None


def normalize_response_text(rendered_text: str) -> str:
    fields = historical_candidate_fields(None, rendered_text)
    return render_response(*fields) if fields else ""


def candidate_to_json(candidate: CandidateAssessment) -> str:
    return json.dumps(candidate.model_dump(), ensure_ascii=False, sort_keys=True)

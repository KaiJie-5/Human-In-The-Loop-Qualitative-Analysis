from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CATEGORY_CONTRACT_VERSION = "hitl_code_categories_v2"
CategoryId = Literal[
    "wrong_code",
    "descriptive_not_answering_rq",
    "too_broad",
    "useful_analytical_code",
]


@dataclass(frozen=True, slots=True)
class CategorySpec:
    id: CategoryId
    display_label: str
    definition: str
    rendered_fields: tuple[tuple[str, str], ...]


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        id="wrong_code",
        display_label="Wrong code",
        definition=(
            "The supplied code contradicts, misattributes, is unrelated to, or is "
            "unsupported by the target segment."
        ),
        rendered_fields=(
            ("why_plausible_for_wider_dataset", "Why plausible for wider dataset"),
            ("why_unsupported_by_this_segment", "Why unsupported by this segment"),
            ("relation_to_research_questions", "Relation to research questions"),
            ("category_boundary", "Category boundary"),
            ("reflective_question", "Reflective question"),
        ),
    ),
    CategorySpec(
        id="descriptive_not_answering_rq",
        display_label="Descriptive but not answering the research question",
        definition=(
            "The code is factually true at a surface level but does not meaningfully "
            "help answer the selected research questions."
        ),
        rendered_fields=(
            ("surface_description", "Surface description"),
            ("why_true_of_segment", "Why true of segment"),
            ("why_not_useful_for_research_questions", "Why not useful for research questions"),
            ("relation_to_research_questions", "Relation to research questions"),
            ("category_boundary", "Category boundary"),
            ("reflective_question", "Reflective question"),
        ),
    ),
    CategorySpec(
        id="too_broad",
        display_label="Too broad code",
        definition=(
            "The code is relevant but too vague or general and loses an important "
            "specific meaning in the segment."
        ),
        rendered_fields=(
            ("broad_relevance_to_research_questions", "Broad relevance to research questions"),
            ("specific_meaning_lost", "Specific meaning lost"),
            ("why_it_is_too_broad", "Why it is too broad"),
            ("relation_to_research_questions", "Relation to research questions"),
            ("category_boundary", "Category boundary"),
            ("reflective_question", "Reflective question"),
        ),
    ),
    CategorySpec(
        id="useful_analytical_code",
        display_label="Useful analytical code",
        definition=(
            "The code is grounded, sufficiently specific, interpretive, and useful "
            "for answering one or more selected research questions."
        ),
        rendered_fields=(
            ("specific_analytical_insight", "Specific analytical insight"),
            ("why_it_is_useful", "Why it is useful"),
            ("relation_to_research_questions", "Relation to research questions"),
            ("category_boundary", "Category boundary"),
            ("reflective_question", "Reflective question"),
        ),
    ),
)

CATEGORY_BY_ID = {spec.id: spec for spec in CATEGORY_SPECS}

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CATEGORY_CONTRACT_VERSION = "hitl_code_categories_v4"
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


CATEGORY_SPECS: tuple[CategorySpec, ...] = (
    CategorySpec(
        id="wrong_code",
        display_label="Wrong code",
        definition=(
            "The supplied code contradicts, misattributes, is unrelated to, or is "
            "unsupported by the target segment."
        ),
    ),
    CategorySpec(
        id="descriptive_not_answering_rq",
        display_label="Descriptive but not answering the research question",
        definition=(
            "The code is factually true at a surface level but does not meaningfully "
            "help answer the selected research questions."
        ),
    ),
    CategorySpec(
        id="too_broad",
        display_label="Too broad code",
        definition=(
            "The code is relevant but too vague or general and loses an important "
            "specific meaning in the segment."
        ),
    ),
    CategorySpec(
        id="useful_analytical_code",
        display_label="Useful analytical code",
        definition=(
            "The code is grounded, sufficiently specific, interpretive, and useful "
            "for answering one or more selected research questions."
        ),
    ),
)

CATEGORY_BY_ID = {spec.id: spec for spec in CATEGORY_SPECS}

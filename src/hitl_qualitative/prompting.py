from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from .candidates import candidate_json_schema
from .categories import CATEGORY_CONTRACT_VERSION, CATEGORY_SPECS
from .transcripts import TranscriptTurn


PROMPT_VERSION = "hitl_code_assessment_v4"


@dataclass(frozen=True, slots=True)
class QuestionSnapshot:
    id: int
    version: int
    text: str


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    prompt: str
    prompt_sha256: str
    input_fingerprint: str


def build_prompt(
    *,
    previous: Sequence[TranscriptTurn],
    target_text: str,
    target_turns: Sequence[TranscriptTurn],
    following: Sequence[TranscriptTurn],
    exact_code_label: str,
    questions: Sequence[QuestionSnapshot],
    generation_identity: dict[str, Any],
) -> PromptSnapshot:
    if not exact_code_label.strip():
        raise ValueError("Researcher code must not be empty.")
    if not questions:
        raise ValueError("At least one research question must be selected.")
    categories = "\n".join(
        f"- {spec.id} ({spec.display_label}): {spec.definition}" for spec in CATEGORY_SPECS
    )
    question_text = "\n".join(
        f"{index}. {question.text}" for index, question in enumerate(questions, 1)
    )
    schema_text = json.dumps(candidate_json_schema(), ensure_ascii=False, sort_keys=True)
    prompt = (
        "You are assisting a reflexive qualitative researcher using reflexive thematic analysis.\n\n"
        "EVIDENCE BOUNDARY\n"
        "- The Target segment is the primary and only evidence for the coding decision.\n"
        "- Previous and next transcript turns may clarify language, sequence, or ambiguity, "
        "but do not present text from those turns as evidence for the target segment.\n"
        "- Do not invent quotations or rely on another turn as evidence for the target.\n\n"
        "TASK\n"
        "Assess the exact researcher-supplied code shown below. Do not replace, rewrite, or "
        "repeat the code label in your JSON. Use the target evidence and selected research "
        "questions to classify the coding decision using exactly one category, then produce "
        "one specific open-ended reflective question for the qualitative "
        "researcher. The question must not be yes/no and must not merely ask whether the "
        "category is correct.\n\n"
        f"CATEGORY CONTRACT ({CATEGORY_CONTRACT_VERSION})\n{categories}\n\n"
        f"SELECTED RESEARCH QUESTIONS\n{question_text}\n\n"
        f"PREVIOUS CONTEXT (clarification only)\n{_render_turns(previous)}\n\n"
        f"TARGET SEGMENT (only evidence)\n{_render_target(target_text, target_turns)}\n\n"
        f"NEXT CONTEXT (clarification only)\n{_render_turns(following)}\n\n"
        f"RESEARCHER-SUPPLIED CODE\n{exact_code_label}\n\n"
        "OUTPUT\nReturn only one JSON object matching this JSON Schema, with no markdown or "
        f"additional fields:\n{schema_text}"
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    fingerprint_payload = {
        "prompt_version": PROMPT_VERSION,
        "category_version": CATEGORY_CONTRACT_VERSION,
        "prompt_sha256": prompt_sha,
        "code_label": exact_code_label,
        "questions": [
            {"id": question.id, "version": question.version, "text": question.text}
            for question in questions
        ],
        "generation": generation_identity,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PromptSnapshot(prompt, prompt_sha, fingerprint)


def build_repair_prompt(
    *,
    original_prompt: str,
    invalid_content: str,
    errors: Sequence[str],
) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        f"{original_prompt}\n\n"
        "REPAIR REQUEST\n"
        "The prior response below failed validation. Return a complete replacement JSON "
        "object, correcting only the listed problems and adding no fields outside the schema.\n"
        f"Validation errors:\n{listed}\n"
        f"Prior response:\n{invalid_content}"
    )


def _render_turns(turns: Sequence[TranscriptTurn]) -> str:
    if not turns:
        return "[No turns in this direction]"
    return "\n".join(
        f"Turn {turn.turn_index} | {turn.speaker_label or turn.speaker.capitalize()}: {turn.text}"
        for turn in turns
    )


def _render_target(target_text: str, turns: Sequence[TranscriptTurn]) -> str:
    indexes = ", ".join(str(turn.turn_index) for turn in turns)
    return f"Target turn index(es): {indexes}\n{target_text}"

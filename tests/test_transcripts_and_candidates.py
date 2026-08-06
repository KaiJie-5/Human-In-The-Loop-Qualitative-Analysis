from __future__ import annotations

import json

import pytest

from hitl_qualitative.candidates import (
    CANDIDATE_ADAPTER,
    normalize_response_text,
    render_candidate,
    response_sections,
)
from hitl_qualitative.categories import CATEGORY_BY_ID
from hitl_qualitative.prompting import QuestionSnapshot, build_prompt
from hitl_qualitative.transcripts import TranscriptAdapter, select_context
from hitl_qualitative.ui import _response_card_html

from conftest import segment_payload, valid_candidate


def test_context_counts_are_turn_counts_and_stay_inside_transcript() -> None:
    payload = segment_payload()
    bundle = TranscriptAdapter().from_upload(
        "segments.jsonl", (json.dumps(payload) + "\n").encode()
    )
    transcript = bundle.transcripts[0]
    target = transcript.targets[0]
    previous, following = select_context(transcript.turns, target.target_turn_indexes, 20, 2)
    assert [turn.turn_index for turn in previous] == [1]
    assert [turn.turn_index for turn in following] == [3, 4]


def test_import_rejects_inconsistent_transcript_and_resumes_by_checksum(prepared_store) -> None:
    store, study_id, dataset_id = prepared_store
    payload = segment_payload()
    bundle = TranscriptAdapter().from_upload(
        "segments.jsonl", (json.dumps(payload) + "\n").encode()
    )
    resumed_id, created = store.import_dataset(
        study_id=study_id, name="Another name", split="adaptation",
        source_kind="upload", bundle=bundle,
    )
    assert not created
    assert resumed_id == dataset_id

    second = segment_payload(record_id="INT001_SEG002")
    second["text"] = "The total was unclear to me."
    second["turn_index"] = 4
    second["segment_id"] = "SEG002"
    second["interview_turns"][0]["text"] = "Changed transcript"
    data = "\n".join(json.dumps(value) for value in (payload, second)).encode()
    with pytest.raises(ValueError, match="inconsistent interview_turns"):
        TranscriptAdapter().from_upload("bad.jsonl", data)


@pytest.mark.parametrize(
    ("category", "expected_headings"),
    [
        (
            "wrong_code",
            [
                "Code category", "Evidence quote", "Why plausible for wider dataset",
                "Why unsupported by this segment", "Relation to research questions",
                "Reflective question",
            ],
        ),
        (
            "descriptive_not_answering_rq",
            [
                "Code category", "Evidence quote", "Surface description",
                "Why true of segment", "Why not useful for research questions",
                "Relation to research questions", "Reflective question",
            ],
        ),
        (
            "too_broad",
            [
                "Code category", "Evidence quote",
                "Broad relevance to research questions", "Specific meaning lost",
                "Why it is too broad", "Relation to research questions",
                "Reflective question",
            ],
        ),
        (
            "useful_analytical_code",
            [
                "Code category", "Evidence quote", "Specific analytical insight",
                "Why it is useful", "Relation to research questions", "Reflective question",
            ],
        ),
    ],
)
def test_all_category_schemas_render_exact_headings(category: str, expected_headings: list[str]) -> None:
    candidate = CANDIDATE_ADAPTER.validate_python(valid_candidate(category))
    rendered = render_candidate(candidate)
    assert rendered.splitlines()[0] == f"Code category: {CATEGORY_BY_ID[category].display_label}"
    assert rendered.splitlines()[1] == "Evidence quote: I checked the bill twice."
    assert "Code label:" not in rendered
    assert "Actual segment quote:" not in rendered
    assert "Category boundary:" not in rendered
    assert [line.split(":", 1)[0] for line in rendered.splitlines()] == expected_headings


@pytest.mark.parametrize(
    "quote",
    [
        "context-only quote",
        "invented evidence",
        "bill and bill",
        "non contiguous evidence",
    ],
)
def test_evidence_is_preserved_verbatim_without_semantic_validation(quote: str) -> None:
    payload = valid_candidate()
    payload["evidence_quote"] = quote
    payload["reflective_question"] = "A reflective prompt without question punctuation"
    candidate = CANDIDATE_ADAPTER.validate_python(payload)
    assert candidate.evidence_quote == quote
    assert f"Evidence quote: {quote}" in render_candidate(candidate)


def test_schema_rejects_retired_boundary_and_empty_reflection() -> None:
    payload = valid_candidate()
    payload["category_boundary"] = "Retired field"
    with pytest.raises(ValueError):
        CANDIDATE_ADAPTER.validate_python(payload)
    payload.pop("category_boundary")
    payload["reflective_question"] = "   "
    with pytest.raises(ValueError):
        CANDIDATE_ADAPTER.validate_python(payload)


def test_schema_requires_evidence_quote() -> None:
    payload = valid_candidate()
    payload.pop("evidence_quote")
    with pytest.raises(ValueError):
        CANDIDATE_ADAPTER.validate_python(payload)


def test_legacy_response_is_normalized_without_losing_evidence() -> None:
    legacy = (
        "Code category: Wrong code\n"
        "Code label: regular check-ups\n"
        "Actual segment quote: regular check-ups\n"
        "Category boundary: Segment\n"
        "Why it is useful: It identifies a pattern."
    )
    visible = normalize_response_text(legacy)
    assert "Code label:" not in visible
    assert "Category boundary:" not in visible
    assert "Evidence quote: regular check-ups" in visible
    assert "Why it is useful: It identifies a pattern." in visible


def test_response_card_highlights_labels_and_escapes_model_content() -> None:
    payload = valid_candidate()
    payload["evidence_quote"] = "<script>alert('unsafe')</script>"
    sections = response_sections(payload)
    rendered_html = _response_card_html(sections)
    assert 'class="response-field-label"' in rendered_html
    assert "Evidence quote" in rendered_html
    assert "response-field-evidence" in rendered_html
    assert "&lt;script&gt;" in rendered_html
    assert "<script>" not in rendered_html


def test_prompt_keeps_context_target_and_questions_separate() -> None:
    bundle = TranscriptAdapter().from_upload(
        "segments.jsonl", (json.dumps(segment_payload()) + "\n").encode()
    )
    transcript = bundle.transcripts[0]
    target = transcript.targets[0]
    previous, following = select_context(transcript.turns, target.target_turn_indexes, 1, 1)
    target_turns = tuple(turn for turn in transcript.turns if turn.turn_index == 2)
    snapshot = build_prompt(
        previous=previous,
        target_text=target.text,
        target_turns=target_turns,
        following=following,
        exact_code_label="rechecking uncertain bills",
        questions=[QuestionSnapshot(1, 2, "How do people respond to unclear bills?")],
        generation_identity={"model": "test", "seed_policy": "independent"},
    )
    assert "PREVIOUS CONTEXT (clarification only)" in snapshot.prompt
    assert "TARGET SEGMENT (only evidence)" in snapshot.prompt
    assert "NEXT CONTEXT (clarification only)" in snapshot.prompt
    assert "Return one evidence_quote taken from the Target segment" in snapshot.prompt
    assert "category_boundary" not in snapshot.prompt
    assert snapshot.prompt.index("What happened?") < snapshot.prompt.index(target.text)
    assert snapshot.prompt.index(target.text) < snapshot.prompt.index("Why twice?")

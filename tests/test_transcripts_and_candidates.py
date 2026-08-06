from __future__ import annotations

import json

import pytest

from hitl_qualitative.candidates import (
    CANDIDATE_ADAPTER,
    render_candidate,
    validate_candidate,
    without_redundant_response_fields,
)
from hitl_qualitative.categories import CATEGORY_BY_ID
from hitl_qualitative.prompting import QuestionSnapshot, build_prompt
from hitl_qualitative.transcripts import TranscriptAdapter, select_context

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
        ("wrong_code", ["Why plausible for wider dataset"]),
        ("descriptive_not_answering_rq", ["Surface description"]),
        ("too_broad", ["Broad relevance to research questions"]),
        ("useful_analytical_code", ["Specific analytical insight"]),
    ],
)
def test_all_category_schemas_render_exact_headings(category: str, expected_headings: list[str]) -> None:
    candidate = CANDIDATE_ADAPTER.validate_python(valid_candidate(category))
    rendered = render_candidate(candidate)
    assert rendered.splitlines()[0] == f"Code category: {CATEGORY_BY_ID[category].display_label}"
    assert "Code label:" not in rendered
    assert "Evidence quote:" not in rendered
    assert "Actual segment quote:" not in rendered
    for heading in expected_headings:
        assert f"{heading}:" in rendered


def test_deprecated_response_fields_are_rejected_and_open_question_is_enforced() -> None:
    payload = valid_candidate()
    payload["evidence_quote"] = "A deprecated field."
    with pytest.raises(ValueError):
        CANDIDATE_ADAPTER.validate_python(payload)
    payload.pop("evidence_quote")
    payload["reflective_question"] = "Is this code correct?"
    candidate = CANDIDATE_ADAPTER.validate_python(payload)
    validated, errors = validate_candidate(candidate)
    assert validated is None
    assert any("open-ended" in error for error in errors)


def test_legacy_redundant_lines_are_hidden() -> None:
    legacy = (
        "Code category: Useful analytical code\n"
        "Code label: regular check-ups\n"
        "Evidence quote: regular check-ups\n"
        "Why it is useful: It identifies a pattern."
    )
    visible = without_redundant_response_fields(legacy)
    assert "Code label:" not in visible
    assert "Evidence quote:" not in visible
    assert "Why it is useful: It identifies a pattern." in visible


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
    assert snapshot.prompt.index("What happened?") < snapshot.prompt.index(target.text)
    assert snapshot.prompt.index(target.text) < snapshot.prompt.index("Why twice?")

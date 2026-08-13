from __future__ import annotations

import json

import pytest

from hitl_qualitative.candidates import (
    CandidateAssessment,
    historical_candidate_fields,
    normalize_response_text,
    render_candidate,
    response_sections,
)
from hitl_qualitative.categories import CATEGORY_BY_ID
from hitl_qualitative.prompting import QuestionSnapshot, build_prompt
from hitl_qualitative.transcripts import TranscriptAdapter, select_context
from hitl_qualitative.ui import _response_card_html, sticky_reference_html

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
    resumed_id, created = store.import_adaptation_dataset(
        study_id=study_id, name="Another name",
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


@pytest.mark.parametrize("category", list(CATEGORY_BY_ID))
def test_all_categories_use_the_exact_two_field_contract(category: str) -> None:
    candidate = CandidateAssessment.model_validate(valid_candidate(category))
    rendered = render_candidate(candidate)
    assert rendered.splitlines() == [
        f"Code category: {CATEGORY_BY_ID[category].display_label}",
        "Reflective question: How does checking twice illuminate the participant's response to uncertainty?",
    ]
    assert [section.label for section in response_sections(candidate.model_dump())] == [
        "Code category", "Reflective question"
    ]


def test_schema_rejects_unknown_fields_and_empty_values() -> None:
    payload = valid_candidate()
    payload["evidence_quote"] = "retired"
    with pytest.raises(ValueError):
        CandidateAssessment.model_validate(payload)
    payload.pop("evidence_quote")
    payload["reflective_question"] = "   "
    with pytest.raises(ValueError):
        CandidateAssessment.model_validate(payload)
    payload = valid_candidate()
    payload["category_id"] = "unknown"
    with pytest.raises(ValueError):
        CandidateAssessment.model_validate(payload)


def test_historical_full_response_normalizes_to_two_fields() -> None:
    legacy = (
        "Code category: Wrong code\n"
        "Code label: regular check-ups\n"
        "Actual segment quote: regular check-ups\n"
        "Why unsupported by this segment: It is not grounded.\n"
        "Category boundary: Historical explanation\n"
        "Reflective question: How could the code attend more closely to the target?"
    )
    fields = historical_candidate_fields(None, legacy)
    assert fields == (
        "wrong_code", "How could the code attend more closely to the target?"
    )
    assert normalize_response_text(legacy).splitlines() == [
        "Code category: Wrong code",
        "Reflective question: How could the code attend more closely to the target?",
    ]


def test_response_card_and_sticky_reference_escape_protected_content() -> None:
    sections = response_sections(
        {
            "category_id": "wrong_code",
            "reflective_question": "<script>alert('unsafe')</script>",
        }
    )
    rendered_html = _response_card_html(sections)
    assert 'class="response-field-label"' in rendered_html
    assert "&lt;script&gt;" in rendered_html
    assert "<script>" not in rendered_html

    sticky = sticky_reference_html(
        transcript_id="<INT>", segment_id="SEG&1", split="adaptation",
        target_text="<protected>",
        turn_labels="2", questions=("Who & why?",), reviewed=1, total=3,
    )
    assert "&lt;protected&gt;" in sticky
    assert "&lt;INT&gt;" in sticky
    assert "Who &amp; why?" in sticky


def test_prompt_keeps_context_target_questions_and_v4_schema_separate() -> None:
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
    assert '"category_id"' in snapshot.prompt
    assert '"reflective_question"' in snapshot.prompt
    assert "evidence_quote" not in snapshot.prompt
    assert snapshot.prompt.index("What happened?") < snapshot.prompt.index(target.text)
    assert snapshot.prompt.index(target.text) < snapshot.prompt.index("Why twice?")

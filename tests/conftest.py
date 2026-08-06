from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_qualitative.database import QuestionDraft, SQLiteStore
from hitl_qualitative.ollama_client import ModelInfo, OllamaResponse
from hitl_qualitative.transcripts import TranscriptAdapter


def segment_payload(*, transcript_id: str = "INT001", record_id: str = "INT001_SEG001") -> dict[str, Any]:
    turns = [
        {"turn_index": 1, "speaker": "interviewer", "text": "What happened?", "paragraph_index": 1},
        {"turn_index": 2, "speaker": "participant", "text": "I checked the bill twice.", "paragraph_index": 2},
        {"turn_index": 3, "speaker": "interviewer", "text": "Why twice?", "paragraph_index": 3},
        {"turn_index": 4, "speaker": "participant", "text": "The total was unclear to me.", "paragraph_index": 4},
        {"turn_index": 5, "speaker": "interviewer", "text": "What did you do next?", "paragraph_index": 5},
    ]
    return {
        "record_id": record_id,
        "text": "I checked the bill twice.",
        "interview_id": transcript_id,
        "segment_id": "SEG001",
        "speaker": "participant",
        "turn_index": 2,
        "previous_context": "Interviewer: What happened?",
        "next_context": "Interviewer: Why twice?",
        "source_html_path": "normalized/INT001.html",
        "interview_turns": turns,
    }


def valid_candidate(category: str = "useful_analytical_code") -> dict[str, Any]:
    common = {
        "evidence_quote": "I checked the bill twice.",
        "relation_to_research_questions": "It advances the question about how people handle unclear bills.",
        "reflective_question": "How does checking twice illuminate the participant's response to uncertainty?",
    }
    if category == "wrong_code":
        return {
            "category_id": category,
            "why_plausible_for_wider_dataset": "Other participants may ignore their bills.",
            "why_unsupported_by_this_segment": "This participant explicitly describes checking.",
            **common,
        }
    if category == "descriptive_not_answering_rq":
        return {
            "category_id": category,
            "surface_description": "The participant mentions a bill.",
            "why_true_of_segment": "The word bill occurs in the target.",
            "why_not_useful_for_research_questions": "Mention alone does not explain an interaction.",
            **common,
        }
    if category == "too_broad":
        return {
            "category_id": category,
            "broad_relevance_to_research_questions": "It concerns an interaction with billing.",
            "specific_meaning_lost": "It loses the repeated checking caused by uncertainty.",
            "why_it_is_too_broad": "The label does not preserve the checking strategy.",
            **common,
        }
    return {
        "category_id": "useful_analytical_code",
        "specific_analytical_insight": "Repeated checking is a response to billing uncertainty.",
        "why_it_is_useful": "It links a concrete action to the participant's uncertainty.",
        **common,
    }


class FakeOllamaClient:
    def __init__(self, outcomes: list[Any] | None = None):
        self.outcomes = list(outcomes or [valid_candidate(), valid_candidate("too_broad")])
        self.calls: list[dict[str, Any]] = []

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo("test-model:latest", "sha256:test")]

    def show_model(self, model: str) -> ModelInfo:
        return ModelInfo(model, "sha256:test")

    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        schema: dict[str, Any],
        options: dict[str, Any],
        seed: int,
    ) -> OllamaResponse:
        self.calls.append(
            {"model": model, "prompt": prompt, "schema": schema, "options": options, "seed": seed}
        )
        if not self.outcomes:
            raise AssertionError("Fake Ollama client has no configured outcome.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        content = outcome if isinstance(outcome, str) else json.dumps(outcome, ensure_ascii=False)
        payload = {
            "model": model,
            "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content},
            "done": True,
            "done_reason": "stop",
        }
        return OllamaResponse(
            raw_body=json.dumps(payload, ensure_ascii=False), payload=payload,
            content=content, model=model, metadata={"done": True, "done_reason": "stop"},
        )


@pytest.fixture
def prepared_store(tmp_path: Path) -> tuple[SQLiteStore, int, int]:
    store = SQLiteStore(tmp_path / "review.sqlite3")
    store.initialize()
    study_id = store.create_study(
        name="Study", reviewer_id="reviewer-01", ollama_base_url="http://localhost:11434"
    )
    store.save_questions(
        study_id,
        [QuestionDraft(None, "How do people respond to unclear service bills?", selected=True)],
    )
    store.update_study(
        study_id,
        reviewer_id="reviewer-01",
        ollama_base_url="http://localhost:11434",
        model_name="test-model:latest",
        model_digest="sha256:test",
        context_before=20,
        context_after=20,
        symmetric_context=True,
        temperature=0.4,
        top_p=0.9,
        output_tokens=5000,
        context_tokens=65536,
    )
    data = (json.dumps(segment_payload(), ensure_ascii=False) + "\n").encode("utf-8")
    bundle = TranscriptAdapter().from_upload("segments.jsonl", data)
    dataset_id, created = store.import_dataset(
        study_id=study_id, name="Dataset", split="adaptation", source_kind="upload", bundle=bundle
    )
    assert created
    return store, study_id, dataset_id

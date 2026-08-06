from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hitl_qualitative.database import QuestionDraft
from hitl_qualitative.exporting import PreferenceExporter, validate_conversation_row
from hitl_qualitative.ollama_client import OllamaConnectionError
from hitl_qualitative.workflow import ReviewService, decision_idempotency_key

from conftest import FakeOllamaClient, valid_candidate


def test_two_seeds_are_distinct_with_identical_prompt_and_options(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient()
    snapshot = ReviewService(store, fake).generate_pair(item, "checking bills twice")
    assert snapshot.status == "ready"
    assert len(fake.calls) == 2
    assert fake.calls[0]["seed"] != fake.calls[1]["seed"]
    for field in ("model", "prompt", "schema", "options"):
        assert fake.calls[0][field] == fake.calls[1][field]
    assert fake.calls[0]["options"]["num_ctx"] == 65536
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT raw_response_json, parsed_json, rendered_text FROM candidates ORDER BY candidate_number"
        ).fetchall()
    assert len(rows) == 2
    assert all(row["raw_response_json"] and row["parsed_json"] and row["rendered_text"] for row in rows)


def test_multiple_question_order_and_versions_are_durable(prepared_store) -> None:
    store, study_id, _ = prepared_store
    first = store.get_questions(study_id)[0]
    store.save_questions(
        study_id,
        [
            QuestionDraft(int(first["id"]), str(first["text"]), selected=False),
            QuestionDraft(None, "What strategies do people use when billing is unclear?", selected=True),
        ],
    )
    rows = store.get_questions(study_id)
    assert [row["display_order"] for row in rows] == [1, 2]
    assert [row["text"] for row in store.get_questions(study_id, selected_only=True)] == [
        "What strategies do people use when billing is unclear?"
    ]
    store.save_questions(
        study_id,
        [
            QuestionDraft(int(rows[1]["id"]), str(rows[1]["text"]), selected=True),
            QuestionDraft(int(rows[0]["id"]), "How does uncertainty affect bill checking?", selected=True),
        ],
    )
    revised = store.get_questions(study_id)
    assert [row["id"] for row in revised] == [rows[1]["id"], rows[0]["id"]]
    assert revised[1]["version"] == 2


def test_generation_and_decision_are_idempotent(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient()
    service = ReviewService(store, fake)
    first = service.generate_pair(item, "checking bills twice")
    second = service.generate_pair(item, "checking bills twice")
    assert first.id == second.id
    assert [candidate.candidate_number for candidate in first.candidates] == [
        candidate.candidate_number for candidate in second.candidates
    ]
    assert len(fake.calls) == 2
    key = decision_idempotency_key(item.id, "prefer_a", first.id)
    decision_1 = service.save_decision(item=item, decision="prefer_a", idempotency_key=key)
    decision_2 = service.save_decision(item=item, decision="prefer_a", idempotency_key=key)
    assert decision_1 == decision_2


def test_ab_mapping_drives_chosen_and_rejected_export(prepared_store, tmp_path: Path) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient([valid_candidate("wrong_code"), valid_candidate("too_broad")])
    service = ReviewService(store, fake)
    snapshot = service.generate_pair(item, "checking bills twice")
    response_a = next(candidate for candidate in snapshot.candidates if candidate.display_label == "A")
    service.save_decision(
        item=item,
        decision="prefer_a",
        idempotency_key=decision_idempotency_key(item.id, "prefer_a", snapshot.id),
    )
    exporter = PreferenceExporter(
        store,
        tmp_path / "exports",
        clock=lambda: datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
    )
    result = exporter.export(dataset_id)
    row = json.loads(result.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    validate_conversation_row(row)
    assert row["chosen"][0]["content"] == response_a.rendered_text
    assert "Code label:" not in row["chosen"][0]["content"]
    assert "Evidence quote:" not in row["chosen"][0]["content"]
    assert "Actual segment quote:" not in row["chosen"][0]["content"]
    assert "reviewer-01" not in result.jsonl_path.read_text(encoding="utf-8")
    assert "reviewer-01" not in result.manifest_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("invalid_candidate", "invalid_candidate"),
        ("superseded_snapshot", "invalid_or_superseded_snapshot"),
    ],
)
def test_invalid_and_superseded_pairs_are_excluded(
    prepared_store, tmp_path: Path, mutation: str, reason: str
) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    service = ReviewService(store, FakeOllamaClient())
    snapshot = service.generate_pair(item, "checking bills twice")
    service.save_decision(
        item=item,
        decision="prefer_a",
        idempotency_key=decision_idempotency_key(item.id, "prefer_a", snapshot.id),
    )
    with store.transaction() as connection:
        if mutation == "invalid_candidate":
            connection.execute(
                "UPDATE candidates SET valid = 0 WHERE snapshot_id = ? AND candidate_number = 1",
                (snapshot.id,),
            )
        else:
            connection.execute(
                "UPDATE generation_snapshots SET status = 'superseded' WHERE id = ?",
                (snapshot.id,),
            )
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {reason: 1}


def test_refresh_recovers_partial_generation_without_repeating_candidate_one(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    interrupted = FakeOllamaClient([valid_candidate(), OllamaConnectionError("offline")])
    with pytest.raises(OllamaConnectionError):
        ReviewService(store, interrupted).generate_pair(item, "checking bills twice")
    active = ReviewService(store, FakeOllamaClient()).active_snapshot(item.id)
    assert active is not None and active.status == "generating"
    resumed_client = FakeOllamaClient([valid_candidate("too_broad")])
    recovered = ReviewService(store, resumed_client).generate_pair(item, "checking bills twice")
    assert recovered.status == "ready"
    assert len(resumed_client.calls) == 1


def test_invalid_candidate_is_audited_and_excluded_without_live_ollama(
    prepared_store, tmp_path: Path
) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient([{}, {}, valid_candidate("too_broad")])
    snapshot = ReviewService(store, fake).generate_pair(item, "checking bills twice")
    assert snapshot.status == "invalid"
    assert len(fake.calls) == 3
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {"invalid_candidate": 1}


def test_research_question_edits_supersede_but_do_not_mutate_snapshot(prepared_store) -> None:
    store, study_id, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    service = ReviewService(store, FakeOllamaClient())
    snapshot = service.generate_pair(item, "checking bills twice")
    old_text = snapshot.questions[0].text
    question = store.get_questions(study_id)[0]
    store.save_questions(
        study_id,
        [QuestionDraft(int(question["id"]), "How does uncertainty shape repeated checking?", True)],
    )
    assert service.active_snapshot(item.id) is None
    historical = service.load_snapshot(snapshot.id)
    assert historical.status == "superseded"
    assert historical.questions[0].text == old_text


def test_three_attempt_cap_and_code_change_supersession(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 3)
    service = ReviewService(store, fake, maximum_pair_attempts=3)
    first = service.generate_pair(item, "code one")
    second = service.generate_pair(item, "code two")
    third = service.generate_pair(item, "code three")
    assert service.load_snapshot(first.id).status == "superseded"
    assert service.load_snapshot(second.id).status == "superseded"
    assert third.status == "ready"
    with pytest.raises(ValueError, match="limit of 3"):
        service.generate_pair(item, "code four")


def test_pre_generation_skip_is_final_and_excluded(prepared_store, tmp_path: Path) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    service = ReviewService(store, FakeOllamaClient())
    key = decision_idempotency_key(item.id, "skip", None)
    service.save_decision(
        item=item, decision="skip", reason="Not analytically useful", idempotency_key=key
    )
    assert store.get_next_item(dataset_id) is None
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {"skip": 1}
    with pytest.raises(ValueError, match="immutable decision"):
        service.save_decision(item=item, decision="skip", idempotency_key="different-key")


@pytest.mark.parametrize("decision", ["both_poor", "too_similar"])
def test_non_preference_decisions_are_excluded(prepared_store, tmp_path: Path, decision: str) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    service = ReviewService(store, FakeOllamaClient())
    snapshot = service.generate_pair(item, "checking bills twice")
    service.save_decision(
        item=item,
        decision=decision,
        idempotency_key=decision_idempotency_key(item.id, decision, snapshot.id),
    )
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {decision: 1}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_validation_and_test_splits_can_never_export(
    prepared_store, tmp_path: Path, split: str
) -> None:
    store, study_id, _ = prepared_store
    from conftest import segment_payload
    from hitl_qualitative.transcripts import TranscriptAdapter

    data = (
        json.dumps(segment_payload(transcript_id="INT999", record_id="INT999_SEG001")) + "\n"
    ).encode()
    bundle = TranscriptAdapter().from_upload(f"{split}.jsonl", data)
    dataset_id, _ = store.import_dataset(
        study_id=study_id,
        name=split.title(),
        split=split,
        source_kind="upload",
        bundle=bundle,
    )
    exporter = PreferenceExporter(store, tmp_path / "exports")
    assert exporter.preview(dataset_id).exclusion_counts == {"non_adaptation_split": 1}
    with pytest.raises(ValueError, match="Only adaptation"):
        exporter.export(dataset_id)


def test_schema_migration_and_foreign_keys_are_enabled(prepared_store) -> None:
    store, study_id, dataset_id = prepared_store
    store.set_active_dataset(study_id, dataset_id)
    from hitl_qualitative.database import SQLiteStore

    restarted = SQLiteStore(store.path)
    restarted.initialize()
    assert restarted.recover_active_dataset() == dataset_id
    with store.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1

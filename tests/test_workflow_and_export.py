from __future__ import annotations

import hashlib
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
    assert "Evidence quote:" in row["chosen"][0]["content"]
    assert "Actual segment quote:" not in row["chosen"][0]["content"]
    assert "Category boundary:" not in row["chosen"][0]["content"]
    assert "reviewer-01" not in result.jsonl_path.read_text(encoding="utf-8")
    assert "reviewer-01" not in result.manifest_path.read_text(encoding="utf-8")


def test_historical_export_normalizes_retired_headings_and_preserves_evidence(
    prepared_store, tmp_path: Path
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
        connection.execute(
            "UPDATE generation_snapshots SET category_version = 'hitl_code_categories_v1' WHERE id = ?",
            (snapshot.id,),
        )
        candidates = connection.execute(
            "SELECT id, rendered_text FROM candidates WHERE snapshot_id = ?",
            (snapshot.id,),
        ).fetchall()
        for candidate in candidates:
            historical = str(candidate["rendered_text"]).replace(
                "Evidence quote:",
                "Code label: checking bills twice\nActual segment quote:",
            )
            historical += "\nCategory boundary: Retired historical explanation"
            connection.execute(
                "UPDATE candidates SET rendered_text = ?, rendered_sha256 = ? WHERE id = ?",
                (
                    historical,
                    hashlib.sha256(historical.encode("utf-8")).hexdigest(),
                    candidate["id"],
                ),
            )
    exporter = PreferenceExporter(
        store,
        tmp_path / "historical-exports",
        clock=lambda: datetime(2026, 8, 5, 13, 0, tzinfo=timezone.utc),
    )
    result = exporter.export(dataset_id)
    row = json.loads(result.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    for side in ("chosen", "rejected"):
        content = row[side][0]["content"]
        assert "Evidence quote: I checked the bill twice." in content
        assert "Actual segment quote:" not in content
        assert "Code label:" not in content
        assert "Category boundary:" not in content


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


def test_research_question_edits_preserve_history_until_replacement_completes(
    prepared_store,
) -> None:
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
    replacement = ReviewService(store, FakeOllamaClient()).generate_pair(
        item, "checking bills twice"
    )
    assert replacement.status == "ready"
    assert replacement.questions[0].text == "How does uncertainty shape repeated checking?"
    with pytest.raises(KeyError, match="Unknown snapshot"):
        service.load_snapshot(snapshot.id)


def test_more_than_three_explicit_regenerations_replace_the_previous_pair(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 6)
    service = ReviewService(store, fake)
    current = service.generate_pair(item, "code 0")
    retired_ids: list[int] = []
    for attempt in range(1, 6):
        retired_ids.append(current.id)
        prior_seeds = _snapshot_seeds(store, current.id)
        current = service.generate_pair(
            item,
            f"code {attempt}",
            replace_snapshot_id=current.id,
        )
        assert current.status == "ready"
        assert _snapshot_seeds(store, current.id) != prior_seeds
    assert current.attempt_number == 6
    assert len(fake.calls) == 12
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM generation_snapshots").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM candidate_calls").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM snapshot_context").fetchone()[0] > 0
        assert connection.execute("SELECT COUNT(*) FROM snapshot_questions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM ab_assignments").fetchone()[0] == 2
    for snapshot_id in retired_ids:
        with pytest.raises(KeyError, match="Unknown snapshot"):
            service.load_snapshot(snapshot_id)


def test_regeneration_double_submit_returns_the_same_replacement(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    fake = FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 2)
    service = ReviewService(store, fake)
    original = service.generate_pair(item, "code")
    replacement = service.generate_pair(item, "code", replace_snapshot_id=original.id)
    repeated = service.generate_pair(item, "code", replace_snapshot_id=original.id)
    assert repeated.id == replacement.id
    assert len(fake.calls) == 4
    assert [
        (candidate.display_label, candidate.candidate_number)
        for candidate in repeated.candidates
    ] == [
        (candidate.display_label, candidate.candidate_number)
        for candidate in replacement.candidates
    ]


def test_failed_regeneration_keeps_old_pair_until_resumed_replacement_completes(
    prepared_store,
) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    original = ReviewService(store, FakeOllamaClient()).generate_pair(item, "code")
    interrupted = ReviewService(
        store,
        FakeOllamaClient([valid_candidate(), OllamaConnectionError("offline")]),
    )
    with pytest.raises(OllamaConnectionError):
        interrupted.generate_pair(item, "replacement code", replace_snapshot_id=original.id)
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT id, status FROM generation_snapshots ORDER BY attempt_number"
        ).fetchall()
    assert [(row["id"], row["status"]) for row in rows] == [
        (original.id, "superseded"),
        (rows[1]["id"], "generating"),
    ]
    resumed = ReviewService(store, FakeOllamaClient([valid_candidate("too_broad")]))
    replacement = resumed.generate_pair(item, "replacement code")
    assert replacement.status == "ready"
    with pytest.raises(KeyError, match="Unknown snapshot"):
        resumed.load_snapshot(original.id)
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM generation_snapshots").fetchone()[0] == 1


def test_invalid_replacement_keeps_prior_pair_for_audit(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    original = ReviewService(store, FakeOllamaClient()).generate_pair(item, "code")
    invalid_client = FakeOllamaClient([{}, {}, {}, {}])
    replacement = ReviewService(store, invalid_client).generate_pair(
        item,
        "replacement code",
        replace_snapshot_id=original.id,
    )
    assert replacement.status == "invalid"
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT id, status FROM generation_snapshots ORDER BY attempt_number"
        ).fetchall()
    assert [(row["id"], row["status"]) for row in rows] == [
        (original.id, "superseded"),
        (replacement.id, "invalid"),
    ]


def test_regeneration_is_prohibited_after_an_immutable_decision(prepared_store) -> None:
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    service = ReviewService(store, FakeOllamaClient())
    snapshot = service.generate_pair(item, "code")
    service.save_decision(
        item=item,
        decision="prefer_a",
        idempotency_key=decision_idempotency_key(item.id, "prefer_a", snapshot.id),
    )
    with pytest.raises(ValueError, match="immutable decision"):
        service.generate_pair(item, "new code", replace_snapshot_id=snapshot.id)


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


def _snapshot_seeds(store, snapshot_id: int) -> tuple[int, int]:
    with store.connection() as connection:
        row = connection.execute(
            "SELECT seed_1, seed_2 FROM generation_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
    assert row is not None
    return int(row["seed_1"]), int(row["seed_2"])

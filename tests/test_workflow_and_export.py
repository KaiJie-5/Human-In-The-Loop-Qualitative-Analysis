from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hitl_qualitative.database import MIGRATION_1, MIGRATION_2, QuestionDraft, SQLiteStore
from hitl_qualitative.exporting import PreferenceExporter, validate_conversation_row
from hitl_qualitative.ollama_client import OllamaConnectionError
from hitl_qualitative.workflow import ReviewService, segment_idempotency_key

from conftest import FakeOllamaClient, valid_candidate


def _item(prepared_store):
    store, _, dataset_id = prepared_store
    item = store.get_next_item(dataset_id)
    assert item is not None
    return store, dataset_id, item


def _generated_code(service: ReviewService, item, label: str = "checking bills twice"):
    code_id = service.add_code(item, label)
    service.generate_pending_codes(item)
    code = next(code for code in service.list_code_reviews(item) if code.id == code_id)
    assert code.snapshot is not None
    return code


def _save_draft(service: ReviewService, item, code, decision: str = "prefer_a", **categories):
    assert code.snapshot is not None
    candidates = {candidate.display_label: candidate for candidate in code.snapshot.candidates}
    service.save_code_draft(
        item=item,
        code_review_id=code.id,
        snapshot_id=code.snapshot.id,
        decision=decision,
        category_a_id=categories.get("category_a", candidates["A"].model_category_id),
        category_b_id=categories.get("category_b", candidates["B"].model_category_id),
        reason="saved draft",
        issue_tags=("Reflective question",),
    )


def _finish(service: ReviewService, item) -> int:
    return service.finalize_segment(
        item,
        idempotency_key=segment_idempotency_key(item.id, item.reviewer_id, "complete"),
    )


def test_two_seeds_are_distinct_with_identical_prompt_and_options(prepared_store) -> None:
    store, _, item = _item(prepared_store)
    fake = FakeOllamaClient()
    code = _generated_code(ReviewService(store, fake), item)
    assert code.snapshot is not None and code.snapshot.status == "ready"
    assert len(fake.calls) == 2
    assert fake.calls[0]["seed"] != fake.calls[1]["seed"]
    for field in ("model", "prompt", "schema", "options"):
        assert fake.calls[0][field] == fake.calls[1][field]
    assert fake.calls[0]["options"]["num_ctx"] == 65536
    with store.connection() as connection:
        rows = connection.execute(
            "SELECT raw_response_json, parsed_json, rendered_text FROM candidates"
        ).fetchall()
    assert len(rows) == 2
    assert all(row["raw_response_json"] and row["parsed_json"] and row["rendered_text"] for row in rows)


def test_multiple_question_order_edits_and_snapshots_are_durable(prepared_store) -> None:
    store, study_id, dataset_id = prepared_store
    first = store.get_questions(study_id)[0]
    store.save_questions(
        study_id,
        [
            QuestionDraft(int(first["id"]), str(first["text"]), selected=False),
            QuestionDraft(None, "What strategies do people use when billing is unclear?", True),
        ],
    )
    item = store.get_next_item(dataset_id)
    assert item is not None
    code = _generated_code(ReviewService(store, FakeOllamaClient()), item)
    assert code.snapshot is not None
    assert [question.text for question in code.snapshot.questions] == [
        "What strategies do people use when billing is unclear?"
    ]
    rows = store.get_questions(study_id)
    store.save_questions(
        study_id,
        [
            QuestionDraft(int(rows[1]["id"]), str(rows[1]["text"]), True),
            QuestionDraft(int(rows[0]["id"]), "How does uncertainty affect checking?", True),
        ],
    )
    assert store.get_questions(study_id)[1]["version"] == 2
    assert code.snapshot.questions[0].text == "What strategies do people use when billing is unclear?"


def test_multiple_codes_are_ordered_unique_lock_independently_and_allow_later_additions(
    prepared_store,
) -> None:
    store, _, item = _item(prepared_store)
    service = ReviewService(
        store,
        FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 3),
    )
    first_id = service.add_code(item, "  First code  ")
    second_id = service.add_code(item, "Second code")
    with pytest.raises(ValueError, match="already contains"):
        service.add_code(item, "Second code ")
    case_sensitive_id = service.add_code(item, "second code")
    service.remove_code(item, case_sensitive_id)
    service.generate_pending_codes(item)
    codes = service.list_code_reviews(item)
    assert [(code.ordinal, code.code_label) for code in codes] == [
        (1, "  First code  "), (2, "Second code")
    ]
    assert codes[0].snapshot is not None
    assert codes[0].snapshot.code_label == "  First code  "
    assert "\n  First code  \n" in next(
        call["prompt"] for call in service.ollama.calls if "  First code  " in call["prompt"]
    )
    with pytest.raises(ValueError, match="locked"):
        service.update_code(item, first_id, "Changed")
    with pytest.raises(ValueError, match="cannot be removed"):
        service.remove_code(item, second_id)
    third_id = service.add_code(item, "Third code")
    service.generate_pending_codes(item)
    assert [code.id for code in service.list_code_reviews(item)] == [first_id, second_id, third_id]


def test_sequential_batch_recovery_does_not_repeat_completed_candidates(prepared_store) -> None:
    store, _, item = _item(prepared_store)
    interrupted = FakeOllamaClient(
        [valid_candidate(), valid_candidate("too_broad"), valid_candidate(), OllamaConnectionError("offline")]
    )
    service = ReviewService(store, interrupted)
    service.add_code(item, "First")
    second_id = service.add_code(item, "Second")
    with pytest.raises(OllamaConnectionError):
        service.generate_pending_codes(item)
    first = service.list_code_reviews(item)[0]
    assert first.snapshot is not None and first.snapshot.status == "ready"
    resumed = ReviewService(store, FakeOllamaClient([valid_candidate("wrong_code")]))
    recovered = resumed.resume_pending_generation(item, second_id)
    assert recovered.status == "ready"
    assert len(resumed.ollama.calls) == 1


def test_autosaved_category_overrides_recover_and_export_per_code(prepared_store, tmp_path: Path) -> None:
    store, dataset_id, item = _item(prepared_store)
    service = ReviewService(store, FakeOllamaClient())
    code = _generated_code(service, item)
    _save_draft(
        service, item, code,
        category_a="wrong_code", category_b="useful_analytical_code",
    )
    recovered = ReviewService(store, FakeOllamaClient()).list_code_reviews(item)[0]
    assert recovered.draft.category_a_id == "wrong_code"
    assert recovered.draft.category_b_id == "useful_analytical_code"
    first_completion = _finish(service, item)
    assert _finish(service, item) == first_completion

    exporter = PreferenceExporter(
        store,
        tmp_path / "exports",
        clock=lambda: datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
    )
    result = exporter.export(dataset_id)
    row = json.loads(result.jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    validate_conversation_row(row)
    for field in ("chosen", "rejected"):
        assert [line.split(":", 1)[0] for line in row[field][0]["content"].splitlines()] == [
            "Code category", "Reflective question"
        ]
    chosen = next(
        candidate for candidate in code.snapshot.candidates if candidate.display_label == "A"
    )
    assert row["chosen"][0]["content"].startswith("Code category: Wrong code\n")
    assert chosen.reflective_question in row["chosen"][0]["content"]
    assert "reviewer-01" not in result.jsonl_path.read_text(encoding="utf-8")
    assert "reviewer-01" not in result.manifest_path.read_text(encoding="utf-8")


def test_one_segment_exports_one_preference_row_per_preferred_code(
    prepared_store, tmp_path: Path
) -> None:
    store, dataset_id, item = _item(prepared_store)
    service = ReviewService(
        store,
        FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 2),
    )
    first_id = service.add_code(item, "First exact code")
    second_id = service.add_code(item, "Second exact code")
    service.generate_pending_codes(item)
    codes = {code.id: code for code in service.list_code_reviews(item)}
    _save_draft(
        service, item, codes[first_id], decision="prefer_a",
        category_a="wrong_code", category_b="too_broad",
    )
    _save_draft(
        service, item, codes[second_id], decision="prefer_b",
        category_a="descriptive_not_answering_rq",
        category_b="useful_analytical_code",
    )
    _finish(service, item)

    result = PreferenceExporter(
        store,
        tmp_path / "exports",
        clock=lambda: datetime(2026, 8, 13, 12, 1, tzinfo=timezone.utc),
    ).export(dataset_id)
    rows = [json.loads(line) for line in result.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert result.row_count == 2
    assert "\nFirst exact code\n" in rows[0]["prompt"][0]["content"]
    assert "\nSecond exact code\n" in rows[1]["prompt"][0]["content"]
    assert rows[0]["chosen"][0]["content"].startswith("Code category: Wrong code\n")
    assert rows[1]["chosen"][0]["content"].startswith(
        "Code category: Useful analytical code\n"
    )


def test_finish_is_atomic_and_requires_every_code_draft(prepared_store) -> None:
    store, _, item = _item(prepared_store)
    service = ReviewService(
        store,
        FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 2),
    )
    service.add_code(item, "First")
    service.add_code(item, "Second")
    service.generate_pending_codes(item)
    codes = service.list_code_reviews(item)
    _save_draft(service, item, codes[0])
    with pytest.raises(ValueError, match="every code"):
        _finish(service, item)
    with store.connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM code_decisions").fetchone()[0] == 0
    _save_draft(service, item, codes[1], decision="too_similar")
    _finish(service, item)
    assert store.get_next_item(item.dataset_id) is None
    with pytest.raises(ValueError, match="immutable"):
        service.add_code(item, "Too late")


def test_more_than_three_per_code_regenerations_replace_only_that_pair(prepared_store) -> None:
    store, _, item = _item(prepared_store)
    fake = FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 7)
    service = ReviewService(store, fake)
    first = _generated_code(service, item, "First")
    second_id = service.add_code(item, "Second")
    service.generate_pending_codes(item)
    second_snapshot = next(
        code.snapshot.id for code in service.list_code_reviews(item) if code.id == second_id
    )
    current_id = first.snapshot.id
    for _ in range(5):
        replacement = service.regenerate_code(item, first.id)
        assert replacement.status == "ready"
        assert replacement.id != current_id
        current_id = replacement.id
    codes = service.list_code_reviews(item)
    assert next(code.snapshot.id for code in codes if code.id == second_id) == second_snapshot
    with store.connection() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM generation_snapshots WHERE code_review_id = ?", (first.id,)
        ).fetchone()[0] == 1


def test_failed_and_invalid_regeneration_keep_prior_pair_and_draft(prepared_store) -> None:
    store, _, item = _item(prepared_store)
    service = ReviewService(store, FakeOllamaClient())
    code = _generated_code(service, item)
    _save_draft(service, item, code, category_a="wrong_code")
    old_snapshot = code.snapshot.id

    interrupted = ReviewService(
        store, FakeOllamaClient([valid_candidate(), OllamaConnectionError("offline")])
    )
    with pytest.raises(OllamaConnectionError):
        interrupted.regenerate_code(item, code.id)
    during = interrupted.list_code_reviews(item)[0]
    assert during.snapshot.id == old_snapshot
    assert during.draft.category_a_id == "wrong_code"
    assert during.replacement_in_progress

    resumed = ReviewService(store, FakeOllamaClient([valid_candidate("too_broad")]))
    resumed.resume_pending_generation(item, code.id)
    replaced = resumed.list_code_reviews(item)[0]
    assert replaced.snapshot.id != old_snapshot
    assert replaced.draft.decision is None

    invalid = ReviewService(store, FakeOllamaClient([{}, {}, {}, {}]))
    current_id = replaced.snapshot.id
    returned = invalid.regenerate_code(item, code.id)
    assert returned.status == "abandoned"
    restored = invalid.list_code_reviews(item)[0]
    assert restored.snapshot.id == current_id


def test_pre_generation_segment_skip_and_non_preferences_are_excluded(
    prepared_store, tmp_path: Path
) -> None:
    store, dataset_id, item = _item(prepared_store)
    service = ReviewService(store, FakeOllamaClient())
    key = segment_idempotency_key(item.id, item.reviewer_id, "skip")
    first = service.skip_segment(item, reason="Not useful", idempotency_key=key)
    assert service.skip_segment(item, reason="Not useful", idempotency_key=key) == first
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {"segment_skip": 1}


@pytest.mark.parametrize("decision", ["both_poor", "too_similar", "skip"])
def test_each_non_preference_code_decision_is_excluded(
    prepared_store, tmp_path: Path, decision: str
) -> None:
    store, dataset_id, item = _item(prepared_store)
    service = ReviewService(store, FakeOllamaClient())
    code = _generated_code(service, item)
    _save_draft(service, item, code, decision=decision)
    _finish(service, item)
    preview = PreferenceExporter(store, tmp_path / "exports").preview(dataset_id)
    assert preview.eligible_count == 0
    assert preview.exclusion_counts == {decision: 1}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_historical_validation_and_test_datasets_remain_blocked(
    prepared_store, tmp_path: Path, split: str
) -> None:
    store, study_id, _ = prepared_store
    from conftest import segment_payload
    from hitl_qualitative.transcripts import TranscriptAdapter

    data = (json.dumps(segment_payload(transcript_id="INT999", record_id="INT999_SEG001")) + "\n").encode()
    bundle = TranscriptAdapter().from_upload(f"{split}.jsonl", data)
    dataset_id, _ = store.import_dataset(
        study_id=study_id, name=split.title(), split=split, source_kind="upload", bundle=bundle
    )
    exporter = PreferenceExporter(store, tmp_path / "exports")
    assert exporter.preview(dataset_id).exclusion_counts == {"non_adaptation_split": 1}
    with pytest.raises(ValueError, match="Only adaptation"):
        exporter.export(dataset_id)


def test_singleton_initialization_and_multiple_study_guard(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "singleton.sqlite3")
    store.initialize()
    assert store.get_singleton_study() is None
    study_id = store.create_singleton_study(
        reviewer_id="staff-1", ollama_base_url="http://localhost:11434"
    )
    assert store.get_singleton_study()["id"] == study_id
    store.create_study(
        name="Unexpected second study", reviewer_id="staff-2",
        ollama_base_url="http://localhost:11434",
    )
    with pytest.raises(RuntimeError, match="multiple studies"):
        store.get_singleton_study()


def test_schema_v2_multiple_studies_stops_before_migration(tmp_path: Path) -> None:
    path = tmp_path / "legacy-multiple.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATION_1)
    connection.executescript(MIGRATION_2)
    connection.execute("PRAGMA user_version = 2")
    now = "2026-08-01T00:00:00+00:00"
    connection.executemany(
        """
        INSERT INTO studies(id,name,reviewer_id,ollama_base_url,created_at,updated_at)
        VALUES (?,?,?,?,?,?)
        """,
        [
            (1, "First", "staff-1", "http://localhost:11434", now, now),
            (2, "Second", "staff-2", "http://localhost:11434", now, now),
        ],
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="stopped before making any changes"):
        SQLiteStore(path).initialize()
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    assert connection.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'code_reviews'"
    ).fetchone()[0] == 0
    connection.close()


def test_progress_reports_segment_and_code_levels(prepared_store) -> None:
    store, dataset_id, item = _item(prepared_store)
    service = ReviewService(
        store,
        FakeOllamaClient([valid_candidate(), valid_candidate("too_broad")] * 2),
    )
    service.add_code(item, "First")
    service.add_code(item, "Second")
    service.generate_pending_codes(item)
    codes = service.list_code_reviews(item)
    _save_draft(service, item, codes[0], decision="prefer_a")
    _save_draft(service, item, codes[1], decision="both_poor")
    _finish(service, item)
    progress = store.progress(dataset_id)
    assert progress["reviewed"] == 1
    assert progress["segment_completed"] == 1
    assert progress["code_total"] == 2
    assert progress["preferred"] == 1
    assert progress["both_poor"] == 1


def test_schema_v2_migration_backfills_historical_generation_and_decision(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(MIGRATION_1)
    connection.executescript(MIGRATION_2)
    connection.execute("PRAGMA user_version = 2")
    now = "2026-08-01T00:00:00+00:00"
    connection.execute(
        """
        INSERT INTO studies(id,name,reviewer_id,ollama_base_url,created_at,updated_at)
        VALUES (1,'Legacy','reviewer','http://localhost:11434',?,?)
        """, (now, now),
    )
    connection.execute(
        """
        INSERT INTO datasets(
            id,study_id,name,split,source_kind,source_locator,source_files_json,
            source_sha256,transcript_count,target_count,created_at
        ) VALUES (1,1,'Data','adaptation','upload','legacy','[]','sha',1,1,?)
        """, (now,),
    )
    connection.execute("INSERT INTO transcripts(id,dataset_id,transcript_id,source_order) VALUES (1,1,'INT',1)")
    connection.execute(
        """
        INSERT INTO review_items(
            id,dataset_id,transcript_pk,record_id,segment_id,speaker,target_text,
            turn_index,target_turn_indexes_json,source_order,source_metadata_json,status,updated_at
        ) VALUES (1,1,1,'R1','S1','participant','text',1,'[1]',1,'{}','decided',?)
        """, (now,),
    )
    connection.execute(
        """
        INSERT INTO generation_snapshots(
            id,review_item_id,reviewer_id,attempt_number,input_fingerprint,status,code_label,
            requested_context_before,requested_context_after,symmetric_context,prompt_version,
            category_version,exact_prompt,prompt_sha256,model_name,model_digest,ollama_base_url,
            options_json,seed_1,seed_2,created_at,updated_at
        ) VALUES (1,1,'reviewer',1,'fp','ready','legacy code',0,0,1,'v3','v3',
                  'prompt','hash','model','digest','url','{}',1,2,?,?)
        """, (now, now),
    )
    for number, category in ((1, "wrong_code"), (2, "too_broad")):
        payload = valid_candidate(category)
        rendered = (
            f"Code category: {'Wrong code' if number == 1 else 'Too broad code'}\n"
            f"Reflective question: {payload['reflective_question']}"
        )
        connection.execute(
            """
            INSERT INTO candidates(
                id,snapshot_id,candidate_number,seed,parsed_json,rendered_text,
                rendered_sha256,valid,validation_errors_json,created_at,updated_at
            ) VALUES (?,?,?,?,?,?,?,1,'[]',?,?)
            """,
            (number, 1, number, number, json.dumps(payload), rendered, __import__('hashlib').sha256(rendered.encode()).hexdigest(), now, now),
        )
    connection.execute("INSERT INTO ab_assignments VALUES (1,'A',1)")
    connection.execute("INSERT INTO ab_assignments VALUES (1,'B',2)")
    connection.execute(
        """
        INSERT INTO decisions(
            review_item_id,snapshot_id,reviewer_id,decision,preferred_candidate_id,
            issue_tags_json,idempotency_key,created_at
        ) VALUES (1,1,'reviewer','prefer_a',1,'[]','legacy-key',?)
        """, (now,),
    )
    connection.commit()
    connection.close()

    SQLiteStore(path).initialize()
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert connection.execute("SELECT COUNT(*) FROM code_reviews").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM code_decisions").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM code_decision_categories").fetchone()[0] == 2
    assert connection.execute("SELECT COUNT(*) FROM segment_completions").fetchone()[0] == 1
    connection.close()

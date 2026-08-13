from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .categories import CATEGORY_SPECS
from .transcripts import ImportBundle, TranscriptTurn


SCHEMA_VERSION = 3
SPLITS = {"adaptation", "validation", "test"}


MIGRATION_1 = """
CREATE TABLE studies (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    reviewer_locked INTEGER NOT NULL DEFAULT 0 CHECK (reviewer_locked IN (0, 1)),
    ollama_base_url TEXT NOT NULL,
    model_name TEXT,
    model_digest TEXT,
    context_before INTEGER NOT NULL DEFAULT 20 CHECK (context_before >= 0),
    context_after INTEGER NOT NULL DEFAULT 20 CHECK (context_after >= 0),
    symmetric_context INTEGER NOT NULL DEFAULT 1 CHECK (symmetric_context IN (0, 1)),
    temperature REAL NOT NULL DEFAULT 0.4,
    top_p REAL NOT NULL DEFAULT 0.9,
    output_tokens INTEGER NOT NULL DEFAULT 5000 CHECK (output_tokens > 0),
    active_dataset_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('adaptation', 'validation', 'test')),
    source_kind TEXT NOT NULL CHECK (source_kind IN ('path', 'upload')),
    source_locator TEXT NOT NULL,
    source_files_json TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    transcript_count INTEGER NOT NULL,
    target_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(study_id, source_sha256, split),
    UNIQUE(study_id, name)
);

CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    transcript_id TEXT NOT NULL,
    source_order INTEGER NOT NULL,
    UNIQUE(dataset_id, transcript_id),
    UNIQUE(dataset_id, source_order)
);

CREATE TABLE transcript_turns (
    id INTEGER PRIMARY KEY,
    transcript_pk INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    speaker_label TEXT,
    text TEXT NOT NULL,
    paragraph_index INTEGER NOT NULL,
    UNIQUE(transcript_pk, turn_index)
);

CREATE TABLE review_items (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    transcript_pk INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    record_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    target_text TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    target_turn_indexes_json TEXT NOT NULL,
    source_order INTEGER NOT NULL,
    source_metadata_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'unreviewed'
        CHECK (status IN ('unreviewed', 'generating', 'generated', 'invalid', 'decided')),
    updated_at TEXT NOT NULL,
    UNIQUE(dataset_id, record_id),
    UNIQUE(dataset_id, source_order)
);

CREATE TABLE research_questions (
    id INTEGER PRIMARY KEY,
    study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    stable_key TEXT NOT NULL,
    display_order INTEGER NOT NULL,
    selected INTEGER NOT NULL DEFAULT 1 CHECK (selected IN (0, 1)),
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(study_id, stable_key),
    UNIQUE(study_id, display_order)
);

CREATE TABLE research_question_versions (
    id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES research_questions(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(question_id, version)
);

CREATE TABLE generation_snapshots (
    id INTEGER PRIMARY KEY,
    review_item_id INTEGER NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    input_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('generating', 'ready', 'invalid', 'superseded', 'abandoned')),
    code_label TEXT NOT NULL,
    requested_context_before INTEGER NOT NULL,
    requested_context_after INTEGER NOT NULL,
    symmetric_context INTEGER NOT NULL CHECK (symmetric_context IN (0, 1)),
    prompt_version TEXT NOT NULL,
    category_version TEXT NOT NULL,
    exact_prompt TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    ollama_base_url TEXT NOT NULL,
    options_json TEXT NOT NULL,
    seed_1 INTEGER NOT NULL,
    seed_2 INTEGER NOT NULL,
    superseded_by INTEGER REFERENCES generation_snapshots(id),
    errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (seed_1 <> seed_2),
    UNIQUE(review_item_id, reviewer_id, attempt_number)
);

CREATE UNIQUE INDEX one_live_snapshot_per_item
ON generation_snapshots(review_item_id, reviewer_id)
WHERE status IN ('generating', 'ready', 'invalid');

CREATE TABLE snapshot_context (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES generation_snapshots(id) ON DELETE CASCADE,
    side TEXT NOT NULL CHECK (side IN ('previous', 'target', 'next')),
    position INTEGER NOT NULL,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    speaker_label TEXT,
    text TEXT NOT NULL,
    UNIQUE(snapshot_id, side, position)
);

CREATE TABLE snapshot_questions (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES generation_snapshots(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    question_id INTEGER NOT NULL REFERENCES research_questions(id),
    question_version INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(snapshot_id, ordinal),
    UNIQUE(snapshot_id, question_id)
);

CREATE TABLE candidate_calls (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES generation_snapshots(id) ON DELETE CASCADE,
    candidate_number INTEGER NOT NULL CHECK (candidate_number IN (1, 2)),
    call_number INTEGER NOT NULL CHECK (call_number IN (1, 2)),
    call_kind TEXT NOT NULL CHECK (call_kind IN ('initial', 'repair')),
    seed INTEGER NOT NULL,
    request_prompt TEXT NOT NULL,
    request_json TEXT NOT NULL,
    raw_response TEXT,
    transport_error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(snapshot_id, candidate_number, call_number)
);

CREATE TABLE candidates (
    id INTEGER PRIMARY KEY,
    snapshot_id INTEGER NOT NULL REFERENCES generation_snapshots(id) ON DELETE CASCADE,
    candidate_number INTEGER NOT NULL CHECK (candidate_number IN (1, 2)),
    seed INTEGER NOT NULL,
    raw_response_json TEXT,
    parsed_json TEXT,
    rendered_text TEXT,
    rendered_sha256 TEXT,
    valid INTEGER NOT NULL CHECK (valid IN (0, 1)),
    validation_errors_json TEXT NOT NULL,
    response_model TEXT,
    response_metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(snapshot_id, candidate_number)
);

CREATE TABLE ab_assignments (
    snapshot_id INTEGER NOT NULL REFERENCES generation_snapshots(id) ON DELETE CASCADE,
    display_label TEXT NOT NULL CHECK (display_label IN ('A', 'B')),
    candidate_id INTEGER NOT NULL REFERENCES candidates(id) ON DELETE CASCADE,
    PRIMARY KEY(snapshot_id, display_label),
    UNIQUE(snapshot_id, candidate_id)
);

CREATE TABLE decisions (
    id INTEGER PRIMARY KEY,
    review_item_id INTEGER NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    snapshot_id INTEGER REFERENCES generation_snapshots(id),
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('prefer_a', 'prefer_b', 'both_poor', 'too_similar', 'skip')
    ),
    preferred_candidate_id INTEGER REFERENCES candidates(id),
    reason TEXT,
    issue_tags_json TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(review_item_id, reviewer_id),
    UNIQUE(idempotency_key)
);

CREATE TABLE exports (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    jsonl_path TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    jsonl_sha256 TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    exclusion_counts_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(jsonl_path),
    UNIQUE(manifest_path)
);

CREATE INDEX review_items_queue ON review_items(dataset_id, status, source_order);
CREATE INDEX snapshots_item ON generation_snapshots(review_item_id, attempt_number);
CREATE INDEX decisions_dataset ON decisions(review_item_id, decision);
"""

MIGRATION_2 = """
ALTER TABLE studies
ADD COLUMN context_tokens INTEGER NOT NULL DEFAULT 65536 CHECK (context_tokens > 0);
"""


MIGRATION_3 = """
CREATE TABLE code_reviews (
    id INTEGER PRIMARY KEY,
    review_item_id INTEGER NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal > 0),
    code_label TEXT NOT NULL,
    dedupe_label TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft' CHECK (
        status IN ('draft', 'generating', 'ready', 'invalid', 'finalized', 'abandoned')
    ),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(review_item_id, reviewer_id, ordinal),
    UNIQUE(review_item_id, reviewer_id, dedupe_label)
);

ALTER TABLE generation_snapshots
ADD COLUMN code_review_id INTEGER REFERENCES code_reviews(id) ON DELETE CASCADE;

DROP INDEX one_live_snapshot_per_item;

CREATE UNIQUE INDEX one_live_snapshot_per_code
ON generation_snapshots(code_review_id)
WHERE code_review_id IS NOT NULL
  AND status IN ('generating', 'ready', 'invalid');

CREATE INDEX code_reviews_item
ON code_reviews(review_item_id, reviewer_id, ordinal);

CREATE TABLE code_review_drafts (
    code_review_id INTEGER PRIMARY KEY REFERENCES code_reviews(id) ON DELETE CASCADE,
    snapshot_id INTEGER REFERENCES generation_snapshots(id) ON DELETE SET NULL,
    decision TEXT CHECK (
        decision IS NULL OR decision IN ('prefer_a', 'prefer_b', 'both_poor', 'too_similar', 'skip')
    ),
    category_a_id TEXT CHECK (
        category_a_id IS NULL OR category_a_id IN (
            'wrong_code', 'descriptive_not_answering_rq', 'too_broad', 'useful_analytical_code'
        )
    ),
    category_b_id TEXT CHECK (
        category_b_id IS NULL OR category_b_id IN (
            'wrong_code', 'descriptive_not_answering_rq', 'too_broad', 'useful_analytical_code'
        )
    ),
    reason TEXT,
    issue_tags_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);

CREATE TABLE code_decisions (
    id INTEGER PRIMARY KEY,
    code_review_id INTEGER NOT NULL REFERENCES code_reviews(id) ON DELETE CASCADE,
    snapshot_id INTEGER REFERENCES generation_snapshots(id),
    reviewer_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('prefer_a', 'prefer_b', 'both_poor', 'too_similar', 'skip')
    ),
    preferred_candidate_id INTEGER REFERENCES candidates(id),
    reason TEXT,
    issue_tags_json TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(code_review_id),
    UNIQUE(idempotency_key)
);

CREATE TABLE code_decision_categories (
    decision_id INTEGER NOT NULL REFERENCES code_decisions(id) ON DELETE CASCADE,
    display_label TEXT NOT NULL CHECK (display_label IN ('A', 'B')),
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    category_id TEXT NOT NULL CHECK (
        category_id IN (
            'wrong_code', 'descriptive_not_answering_rq', 'too_broad', 'useful_analytical_code'
        )
    ),
    PRIMARY KEY(decision_id, display_label),
    UNIQUE(decision_id, candidate_id)
);

CREATE TABLE segment_completions (
    id INTEGER PRIMARY KEY,
    review_item_id INTEGER NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    reviewer_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('completed', 'skipped')),
    reason TEXT,
    issue_tags_json TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(review_item_id, reviewer_id),
    UNIQUE(idempotency_key)
);

CREATE INDEX code_decisions_dataset ON code_decisions(code_review_id, decision);
"""


@dataclass(frozen=True, slots=True)
class QuestionDraft:
    id: int | None
    text: str
    selected: bool = True
    active: bool = True


@dataclass(frozen=True, slots=True)
class ReviewItem:
    id: int
    dataset_id: int
    dataset_name: str
    split: str
    study_id: int
    reviewer_id: str
    transcript_pk: int
    transcript_id: str
    record_id: str
    segment_id: str
    target_text: str
    target_turn_indexes: tuple[int, ...]
    status: str
    turns: tuple[TranscriptTurn, ...]


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema {current} is newer than supported {SCHEMA_VERSION}."
                )
            if current < 1:
                connection.executescript(MIGRATION_1)
                connection.execute("PRAGMA user_version = 1")
                current = 1
            if current < 2:
                connection.executescript(MIGRATION_2)
                connection.execute("PRAGMA user_version = 2")
                current = 2
            if current < 3:
                study_count = int(
                    connection.execute("SELECT COUNT(*) FROM studies").fetchone()[0]
                )
                if study_count > 1:
                    raise RuntimeError(
                        "Schema version 2 contains multiple studies. Migration to the "
                        "singleton interface was stopped before making any changes; use a "
                        "separate database or consolidate the studies first. No study was "
                        "selected or deleted."
                    )
                connection.executescript(MIGRATION_3)
                _backfill_multi_code_schema(connection)
                connection.execute("PRAGMA user_version = 3")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def create_study(
        self,
        *,
        name: str,
        reviewer_id: str,
        ollama_base_url: str,
        context_before: int = 20,
        context_after: int = 20,
        temperature: float = 0.4,
        top_p: float = 0.9,
        output_tokens: int = 5000,
        context_tokens: int = 65536,
    ) -> int:
        if not name.strip() or not reviewer_id.strip():
            raise ValueError("Study name and reviewer ID are required.")
        if context_tokens <= 0 or output_tokens <= 0 or output_tokens >= context_tokens:
            raise ValueError("Context tokens must be positive and greater than output tokens.")
        now = utc_now()
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO studies(
                    name, reviewer_id, ollama_base_url, context_before, context_after,
                    temperature, top_p, output_tokens, context_tokens, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name.strip(), reviewer_id.strip(), ollama_base_url.rstrip("/"),
                    context_before, context_after, temperature, top_p, output_tokens,
                    context_tokens,
                    now, now,
                ),
            )
            return int(cursor.lastrowid)

    def list_studies(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM studies ORDER BY id")]

    def get_singleton_study(self) -> dict[str, Any] | None:
        studies = self.list_studies()
        if len(studies) > 1:
            raise RuntimeError(
                "This database contains multiple studies. The simplified interface will not "
                "choose between them automatically; use a separate database or consolidate "
                "the studies before continuing. No study was selected or deleted."
            )
        return studies[0] if studies else None

    def create_singleton_study(
        self,
        *,
        reviewer_id: str,
        ollama_base_url: str,
        context_before: int = 20,
        context_after: int = 20,
        temperature: float = 0.4,
        top_p: float = 0.9,
        output_tokens: int = 5000,
        context_tokens: int = 65536,
    ) -> int:
        if self.get_singleton_study() is not None:
            raise ValueError("The local study has already been initialized.")
        return self.create_study(
            name="Local qualitative analysis",
            reviewer_id=reviewer_id,
            ollama_base_url=ollama_base_url,
            context_before=context_before,
            context_after=context_after,
            temperature=temperature,
            top_p=top_p,
            output_tokens=output_tokens,
            context_tokens=context_tokens,
        )

    def get_study(self, study_id: int) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown study ID {study_id}.")
        return dict(row)

    def update_study(
        self,
        study_id: int,
        *,
        reviewer_id: str,
        ollama_base_url: str,
        model_name: str | None,
        model_digest: str | None,
        context_before: int,
        context_after: int,
        symmetric_context: bool,
        temperature: float,
        top_p: float,
        output_tokens: int,
        context_tokens: int,
    ) -> None:
        if context_tokens <= 0 or output_tokens <= 0 or output_tokens >= context_tokens:
            raise ValueError("Context tokens must be positive and greater than output tokens.")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM studies WHERE id = ?", (study_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown study ID {study_id}.")
            normalized_reviewer = reviewer_id.strip()
            if not normalized_reviewer:
                raise ValueError("Reviewer ID is required.")
            if row["reviewer_locked"] and row["reviewer_id"] != normalized_reviewer:
                raise ValueError("Reviewer ID is locked because review work already exists.")
            changed_generation_settings = any(
                (
                    row["ollama_base_url"] != ollama_base_url.rstrip("/"),
                    row["model_name"] != model_name,
                    row["model_digest"] != model_digest,
                    int(row["context_before"]) != context_before,
                    int(row["context_after"]) != context_after,
                    bool(row["symmetric_context"]) != symmetric_context,
                    float(row["temperature"]) != temperature,
                    float(row["top_p"]) != top_p,
                    int(row["output_tokens"]) != output_tokens,
                    int(row["context_tokens"]) != context_tokens,
                )
            )
            now = utc_now()
            connection.execute(
                """
                UPDATE studies SET reviewer_id = ?, ollama_base_url = ?, model_name = ?,
                    model_digest = ?, context_before = ?, context_after = ?,
                    symmetric_context = ?, temperature = ?, top_p = ?, output_tokens = ?,
                    context_tokens = ?, updated_at = ? WHERE id = ?
                """,
                (
                    normalized_reviewer, ollama_base_url.rstrip("/"), model_name,
                    model_digest, context_before, context_after, int(symmetric_context),
                    temperature, top_p, output_tokens, context_tokens, now, study_id,
                ),
            )
            if changed_generation_settings:
                _supersede_live_snapshots_for_study(connection, study_id, now)

    def save_questions(self, study_id: int, questions: Sequence[QuestionDraft]) -> None:
        active = [question for question in questions if question.active]
        if not active or not any(question.selected for question in active):
            raise ValueError("At least one active research question must be selected.")
        if any(not question.text.strip() for question in active):
            raise ValueError("Research questions must not be empty.")
        now = utc_now()
        with self.transaction() as connection:
            before_signature = _question_signature(connection, study_id)
            existing_ids = {
                int(row[0])
                for row in connection.execute(
                    "SELECT id FROM research_questions WHERE study_id = ?", (study_id,)
                )
            }
            submitted_ids = {question.id for question in questions if question.id is not None}
            unknown = submitted_ids - existing_ids
            if unknown:
                raise ValueError(f"Unknown research question IDs: {sorted(unknown)}.")
            connection.execute(
                "UPDATE research_questions SET display_order = -id WHERE study_id = ?",
                (study_id,),
            )
            for order, draft in enumerate(questions, 1):
                text = draft.text
                if draft.id is None:
                    stable_key = _stable_key(study_id, order, now, text)
                    cursor = connection.execute(
                        """
                        INSERT INTO research_questions(
                            study_id, stable_key, display_order, selected, active,
                            current_version, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (study_id, stable_key, order, int(draft.selected), int(draft.active), now, now),
                    )
                    question_id = int(cursor.lastrowid)
                    connection.execute(
                        """
                        INSERT INTO research_question_versions(question_id, version, text, created_at)
                        VALUES (?, 1, ?, ?)
                        """,
                        (question_id, text, now),
                    )
                    continue
                current = connection.execute(
                    """
                    SELECT q.current_version, v.text
                    FROM research_questions q
                    JOIN research_question_versions v
                      ON v.question_id = q.id AND v.version = q.current_version
                    WHERE q.id = ? AND q.study_id = ?
                    """,
                    (draft.id, study_id),
                ).fetchone()
                if current is None:
                    raise ValueError(f"Unknown research question ID {draft.id}.")
                version = int(current["current_version"])
                if text != current["text"]:
                    version += 1
                    connection.execute(
                        """
                        INSERT INTO research_question_versions(question_id, version, text, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (draft.id, version, text, now),
                    )
                connection.execute(
                    """
                    UPDATE research_questions SET display_order = ?, selected = ?, active = ?,
                        current_version = ?, updated_at = ? WHERE id = ?
                    """,
                    (order, int(draft.selected), int(draft.active), version, now, draft.id),
                )
            removed = existing_ids - {value for value in submitted_ids if value is not None}
            if removed:
                placeholders = ",".join("?" for _ in removed)
                connection.execute(
                    f"UPDATE research_questions SET active = 0, selected = 0, updated_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (now, *sorted(removed)),
                )
            if _question_signature(connection, study_id) != before_signature:
                _supersede_live_snapshots_for_study(connection, study_id, now)

    def get_questions(self, study_id: int, *, selected_only: bool = False) -> list[dict[str, Any]]:
        where = "AND q.active = 1"
        if selected_only:
            where += " AND q.selected = 1"
        with self.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT q.id, q.stable_key, q.display_order, q.selected, q.active,
                       q.current_version AS version, v.text
                FROM research_questions q
                JOIN research_question_versions v
                  ON v.question_id = q.id AND v.version = q.current_version
                WHERE q.study_id = ? {where}
                ORDER BY q.display_order
                """,
                (study_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def import_dataset(
        self,
        *,
        study_id: int,
        name: str,
        split: str,
        source_kind: str,
        bundle: ImportBundle,
    ) -> tuple[int, bool]:
        if split not in SPLITS:
            raise ValueError(f"Unsupported data split {split!r}.")
        if source_kind not in {"path", "upload"}:
            raise ValueError(f"Unsupported source kind {source_kind!r}.")
        if not name.strip():
            raise ValueError("Dataset name is required.")
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM datasets WHERE study_id = ? AND source_sha256 = ? AND split = ?",
                (study_id, bundle.source_sha256, split),
            ).fetchone()
            if existing is not None:
                return int(existing["id"]), False
            now = utc_now()
            cursor = connection.execute(
                """
                INSERT INTO datasets(
                    study_id, name, split, source_kind, source_locator, source_files_json,
                    source_sha256, transcript_count, target_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    study_id, name.strip(), split, source_kind, bundle.source_locator,
                    _json(bundle.source_files), bundle.source_sha256, len(bundle.transcripts),
                    bundle.target_count, now,
                ),
            )
            dataset_id = int(cursor.lastrowid)
            for transcript_order, transcript in enumerate(bundle.transcripts, 1):
                cursor = connection.execute(
                    "INSERT INTO transcripts(dataset_id, transcript_id, source_order) VALUES (?, ?, ?)",
                    (dataset_id, transcript.transcript_id, transcript_order),
                )
                transcript_pk = int(cursor.lastrowid)
                connection.executemany(
                    """
                    INSERT INTO transcript_turns(
                        transcript_pk, turn_index, speaker, speaker_label, text, paragraph_index
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            transcript_pk, turn.turn_index, turn.speaker,
                            turn.speaker_label, turn.text, turn.paragraph_index,
                        )
                        for turn in transcript.turns
                    ],
                )
                for target in transcript.targets:
                    connection.execute(
                        """
                        INSERT INTO review_items(
                            dataset_id, transcript_pk, record_id, segment_id, speaker,
                            target_text, turn_index, target_turn_indexes_json, source_order,
                            source_metadata_json, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dataset_id, transcript_pk, target.record_id, target.segment_id,
                            target.speaker, target.text, target.turn_index,
                            _json(target.target_turn_indexes), target.source_order,
                            _json(target.metadata), now,
                        ),
                    )
            return dataset_id, True

    def import_adaptation_dataset(
        self,
        *,
        study_id: int,
        name: str,
        source_kind: str,
        bundle: ImportBundle,
    ) -> tuple[int, bool]:
        """Import a new UI dataset with the fixed training-eligible split."""
        return self.import_dataset(
            study_id=study_id,
            name=name,
            split="adaptation",
            source_kind=source_kind,
            bundle=bundle,
        )

    def list_datasets(self, study_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM datasets WHERE study_id = ? ORDER BY id", (study_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get_dataset(self, dataset_id: int) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown dataset ID {dataset_id}.")
        return dict(row)

    def set_active_dataset(self, study_id: int, dataset_id: int) -> None:
        with self.transaction() as connection:
            valid = connection.execute(
                "SELECT 1 FROM datasets WHERE id = ? AND study_id = ?",
                (dataset_id, study_id),
            ).fetchone()
            if valid is None:
                raise ValueError("The selected dataset does not belong to this study.")
            connection.execute(
                "UPDATE studies SET active_dataset_id = ?, updated_at = ? WHERE id = ?",
                (dataset_id, utc_now(), study_id),
            )

    def recover_active_dataset(self) -> int | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT s.active_dataset_id FROM studies s
                JOIN datasets d ON d.id = s.active_dataset_id AND d.study_id = s.id
                WHERE s.active_dataset_id IS NOT NULL
                ORDER BY s.updated_at DESC LIMIT 1
                """
            ).fetchone()
        return int(row[0]) if row is not None else None

    def get_next_item(self, dataset_id: int) -> ReviewItem | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT ri.*, d.name AS dataset_name, d.split, d.study_id,
                       s.reviewer_id, t.transcript_id
                FROM review_items ri
                JOIN datasets d ON d.id = ri.dataset_id
                JOIN studies s ON s.id = d.study_id
                JOIN transcripts t ON t.id = ri.transcript_pk
                LEFT JOIN segment_completions sc
                  ON sc.review_item_id = ri.id AND sc.reviewer_id = s.reviewer_id
                WHERE ri.dataset_id = ? AND sc.id IS NULL
                ORDER BY ri.source_order LIMIT 1
                """,
                (dataset_id,),
            ).fetchone()
            if row is None:
                return None
            turn_rows = connection.execute(
                "SELECT * FROM transcript_turns WHERE transcript_pk = ? ORDER BY turn_index",
                (row["transcript_pk"],),
            ).fetchall()
        turns = tuple(
            TranscriptTurn(
                turn_index=int(turn["turn_index"]), speaker=str(turn["speaker"]),
                speaker_label=turn["speaker_label"], text=str(turn["text"]),
                paragraph_index=int(turn["paragraph_index"]),
            )
            for turn in turn_rows
        )
        return ReviewItem(
            id=int(row["id"]), dataset_id=int(row["dataset_id"]),
            dataset_name=str(row["dataset_name"]), split=str(row["split"]),
            study_id=int(row["study_id"]), reviewer_id=str(row["reviewer_id"]),
            transcript_pk=int(row["transcript_pk"]), transcript_id=str(row["transcript_id"]),
            record_id=str(row["record_id"]), segment_id=str(row["segment_id"]),
            target_text=str(row["target_text"]),
            target_turn_indexes=tuple(json.loads(row["target_turn_indexes_json"])),
            status=str(row["status"]), turns=turns,
        )

    def progress(self, dataset_id: int) -> dict[str, int]:
        with self.connection() as connection:
            total = int(connection.execute(
                "SELECT COUNT(*) FROM review_items WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()[0])
            decision_counts = {
                str(row["decision"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT cd.decision, COUNT(*) AS count FROM code_decisions cd
                    JOIN code_reviews cr ON cr.id = cd.code_review_id
                    JOIN review_items ri ON ri.id = cr.review_item_id
                    WHERE ri.dataset_id = ? GROUP BY cd.decision
                    """,
                    (dataset_id,),
                )
            }
            segment_counts = {
                str(row["outcome"]): int(row["count"])
                for row in connection.execute(
                    """
                    SELECT sc.outcome, COUNT(*) AS count FROM segment_completions sc
                    JOIN review_items ri ON ri.id = sc.review_item_id
                    WHERE ri.dataset_id = ? GROUP BY sc.outcome
                    """,
                    (dataset_id,),
                )
            }
            reviewed = sum(segment_counts.values())
            code_total = int(connection.execute(
                """
                SELECT COUNT(*) FROM code_reviews cr
                JOIN review_items ri ON ri.id = cr.review_item_id
                WHERE ri.dataset_id = ? AND cr.status <> 'abandoned'
                """,
                (dataset_id,),
            ).fetchone()[0])
            generated = int(connection.execute(
                """
                SELECT COUNT(*) FROM code_reviews cr
                JOIN review_items ri ON ri.id = cr.review_item_id
                WHERE ri.dataset_id = ? AND cr.status IN ('ready', 'invalid', 'finalized')
                """,
                (dataset_id,),
            ).fetchone()[0])
            invalid = int(connection.execute(
                """
                SELECT COUNT(DISTINCT cr.id) FROM code_reviews cr
                JOIN review_items ri ON ri.id = cr.review_item_id
                LEFT JOIN code_decisions cd ON cd.code_review_id = cr.id
                LEFT JOIN code_review_drafts draft ON draft.code_review_id = cr.id
                JOIN candidates c
                  ON c.snapshot_id = COALESCE(cd.snapshot_id, draft.snapshot_id)
                WHERE ri.dataset_id = ? AND cr.status <> 'abandoned' AND c.valid = 0
                """,
                (dataset_id,),
            ).fetchone()[0])
        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": total - reviewed,
            "segment_completed": segment_counts.get("completed", 0),
            "generated": generated,
            "code_total": code_total,
            "code_unfinished": code_total - sum(decision_counts.values()),
            "preferred": decision_counts.get("prefer_a", 0) + decision_counts.get("prefer_b", 0),
            "both_poor": decision_counts.get("both_poor", 0),
            "too_similar": decision_counts.get("too_similar", 0),
            "skipped": decision_counts.get("skip", 0),
            "segment_skipped": segment_counts.get("skipped", 0),
            "invalid": invalid,
        }

    def list_records(self, dataset_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT ri.id, t.transcript_id, ri.segment_id, ri.record_id, ri.status,
                       sc.outcome AS segment_outcome, sc.created_at,
                       COUNT(DISTINCT cr.id) AS code_count,
                       COUNT(DISTINCT cd.id) AS code_decision_count,
                       SUM(CASE WHEN cd.decision IN ('prefer_a', 'prefer_b') THEN 1 ELSE 0 END)
                           AS preferred_count,
                       SUM(CASE WHEN cr.status = 'invalid' THEN 1 ELSE 0 END) AS invalid_count
                FROM review_items ri
                JOIN transcripts t ON t.id = ri.transcript_pk
                LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
                LEFT JOIN code_reviews cr
                  ON cr.review_item_id = ri.id AND cr.status <> 'abandoned'
                LEFT JOIN code_decisions cd ON cd.code_review_id = cr.id
                WHERE ri.dataset_id = ?
                GROUP BY ri.id, t.transcript_id, ri.segment_id, ri.record_id, ri.status,
                         sc.outcome, sc.created_at
                ORDER BY ri.source_order
                """,
                (dataset_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_key(study_id: int, order: int, now: str, text: str) -> str:
    import hashlib

    return hashlib.sha256(f"{study_id}\0{order}\0{now}\0{text}".encode("utf-8")).hexdigest()


def _question_signature(connection: sqlite3.Connection, study_id: int) -> tuple[tuple[Any, ...], ...]:
    rows = connection.execute(
        """
        SELECT q.id, q.display_order, q.selected, q.active, q.current_version, v.text
        FROM research_questions q
        JOIN research_question_versions v
          ON v.question_id = q.id AND v.version = q.current_version
        WHERE q.study_id = ? ORDER BY q.display_order, q.id
        """,
        (study_id,),
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def _supersede_live_snapshots_for_study(
    connection: sqlite3.Connection,
    study_id: int,
    now: str,
) -> None:
    item_rows = connection.execute(
        """
        SELECT DISTINCT gs.review_item_id
        FROM generation_snapshots gs
        JOIN review_items ri ON ri.id = gs.review_item_id
        JOIN datasets d ON d.id = ri.dataset_id
        LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
        WHERE d.study_id = ? AND sc.id IS NULL
          AND gs.status IN ('generating', 'ready', 'invalid')
        """,
        (study_id,),
    ).fetchall()
    connection.execute(
        """
        UPDATE generation_snapshots SET status = 'superseded', updated_at = ?
        WHERE id IN (
            SELECT gs.id FROM generation_snapshots gs
            JOIN review_items ri ON ri.id = gs.review_item_id
            JOIN datasets d ON d.id = ri.dataset_id
            LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
            WHERE d.study_id = ? AND sc.id IS NULL
              AND gs.status IN ('generating', 'ready', 'invalid')
        )
        """,
        (now, study_id),
    )
    connection.execute(
        """
        UPDATE code_reviews SET status = 'draft', updated_at = ?
        WHERE review_item_id IN (
            SELECT ri.id FROM review_items ri
            JOIN datasets d ON d.id = ri.dataset_id
            LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
            WHERE d.study_id = ? AND sc.id IS NULL
        ) AND status IN ('generating', 'ready', 'invalid')
        """,
        (now, study_id),
    )
    if item_rows:
        placeholders = ",".join("?" for _ in item_rows)
        connection.execute(
            f"UPDATE review_items SET status = 'unreviewed', updated_at = ? "
            f"WHERE id IN ({placeholders})",
            (now, *(int(row[0]) for row in item_rows)),
        )


def _backfill_multi_code_schema(connection: sqlite3.Connection) -> None:
    """Wrap schema-v2 single-code records without rewriting their audit payloads."""
    category_by_label = {spec.display_label: spec.id for spec in CATEGORY_SPECS}
    item_rows = connection.execute(
        """
        SELECT DISTINCT ri.id AS review_item_id, s.reviewer_id
        FROM review_items ri
        JOIN datasets ds ON ds.id = ri.dataset_id
        JOIN studies s ON s.id = ds.study_id
        LEFT JOIN generation_snapshots gs ON gs.review_item_id = ri.id
        LEFT JOIN decisions d ON d.review_item_id = ri.id
        WHERE gs.id IS NOT NULL OR d.id IS NOT NULL
        ORDER BY ri.id
        """
    ).fetchall()
    for item in item_rows:
        item_id = int(item["review_item_id"])
        reviewer_id = str(item["reviewer_id"])
        decision = connection.execute(
            "SELECT * FROM decisions WHERE review_item_id = ? AND reviewer_id = ?",
            (item_id, reviewer_id),
        ).fetchone()
        snapshots = connection.execute(
            """
            SELECT * FROM generation_snapshots
            WHERE review_item_id = ? AND reviewer_id = ? ORDER BY attempt_number, id
            """,
            (item_id, reviewer_id),
        ).fetchall()
        if not snapshots:
            if decision is not None and decision["decision"] == "skip":
                connection.execute(
                    """
                    INSERT INTO segment_completions(
                        review_item_id, reviewer_id, outcome, reason, issue_tags_json,
                        idempotency_key, created_at
                    ) VALUES (?, ?, 'skipped', ?, ?, ?, ?)
                    """,
                    (
                        item_id, reviewer_id, decision["reason"], decision["issue_tags_json"],
                        f"legacy-segment-{item_id}", decision["created_at"],
                    ),
                )
            continue
        selected = None
        if decision is not None and decision["snapshot_id"] is not None:
            selected = next(
                (row for row in snapshots if int(row["id"]) == int(decision["snapshot_id"])),
                None,
            )
        if selected is None:
            selected = next(
                (
                    row for row in reversed(snapshots)
                    if row["status"] in {"generating", "ready", "invalid"}
                ),
                snapshots[-1],
            )
        code_label = str(selected["code_label"])
        source_status = str(selected["status"])
        status = (
            "finalized" if decision is not None
            else source_status if source_status in {"generating", "ready", "invalid"}
            else "draft"
        )
        cursor = connection.execute(
            """
            INSERT INTO code_reviews(
                review_item_id, reviewer_id, ordinal, code_label, dedupe_label,
                status, created_at, updated_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                item_id, reviewer_id, code_label, code_label.strip(), status,
                selected["created_at"], selected["updated_at"],
            ),
        )
        code_review_id = int(cursor.lastrowid)
        connection.execute(
            "UPDATE generation_snapshots SET code_review_id = ? WHERE review_item_id = ?",
            (code_review_id, item_id),
        )
        categories = _snapshot_display_categories(
            connection, int(selected["id"]), category_by_label
        )
        if decision is None:
            connection.execute(
                """
                INSERT INTO code_review_drafts(
                    code_review_id, snapshot_id, category_a_id, category_b_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    code_review_id, int(selected["id"]), categories.get("A"),
                    categories.get("B"), selected["updated_at"],
                ),
            )
            continue
        cursor = connection.execute(
            """
            INSERT INTO code_decisions(
                code_review_id, snapshot_id, reviewer_id, decision,
                preferred_candidate_id, reason, issue_tags_json, idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                code_review_id, decision["snapshot_id"], reviewer_id, decision["decision"],
                decision["preferred_candidate_id"], decision["reason"],
                decision["issue_tags_json"], decision["idempotency_key"],
                decision["created_at"],
            ),
        )
        code_decision_id = int(cursor.lastrowid)
        candidate_rows = connection.execute(
            """
            SELECT ab.display_label, c.id
            FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
            WHERE ab.snapshot_id = ? ORDER BY ab.display_label
            """,
            (selected["id"],),
        ).fetchall()
        for candidate in candidate_rows:
            display_label = str(candidate["display_label"])
            category_id = categories.get(display_label)
            if category_id:
                connection.execute(
                    """
                    INSERT INTO code_decision_categories(
                        decision_id, display_label, candidate_id, category_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        code_decision_id, display_label, int(candidate["id"]), category_id,
                    ),
                )
        connection.execute(
            """
            INSERT INTO segment_completions(
                review_item_id, reviewer_id, outcome, reason, issue_tags_json,
                idempotency_key, created_at
            ) VALUES (?, ?, 'completed', NULL, '[]', ?, ?)
            """,
            (item_id, reviewer_id, f"legacy-segment-{item_id}", decision["created_at"]),
        )


def _snapshot_display_categories(
    connection: sqlite3.Connection,
    snapshot_id: int,
    category_by_label: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    rows = connection.execute(
        """
        SELECT ab.display_label, c.parsed_json, c.rendered_text
        FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
        WHERE ab.snapshot_id = ?
        """,
        (snapshot_id,),
    ).fetchall()
    for row in rows:
        category_id = None
        if row["parsed_json"]:
            try:
                payload = json.loads(str(row["parsed_json"]))
                if isinstance(payload, dict):
                    category_id = payload.get("category_id")
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        if not category_id and row["rendered_text"]:
            first_line = str(row["rendered_text"]).splitlines()[0]
            if first_line.startswith("Code category:"):
                category_id = category_by_label.get(first_line.split(":", 1)[1].strip())
        if category_id in {spec.id for spec in CATEGORY_SPECS}:
            result[str(row["display_label"])] = str(category_id)
    return result

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .transcripts import ImportBundle, TranscriptTurn


SCHEMA_VERSION = 2
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
                LEFT JOIN decisions dec ON dec.review_item_id = ri.id
                WHERE ri.dataset_id = ? AND dec.id IS NULL
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
                    SELECT d.decision, COUNT(*) AS count FROM decisions d
                    JOIN review_items ri ON ri.id = d.review_item_id
                    WHERE ri.dataset_id = ? GROUP BY d.decision
                    """,
                    (dataset_id,),
                )
            }
            reviewed = sum(decision_counts.values())
            generated = int(connection.execute(
                """
                SELECT COUNT(DISTINCT gs.review_item_id) FROM generation_snapshots gs
                JOIN review_items ri ON ri.id = gs.review_item_id
                WHERE ri.dataset_id = ? AND gs.status IN ('ready', 'invalid')
                """,
                (dataset_id,),
            ).fetchone()[0])
            invalid = int(connection.execute(
                """
                SELECT COUNT(DISTINCT gs.review_item_id) FROM generation_snapshots gs
                JOIN review_items ri ON ri.id = gs.review_item_id
                WHERE ri.dataset_id = ? AND gs.status = 'invalid'
                """,
                (dataset_id,),
            ).fetchone()[0])
        return {
            "total": total,
            "reviewed": reviewed,
            "unreviewed": total - reviewed,
            "generated": generated,
            "preferred": decision_counts.get("prefer_a", 0) + decision_counts.get("prefer_b", 0),
            "both_poor": decision_counts.get("both_poor", 0),
            "too_similar": decision_counts.get("too_similar", 0),
            "skipped": decision_counts.get("skip", 0),
            "invalid": invalid,
        }

    def list_records(self, dataset_id: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT ri.id, t.transcript_id, ri.segment_id, ri.record_id, ri.status,
                       d.decision, d.reason, d.issue_tags_json, d.created_at,
                       gs.status AS generation_status
                FROM review_items ri
                JOIN transcripts t ON t.id = ri.transcript_pk
                LEFT JOIN decisions d ON d.review_item_id = ri.id
                LEFT JOIN generation_snapshots gs
                  ON gs.review_item_id = ri.id AND gs.status IN ('ready', 'invalid')
                WHERE ri.dataset_id = ? ORDER BY ri.source_order
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
        LEFT JOIN decisions dec ON dec.review_item_id = ri.id
        WHERE d.study_id = ? AND dec.id IS NULL
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
            LEFT JOIN decisions dec ON dec.review_item_id = ri.id
            WHERE d.study_id = ? AND dec.id IS NULL
              AND gs.status IN ('generating', 'ready', 'invalid')
        )
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

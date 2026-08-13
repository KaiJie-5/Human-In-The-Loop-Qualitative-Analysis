from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import ValidationError

from .candidates import (
    candidate_json_schema,
    candidate_to_json,
    historical_candidate_fields,
    parse_candidate,
    render_candidate,
)
from .categories import CATEGORY_BY_ID, CATEGORY_CONTRACT_VERSION
from .database import ReviewItem, SQLiteStore, utc_now
from .ollama_client import OllamaClient, OllamaError, OllamaResponse, OllamaResponseError
from .prompting import PROMPT_VERSION, QuestionSnapshot, build_prompt, build_repair_prompt
from .transcripts import TranscriptTurn, select_context


DECISIONS = {"prefer_a", "prefer_b", "both_poor", "too_similar", "skip"}
ISSUE_TAGS = (
    "Evidence problem",
    "Context used as evidence",
    "Research-question relation",
    "Reflective question",
    "Specificity",
    "Formatting",
    "Other",
)


@dataclass(frozen=True, slots=True)
class CandidateView:
    id: int
    display_label: str
    candidate_number: int
    valid: bool
    rendered_text: str | None
    validation_errors: tuple[str, ...]
    parsed: dict[str, Any] | None

    @property
    def model_category_id(self) -> str | None:
        fields = historical_candidate_fields(self.parsed, self.rendered_text)
        return fields[0] if fields else None

    @property
    def reflective_question(self) -> str | None:
        fields = historical_candidate_fields(self.parsed, self.rendered_text)
        return fields[1] if fields else None


@dataclass(frozen=True, slots=True)
class SnapshotView:
    id: int
    code_review_id: int
    status: str
    code_label: str
    attempt_number: int
    category_version: str
    previous: tuple[TranscriptTurn, ...]
    target: tuple[TranscriptTurn, ...]
    following: tuple[TranscriptTurn, ...]
    questions: tuple[QuestionSnapshot, ...]
    candidates: tuple[CandidateView, ...]


@dataclass(frozen=True, slots=True)
class DraftView:
    snapshot_id: int | None
    decision: str | None
    category_a_id: str | None
    category_b_id: str | None
    reason: str
    issue_tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodeReviewView:
    id: int
    ordinal: int
    code_label: str
    status: str
    locked: bool
    snapshot: SnapshotView | None
    replacement_in_progress: bool
    draft: DraftView


class ReviewService:
    def __init__(self, store: SQLiteStore, ollama: OllamaClient):
        self.store = store
        self.ollama = ollama

    def list_code_reviews(self, item: ReviewItem) -> tuple[CodeReviewView, ...]:
        with self.store.connection() as connection:
            rows = connection.execute(
                """
                SELECT cr.*, d.snapshot_id AS draft_snapshot_id, d.decision,
                       d.category_a_id, d.category_b_id, d.reason, d.issue_tags_json,
                       EXISTS(
                           SELECT 1 FROM generation_snapshots history
                           WHERE history.code_review_id = cr.id
                       ) AS locked
                FROM code_reviews cr
                LEFT JOIN code_review_drafts d ON d.code_review_id = cr.id
                WHERE cr.review_item_id = ? AND cr.reviewer_id = ?
                  AND cr.status <> 'abandoned'
                ORDER BY cr.ordinal
                """,
                (item.id, item.reviewer_id),
            ).fetchall()
            live_by_code = {
                int(row["code_review_id"]): int(row["id"])
                for row in connection.execute(
                    """
                    SELECT id, code_review_id FROM generation_snapshots
                    WHERE review_item_id = ? AND reviewer_id = ?
                      AND code_review_id IS NOT NULL
                      AND status IN ('generating', 'ready', 'invalid')
                    """,
                    (item.id, item.reviewer_id),
                )
            }
        result: list[CodeReviewView] = []
        for row in rows:
            draft_snapshot_id = (
                int(row["draft_snapshot_id"]) if row["draft_snapshot_id"] is not None else None
            )
            live_snapshot_id = live_by_code.get(int(row["id"]))
            display_snapshot_id = draft_snapshot_id or live_snapshot_id
            snapshot = self.load_snapshot(display_snapshot_id) if display_snapshot_id else None
            result.append(
                CodeReviewView(
                    id=int(row["id"]),
                    ordinal=int(row["ordinal"]),
                    code_label=str(row["code_label"]),
                    status=str(row["status"]),
                    locked=bool(row["locked"]),
                    snapshot=snapshot,
                    replacement_in_progress=bool(
                        live_snapshot_id
                        and live_snapshot_id != draft_snapshot_id
                        and self._snapshot_status(live_snapshot_id) == "generating"
                    ),
                    draft=DraftView(
                        snapshot_id=draft_snapshot_id,
                        decision=row["decision"],
                        category_a_id=row["category_a_id"],
                        category_b_id=row["category_b_id"],
                        reason=str(row["reason"] or ""),
                        issue_tags=tuple(json.loads(row["issue_tags_json"] or "[]")),
                    ),
                )
            )
        return tuple(result)

    def add_code(self, item: ReviewItem, code_label: str) -> int:
        if not code_label.strip():
            raise ValueError("Qualitative code must not be empty.")
        now = utc_now()
        with self.store.transaction() as connection:
            self._require_open_segment(connection, item)
            duplicate = connection.execute(
                """
                SELECT 1 FROM code_reviews
                WHERE review_item_id = ? AND reviewer_id = ? AND dedupe_label = ?
                  AND status <> 'abandoned'
                """,
                (item.id, item.reviewer_id, code_label.strip()),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("This segment already contains that exact qualitative code.")
            ordinal = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(ordinal), 0) + 1 FROM code_reviews
                    WHERE review_item_id = ? AND reviewer_id = ?
                    """,
                    (item.id, item.reviewer_id),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO code_reviews(
                    review_item_id, reviewer_id, ordinal, code_label, dedupe_label,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'draft', ?, ?)
                """,
                (
                    item.id, item.reviewer_id, ordinal, code_label, code_label.strip(), now, now,
                ),
            )
            code_review_id = int(cursor.lastrowid)
            connection.execute(
                "INSERT INTO code_review_drafts(code_review_id, updated_at) VALUES (?, ?)",
                (code_review_id, now),
            )
            return code_review_id

    def update_code(self, item: ReviewItem, code_review_id: int, code_label: str) -> None:
        if not code_label.strip():
            raise ValueError("Qualitative code must not be empty.")
        with self.store.transaction() as connection:
            self._require_open_segment(connection, item)
            row = self._code_row(connection, item, code_review_id)
            if self._code_is_locked(connection, code_review_id):
                raise ValueError("A code is locked after generation starts.")
            duplicate = connection.execute(
                """
                SELECT 1 FROM code_reviews
                WHERE review_item_id = ? AND reviewer_id = ? AND dedupe_label = ?
                  AND id <> ? AND status <> 'abandoned'
                """,
                (item.id, item.reviewer_id, code_label.strip(), code_review_id),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("This segment already contains that exact qualitative code.")
            connection.execute(
                """
                UPDATE code_reviews SET code_label = ?, dedupe_label = ?, updated_at = ?
                WHERE id = ?
                """,
                (code_label, code_label.strip(), utc_now(), row["id"]),
            )

    def remove_code(self, item: ReviewItem, code_review_id: int) -> None:
        with self.store.transaction() as connection:
            self._require_open_segment(connection, item)
            self._code_row(connection, item, code_review_id)
            if self._code_is_locked(connection, code_review_id):
                raise ValueError("A code cannot be removed after generation starts.")
            connection.execute("DELETE FROM code_reviews WHERE id = ?", (code_review_id,))

    def generate_pending_codes(self, item: ReviewItem) -> tuple[CodeReviewView, ...]:
        codes = self.list_code_reviews(item)
        pending = [code for code in codes if code.snapshot is None or code.status == "draft"]
        if not pending:
            raise ValueError("Add a new code or select Regenerate for an existing code.")
        for code in pending:
            self.generate_code(item, code.id)
        return self.list_code_reviews(item)

    def generate_code(
        self,
        item: ReviewItem,
        code_review_id: int,
        *,
        replace_snapshot_id: int | None = None,
    ) -> SnapshotView:
        code = self._get_code(item, code_review_id)
        if code.status == "finalized":
            raise ValueError("Finalized code decisions cannot be regenerated.")
        study, questions, previous, target_turns, following, options = (
            self._verified_generation_context(item)
        )
        identity = {
            "model_name": study["model_name"],
            "model_digest": study["model_digest"],
            "base_url": study["ollama_base_url"],
            "options": options,
            "context_before": int(study["context_before"]),
            "context_after": int(study["context_after"]),
            "symmetric_context": bool(study["symmetric_context"]),
            "record_id": item.record_id,
            "code_review_id": code_review_id,
        }
        prompt = build_prompt(
            previous=previous,
            target_text=item.target_text,
            target_turns=target_turns,
            following=following,
            exact_code_label=code.code_label,
            questions=questions,
            generation_identity=identity,
        )
        snapshot_id = self._create_or_reuse_snapshot(
            item=item,
            code_review_id=code_review_id,
            code_label=code.code_label,
            study=study,
            questions=questions,
            previous=previous,
            target_turns=target_turns,
            following=following,
            options=options,
            prompt=prompt,
            replace_snapshot_id=replace_snapshot_id,
        )
        self._complete_snapshot(snapshot_id)
        return self.load_snapshot(snapshot_id)

    def regenerate_code(self, item: ReviewItem, code_review_id: int) -> SnapshotView:
        code = self._get_code(item, code_review_id)
        if code.snapshot is None:
            return self.generate_code(item, code_review_id)
        return self.generate_code(
            item, code_review_id, replace_snapshot_id=code.snapshot.id
        )

    def resume_pending_generation(self, item: ReviewItem, code_review_id: int) -> SnapshotView:
        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT id FROM generation_snapshots
                WHERE code_review_id = ? AND status = 'generating'
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (code_review_id,),
            ).fetchone()
        if row is None:
            raise ValueError("There is no interrupted generation for this code.")
        self._complete_snapshot(int(row["id"]))
        return self.load_snapshot(int(row["id"]))

    def load_snapshot(self, snapshot_id: int) -> SnapshotView:
        with self.store.connection() as connection:
            snapshot = connection.execute(
                "SELECT * FROM generation_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            if snapshot is None:
                raise KeyError(f"Unknown snapshot ID {snapshot_id}.")
            context_rows = connection.execute(
                "SELECT * FROM snapshot_context WHERE snapshot_id = ? ORDER BY side, position",
                (snapshot_id,),
            ).fetchall()
            question_rows = connection.execute(
                "SELECT * FROM snapshot_questions WHERE snapshot_id = ? ORDER BY ordinal",
                (snapshot_id,),
            ).fetchall()
            candidate_rows = connection.execute(
                """
                SELECT ab.display_label, c.* FROM ab_assignments ab
                JOIN candidates c ON c.id = ab.candidate_id
                WHERE ab.snapshot_id = ? ORDER BY ab.display_label
                """,
                (snapshot_id,),
            ).fetchall()
        contexts: dict[str, list[TranscriptTurn]] = {"previous": [], "target": [], "next": []}
        for row in context_rows:
            contexts[str(row["side"])].append(
                TranscriptTurn(
                    turn_index=int(row["turn_index"]),
                    speaker=str(row["speaker"]),
                    speaker_label=row["speaker_label"],
                    text=str(row["text"]),
                    paragraph_index=0,
                )
            )
        candidates = tuple(
            CandidateView(
                id=int(row["id"]),
                display_label=str(row["display_label"]),
                candidate_number=int(row["candidate_number"]),
                valid=bool(row["valid"]),
                rendered_text=row["rendered_text"],
                validation_errors=tuple(json.loads(row["validation_errors_json"])),
                parsed=_json_object(row["parsed_json"]),
            )
            for row in candidate_rows
        )
        questions = tuple(
            QuestionSnapshot(
                id=int(row["question_id"]),
                version=int(row["question_version"]),
                text=str(row["text"]),
            )
            for row in question_rows
        )
        if snapshot["code_review_id"] is None:
            raise ValueError("Generation snapshot was not migrated to a code review.")
        return SnapshotView(
            id=int(snapshot["id"]),
            code_review_id=int(snapshot["code_review_id"]),
            status=str(snapshot["status"]),
            code_label=str(snapshot["code_label"]),
            attempt_number=int(snapshot["attempt_number"]),
            category_version=str(snapshot["category_version"]),
            previous=tuple(contexts["previous"]),
            target=tuple(contexts["target"]),
            following=tuple(contexts["next"]),
            questions=questions,
            candidates=candidates,
        )

    def save_code_draft(
        self,
        *,
        item: ReviewItem,
        code_review_id: int,
        snapshot_id: int,
        decision: str | None,
        category_a_id: str | None,
        category_b_id: str | None,
        reason: str = "",
        issue_tags: Sequence[str] = (),
    ) -> None:
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f"Unsupported decision {decision!r}.")
        unknown_tags = set(issue_tags) - set(ISSUE_TAGS)
        if unknown_tags:
            raise ValueError(f"Unsupported issue tags: {sorted(unknown_tags)}.")
        for category_id in (category_a_id, category_b_id):
            if category_id is not None and category_id not in CATEGORY_BY_ID:
                raise ValueError(f"Unsupported category ID {category_id!r}.")
        with self.store.transaction() as connection:
            self._require_open_segment(connection, item)
            self._code_row(connection, item, code_review_id)
            snapshot = connection.execute(
                """
                SELECT * FROM generation_snapshots
                WHERE id = ? AND code_review_id = ? AND review_item_id = ?
                """,
                (snapshot_id, code_review_id, item.id),
            ).fetchone()
            if snapshot is None or snapshot["status"] not in {"ready", "invalid", "superseded"}:
                raise ValueError("Draft choices require a completed response pair.")
            candidate_rows = connection.execute(
                """
                SELECT ab.display_label, c.id, c.valid, c.rendered_text
                FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
                WHERE ab.snapshot_id = ?
                """,
                (snapshot_id,),
            ).fetchall()
            all_valid = len(candidate_rows) == 2 and all(
                row["valid"] and row["rendered_text"] for row in candidate_rows
            )
            if decision in {"prefer_a", "prefer_b", "too_similar"} and not all_valid:
                raise ValueError(f"Decision {decision!r} requires two valid candidates.")
            if all_valid and (category_a_id is None or category_b_id is None):
                raise ValueError("Both effective candidate categories are required.")
            connection.execute(
                """
                INSERT INTO code_review_drafts(
                    code_review_id, snapshot_id, decision, category_a_id, category_b_id,
                    reason, issue_tags_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code_review_id) DO UPDATE SET
                    snapshot_id = excluded.snapshot_id,
                    decision = excluded.decision,
                    category_a_id = excluded.category_a_id,
                    category_b_id = excluded.category_b_id,
                    reason = excluded.reason,
                    issue_tags_json = excluded.issue_tags_json,
                    updated_at = excluded.updated_at
                """,
                (
                    code_review_id, snapshot_id, decision, category_a_id, category_b_id,
                    reason, _json(sorted(set(issue_tags))), utc_now(),
                ),
            )

    def finalize_segment(self, item: ReviewItem, *, idempotency_key: str) -> int:
        with self.store.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM segment_completions
                WHERE review_item_id = ? AND reviewer_id = ?
                """,
                (item.id, item.reviewer_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["idempotency_key"] == idempotency_key
                    and existing["outcome"] == "completed"
                ):
                    return int(existing["id"])
                raise ValueError("This segment already has an immutable completion record.")
            codes = connection.execute(
                """
                SELECT cr.*, d.snapshot_id, d.decision, d.category_a_id, d.category_b_id,
                       d.reason, d.issue_tags_json
                FROM code_reviews cr
                LEFT JOIN code_review_drafts d ON d.code_review_id = cr.id
                WHERE cr.review_item_id = ? AND cr.reviewer_id = ?
                  AND cr.status <> 'abandoned'
                ORDER BY cr.ordinal
                """,
                (item.id, item.reviewer_id),
            ).fetchall()
            if not codes:
                raise ValueError("Add at least one code, or use Skip this segment.")
            if any(row["decision"] is None for row in codes):
                raise ValueError("Choose a decision for every code before finishing the segment.")
            if any(row["status"] not in {"ready", "invalid"} for row in codes):
                raise ValueError(
                    "Generate or resume a current response pair for every code before finishing."
                )
            now = utc_now()
            for code in codes:
                snapshot_id = int(code["snapshot_id"]) if code["snapshot_id"] else None
                if snapshot_id is None:
                    raise ValueError(f"Code {code['code_label']!r} has no completed response pair.")
                candidate_rows = connection.execute(
                    """
                    SELECT ab.display_label, c.id, c.valid, c.rendered_text
                    FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
                    WHERE ab.snapshot_id = ? ORDER BY ab.display_label
                    """,
                    (snapshot_id,),
                ).fetchall()
                decision = str(code["decision"])
                all_valid = len(candidate_rows) == 2 and all(
                    row["valid"] and row["rendered_text"] for row in candidate_rows
                )
                if decision in {"prefer_a", "prefer_b", "too_similar"} and not all_valid:
                    raise ValueError(f"Code {code['code_label']!r} needs two valid candidates.")
                preferred_candidate_id = None
                if decision.startswith("prefer_"):
                    display = decision[-1].upper()
                    preferred_candidate_id = next(
                        int(row["id"]) for row in candidate_rows
                        if row["display_label"] == display
                    )
                code_key = hashlib.sha256(
                    f"{idempotency_key}\0{code['id']}\0{snapshot_id}\0{decision}".encode("utf-8")
                ).hexdigest()
                cursor = connection.execute(
                    """
                    INSERT INTO code_decisions(
                        code_review_id, snapshot_id, reviewer_id, decision,
                        preferred_candidate_id, reason, issue_tags_json,
                        idempotency_key, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        code["id"], snapshot_id, item.reviewer_id, decision,
                        preferred_candidate_id, code["reason"], code["issue_tags_json"],
                        code_key, now,
                    ),
                )
                decision_id = int(cursor.lastrowid)
                categories = {
                    "A": code["category_a_id"],
                    "B": code["category_b_id"],
                }
                if all_valid and (categories["A"] is None or categories["B"] is None):
                    raise ValueError(
                        f"Code {code['code_label']!r} needs a category for both responses."
                    )
                for candidate in candidate_rows:
                    label = str(candidate["display_label"])
                    category_id = categories[label]
                    if category_id is None:
                        continue
                    connection.execute(
                        """
                        INSERT INTO code_decision_categories(
                            decision_id, display_label, candidate_id, category_id
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (decision_id, label, candidate["id"], category_id),
                    )
                connection.execute(
                    "UPDATE code_reviews SET status = 'finalized', updated_at = ? WHERE id = ?",
                    (now, code["id"]),
                )
            cursor = connection.execute(
                """
                INSERT INTO segment_completions(
                    review_item_id, reviewer_id, outcome, reason, issue_tags_json,
                    idempotency_key, created_at
                ) VALUES (?, ?, 'completed', NULL, '[]', ?, ?)
                """,
                (item.id, item.reviewer_id, idempotency_key, now),
            )
            connection.execute(
                "UPDATE review_items SET status = 'decided', updated_at = ? WHERE id = ?",
                (now, item.id),
            )
            return int(cursor.lastrowid)

    def skip_segment(
        self,
        item: ReviewItem,
        *,
        reason: str = "",
        issue_tags: Sequence[str] = (),
        idempotency_key: str,
    ) -> int:
        unknown_tags = set(issue_tags) - set(ISSUE_TAGS)
        if unknown_tags:
            raise ValueError(f"Unsupported issue tags: {sorted(unknown_tags)}.")
        with self.store.transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM segment_completions
                WHERE review_item_id = ? AND reviewer_id = ?
                """,
                (item.id, item.reviewer_id),
            ).fetchone()
            if existing is not None:
                normalized_reason = reason.strip() or None
                normalized_tags = _json(sorted(set(issue_tags)))
                if (
                    existing["idempotency_key"] == idempotency_key
                    and existing["outcome"] == "skipped"
                    and existing["reason"] == normalized_reason
                    and existing["issue_tags_json"] == normalized_tags
                ):
                    return int(existing["id"])
                raise ValueError("This segment already has an immutable completion record.")
            generated = connection.execute(
                """
                SELECT 1 FROM generation_snapshots
                WHERE review_item_id = ? AND reviewer_id = ? LIMIT 1
                """,
                (item.id, item.reviewer_id),
            ).fetchone()
            if generated is not None:
                raise ValueError("Whole-segment Skip is available only before generation.")
            now = utc_now()
            cursor = connection.execute(
                """
                INSERT INTO segment_completions(
                    review_item_id, reviewer_id, outcome, reason, issue_tags_json,
                    idempotency_key, created_at
                ) VALUES (?, ?, 'skipped', ?, ?, ?, ?)
                """,
                (
                    item.id, item.reviewer_id, reason.strip() or None,
                    _json(sorted(set(issue_tags))), idempotency_key, now,
                ),
            )
            connection.execute(
                """
                UPDATE code_reviews SET status = 'abandoned', updated_at = ?
                WHERE review_item_id = ? AND reviewer_id = ?
                """,
                (now, item.id, item.reviewer_id),
            )
            connection.execute(
                "UPDATE review_items SET status = 'decided', updated_at = ? WHERE id = ?",
                (now, item.id),
            )
            connection.execute(
                "UPDATE studies SET reviewer_locked = 1, updated_at = ? WHERE id = ?",
                (now, item.study_id),
            )
            return int(cursor.lastrowid)

    def _verified_generation_context(self, item: ReviewItem) -> tuple[Any, ...]:
        study = self.store.get_study(item.study_id)
        if not study.get("model_name") or not study.get("model_digest"):
            raise ValueError("Select and verify an Ollama model before generation.")
        verified_model = self.ollama.show_model(str(study["model_name"]))
        if verified_model.digest != study["model_digest"]:
            raise ValueError(
                "The selected Ollama model version changed. Verify and save it again on Setup."
            )
        question_rows = self.store.get_questions(item.study_id, selected_only=True)
        questions = tuple(
            QuestionSnapshot(id=int(row["id"]), version=int(row["version"]), text=str(row["text"]))
            for row in question_rows
        )
        if not questions:
            raise ValueError("Select at least one research question before generation.")
        previous, following = select_context(
            item.turns,
            item.target_turn_indexes,
            int(study["context_before"]),
            int(study["context_after"]),
        )
        target_set = set(item.target_turn_indexes)
        target_turns = tuple(turn for turn in item.turns if turn.turn_index in target_set)
        options = {
            "temperature": float(study["temperature"]),
            "top_p": float(study["top_p"]),
            "num_predict": int(study["output_tokens"]),
            "num_ctx": int(study["context_tokens"]),
        }
        return study, questions, previous, target_turns, following, options

    def _create_or_reuse_snapshot(
        self,
        *,
        item: ReviewItem,
        code_review_id: int,
        code_label: str,
        study: dict[str, Any],
        questions: Sequence[QuestionSnapshot],
        previous: Sequence[TranscriptTurn],
        target_turns: Sequence[TranscriptTurn],
        following: Sequence[TranscriptTurn],
        options: dict[str, Any],
        prompt: Any,
        replace_snapshot_id: int | None,
    ) -> int:
        with self.store.transaction() as connection:
            self._require_open_segment(connection, item)
            self._code_row(connection, item, code_review_id)
            existing = connection.execute(
                """
                SELECT * FROM generation_snapshots
                WHERE code_review_id = ? AND status IN ('generating', 'ready', 'invalid')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (code_review_id,),
            ).fetchone()
            source = None
            if replace_snapshot_id is not None:
                source = connection.execute(
                    """
                    SELECT * FROM generation_snapshots
                    WHERE id = ? AND code_review_id = ? AND review_item_id = ?
                    """,
                    (replace_snapshot_id, code_review_id, item.id),
                ).fetchone()
                if source is None:
                    if existing is not None:
                        return int(existing["id"])
                    raise ValueError("The pair selected for regeneration no longer exists.")
                if source["superseded_by"] is not None:
                    replacement = connection.execute(
                        "SELECT id FROM generation_snapshots WHERE id = ?",
                        (source["superseded_by"],),
                    ).fetchone()
                    if replacement is not None:
                        return int(replacement["id"])
            elif existing is not None and existing["input_fingerprint"] == prompt.input_fingerprint:
                return int(existing["id"])
            elif existing is not None and existing["status"] == "generating":
                return int(existing["id"])
            attempt = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(attempt_number), 0) + 1
                    FROM generation_snapshots
                    WHERE review_item_id = ? AND reviewer_id = ?
                    """,
                    (item.id, item.reviewer_id),
                ).fetchone()[0]
            )
            now = utc_now()
            if existing is not None:
                connection.execute(
                    "UPDATE generation_snapshots SET status = 'superseded', updated_at = ? WHERE id = ?",
                    (now, existing["id"]),
                )
                source = existing
            seed_1 = secrets.randbelow(2**31 - 1) + 1
            seed_2 = secrets.randbelow(2**31 - 1) + 1
            while seed_2 == seed_1:
                seed_2 = secrets.randbelow(2**31 - 1) + 1
            cursor = connection.execute(
                """
                INSERT INTO generation_snapshots(
                    review_item_id, reviewer_id, attempt_number, input_fingerprint,
                    status, code_label, requested_context_before, requested_context_after,
                    symmetric_context, prompt_version, category_version, exact_prompt,
                    prompt_sha256, model_name, model_digest, ollama_base_url,
                    options_json, seed_1, seed_2, created_at, updated_at, code_review_id
                ) VALUES (?, ?, ?, ?, 'generating', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.reviewer_id, attempt, prompt.input_fingerprint,
                    code_label, int(study["context_before"]), int(study["context_after"]),
                    int(study["symmetric_context"]), PROMPT_VERSION, CATEGORY_CONTRACT_VERSION,
                    prompt.prompt, prompt.prompt_sha256, study["model_name"], study["model_digest"],
                    study["ollama_base_url"], _json(options), seed_1, seed_2, now, now,
                    code_review_id,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            if source is not None:
                connection.execute(
                    "UPDATE generation_snapshots SET superseded_by = ? WHERE id = ?",
                    (snapshot_id, source["id"]),
                )
            for side, turns in (("previous", previous), ("target", target_turns), ("next", following)):
                connection.executemany(
                    """
                    INSERT INTO snapshot_context(
                        snapshot_id, side, position, turn_index, speaker, speaker_label, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            snapshot_id, side, position, turn.turn_index, turn.speaker,
                            turn.speaker_label, turn.text,
                        )
                        for position, turn in enumerate(turns, 1)
                    ],
                )
            connection.executemany(
                """
                INSERT INTO snapshot_questions(
                    snapshot_id, ordinal, question_id, question_version, text
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (snapshot_id, ordinal, question.id, question.version, question.text)
                    for ordinal, question in enumerate(questions, 1)
                ],
            )
            connection.execute(
                "UPDATE code_reviews SET status = 'generating', updated_at = ? WHERE id = ?",
                (now, code_review_id),
            )
            connection.execute(
                "UPDATE studies SET reviewer_locked = 1, updated_at = ? WHERE id = ?",
                (now, item.study_id),
            )
            connection.execute(
                "UPDATE review_items SET status = 'generating', updated_at = ? WHERE id = ?",
                (now, item.id),
            )
            return snapshot_id

    def _complete_snapshot(self, snapshot_id: int) -> None:
        with self.store.connection() as connection:
            snapshot = connection.execute(
                "SELECT * FROM generation_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            existing_numbers = {
                int(row[0]) for row in connection.execute(
                    "SELECT candidate_number FROM candidates WHERE snapshot_id = ?", (snapshot_id,)
                )
            }
        if snapshot is None:
            raise KeyError(f"Unknown snapshot ID {snapshot_id}.")
        if snapshot["status"] in {"ready", "invalid"} and existing_numbers == {1, 2}:
            return
        options = json.loads(snapshot["options_json"])
        for candidate_number in (1, 2):
            if candidate_number not in existing_numbers:
                self._generate_candidate(
                    snapshot, candidate_number, int(snapshot[f"seed_{candidate_number}"]), options
                )
        with self.store.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM candidates WHERE snapshot_id = ? ORDER BY candidate_number",
                (snapshot_id,),
            ).fetchall()
            if len(rows) != 2:
                return
            if not connection.execute(
                "SELECT 1 FROM ab_assignments WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone():
                first_label = "A" if secrets.randbelow(2) == 0 else "B"
                second_label = "B" if first_label == "A" else "A"
                connection.executemany(
                    "INSERT INTO ab_assignments(snapshot_id, display_label, candidate_id) VALUES (?, ?, ?)",
                    [
                        (snapshot_id, first_label, rows[0]["id"]),
                        (snapshot_id, second_label, rows[1]["id"]),
                    ],
                )
            status = "ready" if all(row["valid"] for row in rows) else "invalid"
            now = utc_now()
            prior = connection.execute(
                "SELECT * FROM generation_snapshots WHERE superseded_by = ? ORDER BY id DESC LIMIT 1",
                (snapshot_id,),
            ).fetchone()
            if status == "invalid" and prior is not None:
                prior_candidates = connection.execute(
                    "SELECT valid FROM candidates WHERE snapshot_id = ?",
                    (prior["id"],),
                ).fetchall()
                prior_status = (
                    "ready"
                    if len(prior_candidates) == 2 and all(row["valid"] for row in prior_candidates)
                    else "invalid"
                )
                connection.execute(
                    """
                    UPDATE generation_snapshots SET status = 'abandoned', updated_at = ?
                    WHERE id = ?
                    """,
                    (now, snapshot_id),
                )
                connection.execute(
                    """
                    UPDATE generation_snapshots SET status = ?, superseded_by = NULL,
                        updated_at = ? WHERE id = ?
                    """,
                    (prior_status, now, prior["id"]),
                )
                connection.execute(
                    "UPDATE code_reviews SET status = ?, updated_at = ? WHERE id = ?",
                    (prior_status, now, snapshot["code_review_id"]),
                )
                self._refresh_item_status(connection, int(snapshot["review_item_id"]), now)
                return
            connection.execute(
                "UPDATE generation_snapshots SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, snapshot_id),
            )
            connection.execute(
                "UPDATE code_reviews SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, snapshot["code_review_id"]),
            )
            if status == "ready":
                categories = self._candidate_categories(connection, snapshot_id)
                connection.execute(
                    """
                    INSERT INTO code_review_drafts(
                        code_review_id, snapshot_id, decision, category_a_id, category_b_id,
                        reason, issue_tags_json, updated_at
                    ) VALUES (?, ?, NULL, ?, ?, NULL, '[]', ?)
                    ON CONFLICT(code_review_id) DO UPDATE SET
                        snapshot_id = excluded.snapshot_id,
                        decision = NULL,
                        category_a_id = excluded.category_a_id,
                        category_b_id = excluded.category_b_id,
                        reason = NULL,
                        issue_tags_json = '[]',
                        updated_at = excluded.updated_at
                    """,
                    (
                        snapshot["code_review_id"], snapshot_id, categories.get("A"),
                        categories.get("B"), now,
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM generation_snapshots
                    WHERE code_review_id = ? AND id <> ?
                    """,
                    (snapshot["code_review_id"], snapshot_id),
                )
            else:
                categories = self._candidate_categories(connection, snapshot_id)
                connection.execute(
                    """
                    INSERT INTO code_review_drafts(
                        code_review_id, snapshot_id, decision, category_a_id, category_b_id,
                        reason, issue_tags_json, updated_at
                    ) VALUES (?, ?, NULL, ?, ?, NULL, '[]', ?)
                    ON CONFLICT(code_review_id) DO UPDATE SET
                        snapshot_id = excluded.snapshot_id,
                        decision = NULL,
                        category_a_id = excluded.category_a_id,
                        category_b_id = excluded.category_b_id,
                        reason = NULL,
                        issue_tags_json = '[]',
                        updated_at = excluded.updated_at
                    """,
                    (
                        snapshot["code_review_id"], snapshot_id, categories.get("A"),
                        categories.get("B"), now,
                    ),
                )
            self._refresh_item_status(connection, int(snapshot["review_item_id"]), now)

    def _generate_candidate(
        self,
        snapshot: sqlite3.Row,
        candidate_number: int,
        seed: int,
        options: dict[str, Any],
    ) -> None:
        prompt = str(snapshot["exact_prompt"])
        response, errors, parsed, rendered = self._call_and_validate(
            snapshot=snapshot,
            candidate_number=candidate_number,
            call_number=1,
            call_kind="initial",
            prompt=prompt,
            seed=seed,
            options=options,
        )
        if errors:
            repair_prompt = build_repair_prompt(
                original_prompt=prompt,
                invalid_content=response.content if response else "[No parseable assistant content]",
                errors=errors,
            )
            response, errors, parsed, rendered = self._call_and_validate(
                snapshot=snapshot,
                candidate_number=candidate_number,
                call_number=2,
                call_kind="repair",
                prompt=repair_prompt,
                seed=seed,
                options=options,
            )
        now = utc_now()
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO candidates(
                    snapshot_id, candidate_number, seed, raw_response_json, parsed_json,
                    rendered_text, rendered_sha256, valid, validation_errors_json, response_model,
                    response_metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id, candidate_number) DO UPDATE SET
                    raw_response_json = excluded.raw_response_json,
                    parsed_json = excluded.parsed_json,
                    rendered_text = excluded.rendered_text,
                    rendered_sha256 = excluded.rendered_sha256,
                    valid = excluded.valid,
                    validation_errors_json = excluded.validation_errors_json,
                    response_model = excluded.response_model,
                    response_metadata_json = excluded.response_metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot["id"], candidate_number, seed,
                    response.raw_body if response else None,
                    candidate_to_json(parsed) if parsed else None,
                    rendered,
                    hashlib.sha256(rendered.encode("utf-8")).hexdigest() if rendered else None,
                    int(not errors), _json(errors), response.model if response else None,
                    _json(response.metadata if response else {}), now, now,
                ),
            )

    def _call_and_validate(
        self,
        *,
        snapshot: sqlite3.Row,
        candidate_number: int,
        call_number: int,
        call_kind: str,
        prompt: str,
        seed: int,
        options: dict[str, Any],
    ) -> tuple[OllamaResponse | None, list[str], Any | None, str | None]:
        request = {
            "model": snapshot["model_name"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": candidate_json_schema(),
            "options": {**options, "seed": seed},
        }
        try:
            response = self.ollama.generate_structured(
                model=str(snapshot["model_name"]),
                prompt=prompt,
                schema=candidate_json_schema(),
                options=options,
                seed=seed,
            )
        except OllamaResponseError as exc:
            self._save_call(
                snapshot_id=int(snapshot["id"]), candidate_number=candidate_number,
                call_number=call_number, call_kind=call_kind, seed=seed, prompt=prompt,
                request=request, raw_response=exc.raw_body, transport_error=str(exc),
            )
            response = OllamaResponse(
                raw_body=exc.raw_body,
                payload={},
                content=exc.raw_body,
                model=str(snapshot["model_name"]),
                metadata={},
            )
            return response, [str(exc)], None, None
        except OllamaError as exc:
            self._save_call(
                snapshot_id=int(snapshot["id"]), candidate_number=candidate_number,
                call_number=call_number, call_kind=call_kind, seed=seed, prompt=prompt,
                request=request, raw_response=getattr(exc, "raw_body", None),
                transport_error=type(exc).__name__,
            )
            raise
        self._save_call(
            snapshot_id=int(snapshot["id"]), candidate_number=candidate_number,
            call_number=call_number, call_kind=call_kind, seed=seed, prompt=prompt,
            request=request, raw_response=response.raw_body, transport_error=None,
        )
        try:
            parsed = parse_candidate(response.content)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            return response, [f"Structured output validation failed: {exc}"], None, None
        return response, [], parsed, render_candidate(parsed)

    def _save_call(
        self,
        *,
        snapshot_id: int,
        candidate_number: int,
        call_number: int,
        call_kind: str,
        seed: int,
        prompt: str,
        request: dict[str, Any],
        raw_response: str | None,
        transport_error: str | None,
    ) -> None:
        with self.store.transaction() as connection:
            connection.execute(
                """
                INSERT INTO candidate_calls(
                    snapshot_id, candidate_number, call_number, call_kind, seed,
                    request_prompt, request_json, raw_response, transport_error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id, candidate_number, call_number) DO UPDATE SET
                    request_prompt = excluded.request_prompt,
                    request_json = excluded.request_json,
                    raw_response = excluded.raw_response,
                    transport_error = excluded.transport_error,
                    created_at = excluded.created_at
                """,
                (
                    snapshot_id, candidate_number, call_number, call_kind, seed,
                    prompt, _json(request), raw_response, transport_error, utc_now(),
                ),
            )

    def _get_code(self, item: ReviewItem, code_review_id: int) -> CodeReviewView:
        for code in self.list_code_reviews(item):
            if code.id == code_review_id:
                return code
        raise KeyError(f"Unknown code review ID {code_review_id}.")

    @staticmethod
    def _code_row(
        connection: sqlite3.Connection, item: ReviewItem, code_review_id: int
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT * FROM code_reviews
            WHERE id = ? AND review_item_id = ? AND reviewer_id = ?
            """,
            (code_review_id, item.id, item.reviewer_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown code review ID {code_review_id}.")
        return row

    @staticmethod
    def _code_is_locked(connection: sqlite3.Connection, code_review_id: int) -> bool:
        return connection.execute(
            "SELECT 1 FROM generation_snapshots WHERE code_review_id = ? LIMIT 1",
            (code_review_id,),
        ).fetchone() is not None

    @staticmethod
    def _require_open_segment(connection: sqlite3.Connection, item: ReviewItem) -> None:
        if connection.execute(
            """
            SELECT 1 FROM segment_completions
            WHERE review_item_id = ? AND reviewer_id = ?
            """,
            (item.id, item.reviewer_id),
        ).fetchone() is not None:
            raise ValueError("This segment already has an immutable completion record.")

    def _snapshot_status(self, snapshot_id: int) -> str:
        with self.store.connection() as connection:
            row = connection.execute(
                "SELECT status FROM generation_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        return str(row["status"]) if row else "missing"

    @staticmethod
    def _candidate_categories(
        connection: sqlite3.Connection, snapshot_id: int
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
            fields = historical_candidate_fields(
                _json_object(row["parsed_json"]), row["rendered_text"]
            )
            if fields:
                result[str(row["display_label"])] = fields[0]
        return result

    @staticmethod
    def _refresh_item_status(
        connection: sqlite3.Connection, review_item_id: int, now: str
    ) -> None:
        statuses = {
            str(row[0]) for row in connection.execute(
                """
                SELECT status FROM code_reviews
                WHERE review_item_id = ? AND status <> 'abandoned'
                """,
                (review_item_id,),
            )
        }
        if "generating" in statuses:
            status = "generating"
        elif "invalid" in statuses:
            status = "invalid"
        elif "ready" in statuses:
            status = "generated"
        else:
            status = "unreviewed"
        connection.execute(
            "UPDATE review_items SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, review_item_id),
        )


def segment_idempotency_key(item_id: int, reviewer_id: str, action: str) -> str:
    return hashlib.sha256(
        f"segment\0{item_id}\0{reviewer_id}\0{action}".encode("utf-8")
    ).hexdigest()


def decision_idempotency_key(item_id: int, decision: str, snapshot_id: int | None) -> str:
    """Retained for compatibility with historical callers and migration fixtures."""
    return hashlib.sha256(
        f"{item_id}\0{decision}\0{snapshot_id or ''}".encode("utf-8")
    ).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_object(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None

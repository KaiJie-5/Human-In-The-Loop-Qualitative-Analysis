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
    parse_candidate,
    render_candidate,
)
from .categories import CATEGORY_CONTRACT_VERSION
from .database import ReviewItem, SQLiteStore, utc_now
from .ollama_client import OllamaClient, OllamaError, OllamaResponse, OllamaResponseError
from .prompting import (
    PROMPT_VERSION,
    QuestionSnapshot,
    build_prompt,
    build_repair_prompt,
)
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
    display_label: str
    candidate_number: int
    valid: bool
    rendered_text: str | None
    validation_errors: tuple[str, ...]
    parsed: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class SnapshotView:
    id: int
    status: str
    code_label: str
    attempt_number: int
    category_version: str
    previous: tuple[TranscriptTurn, ...]
    target: tuple[TranscriptTurn, ...]
    following: tuple[TranscriptTurn, ...]
    questions: tuple[QuestionSnapshot, ...]
    candidates: tuple[CandidateView, ...]


class ReviewService:
    def __init__(
        self,
        store: SQLiteStore,
        ollama: OllamaClient,
    ):
        self.store = store
        self.ollama = ollama

    def generate_pair(
        self,
        item: ReviewItem,
        exact_code_label: str,
        *,
        replace_snapshot_id: int | None = None,
    ) -> SnapshotView:
        if not exact_code_label.strip():
            raise ValueError("Enter a qualitative code before generation.")
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
        before = int(study["context_before"])
        after = int(study["context_after"])
        previous, following = select_context(
            item.turns, item.target_turn_indexes, before, after
        )
        target_set = set(item.target_turn_indexes)
        target_turns = tuple(turn for turn in item.turns if turn.turn_index in target_set)
        options = {
            "temperature": float(study["temperature"]),
            "top_p": float(study["top_p"]),
            "num_predict": int(study["output_tokens"]),
            "num_ctx": int(study["context_tokens"]),
        }
        identity = {
            "model_name": study["model_name"],
            "model_digest": study["model_digest"],
            "base_url": study["ollama_base_url"],
            "options": options,
            "context_before": before,
            "context_after": after,
            "symmetric_context": bool(study["symmetric_context"]),
            "record_id": item.record_id,
        }
        prompt = build_prompt(
            previous=previous,
            target_text=item.target_text,
            target_turns=target_turns,
            following=following,
            exact_code_label=exact_code_label,
            questions=questions,
            generation_identity=identity,
        )
        snapshot_id = self._create_or_reuse_snapshot(
            item=item,
            exact_code_label=exact_code_label,
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

    def active_snapshot(self, review_item_id: int) -> SnapshotView | None:
        with self.store.connection() as connection:
            row = connection.execute(
                """
                SELECT id FROM generation_snapshots
                WHERE review_item_id = ? AND status IN ('generating', 'ready', 'invalid')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (review_item_id,),
            ).fetchone()
        return self.load_snapshot(int(row["id"])) if row else None

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
                    turn_index=int(row["turn_index"]), speaker=str(row["speaker"]),
                    speaker_label=row["speaker_label"], text=str(row["text"]),
                    paragraph_index=0,
                )
            )
        candidates = tuple(
            CandidateView(
                display_label=str(row["display_label"]),
                candidate_number=int(row["candidate_number"]), valid=bool(row["valid"]),
                rendered_text=row["rendered_text"],
                validation_errors=tuple(json.loads(row["validation_errors_json"])),
                parsed=_json_object(row["parsed_json"]),
            )
            for row in candidate_rows
        )
        questions = tuple(
            QuestionSnapshot(
                id=int(row["question_id"]), version=int(row["question_version"]),
                text=str(row["text"]),
            )
            for row in question_rows
        )
        return SnapshotView(
            id=int(snapshot["id"]), status=str(snapshot["status"]),
            code_label=str(snapshot["code_label"]),
            attempt_number=int(snapshot["attempt_number"]),
            category_version=str(snapshot["category_version"]),
            previous=tuple(contexts["previous"]), target=tuple(contexts["target"]),
            following=tuple(contexts["next"]), questions=questions, candidates=candidates,
        )

    def save_decision(
        self,
        *,
        item: ReviewItem,
        decision: str,
        reason: str = "",
        issue_tags: Sequence[str] = (),
        idempotency_key: str,
    ) -> int:
        if decision not in DECISIONS:
            raise ValueError(f"Unsupported decision {decision!r}.")
        unknown_tags = set(issue_tags) - set(ISSUE_TAGS)
        if unknown_tags:
            raise ValueError(f"Unsupported issue tags: {sorted(unknown_tags)}.")
        with self.store.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM decisions WHERE review_item_id = ? AND reviewer_id = ?",
                (item.id, item.reviewer_id),
            ).fetchone()
            if existing is not None:
                normalized_reason = reason.strip() or None
                normalized_tags = _json(sorted(set(issue_tags)))
                if (
                    existing["idempotency_key"] == idempotency_key
                    and existing["decision"] == decision
                    and existing["reason"] == normalized_reason
                    and existing["issue_tags_json"] == normalized_tags
                ):
                    return int(existing["id"])
                raise ValueError(
                    "This review item already has an immutable decision; the repeated payload differs."
                )
            snapshot = connection.execute(
                """
                SELECT * FROM generation_snapshots WHERE review_item_id = ?
                  AND reviewer_id = ? AND status IN ('generating', 'ready', 'invalid')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (item.id, item.reviewer_id),
            ).fetchone()
            preferred_candidate_id: int | None = None
            if decision in {"prefer_a", "prefer_b", "too_similar", "both_poor"} and snapshot is None:
                raise ValueError(f"Decision {decision!r} requires a generated pair.")
            if decision != "skip" and snapshot is not None and snapshot["status"] == "generating":
                raise ValueError(f"Decision {decision!r} requires completed candidate generation.")
            if decision in {"prefer_a", "prefer_b", "too_similar"}:
                candidate_rows = connection.execute(
                    """
                    SELECT ab.display_label, c.id, c.valid, c.rendered_text
                    FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
                    WHERE ab.snapshot_id = ?
                    """,
                    (snapshot["id"],),
                ).fetchall()
                if len(candidate_rows) != 2 or any(
                    not row["valid"] or not row["rendered_text"] for row in candidate_rows
                ):
                    raise ValueError(f"Decision {decision!r} requires two valid rendered candidates.")
                if decision.startswith("prefer_"):
                    label = decision[-1].upper()
                    preferred_candidate_id = next(
                        int(row["id"]) for row in candidate_rows if row["display_label"] == label
                    )
            now = utc_now()
            if decision == "skip" and snapshot is not None and snapshot["status"] == "generating":
                connection.execute(
                    "UPDATE generation_snapshots SET status = 'abandoned', updated_at = ? WHERE id = ?",
                    (now, snapshot["id"]),
                )
            connection.execute(
                "UPDATE studies SET reviewer_locked = 1, updated_at = ? WHERE id = ?",
                (now, item.study_id),
            )
            cursor = connection.execute(
                """
                INSERT INTO decisions(
                    review_item_id, snapshot_id, reviewer_id, decision,
                    preferred_candidate_id, reason, issue_tags_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, int(snapshot["id"]) if snapshot else None, item.reviewer_id,
                    decision, preferred_candidate_id, reason.strip() or None,
                    _json(sorted(set(issue_tags))), idempotency_key, now,
                ),
            )
            connection.execute(
                "UPDATE review_items SET status = 'decided', updated_at = ? WHERE id = ?",
                (now, item.id),
            )
            if snapshot is not None:
                connection.execute(
                    """
                    DELETE FROM generation_snapshots
                    WHERE review_item_id = ? AND reviewer_id = ? AND id <> ?
                    """,
                    (item.id, item.reviewer_id, snapshot["id"]),
                )
            elif decision == "skip":
                connection.execute(
                    "DELETE FROM generation_snapshots WHERE review_item_id = ? AND reviewer_id = ?",
                    (item.id, item.reviewer_id),
                )
            return int(cursor.lastrowid)

    def _create_or_reuse_snapshot(
        self,
        *,
        item: ReviewItem,
        exact_code_label: str,
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
            decided = connection.execute(
                "SELECT 1 FROM decisions WHERE review_item_id = ? AND reviewer_id = ?",
                (item.id, item.reviewer_id),
            ).fetchone()
            if decided is not None:
                raise ValueError("This review item already has an immutable decision.")
            existing = connection.execute(
                """
                SELECT * FROM generation_snapshots WHERE review_item_id = ?
                  AND reviewer_id = ? AND status IN ('generating', 'ready', 'invalid')
                ORDER BY attempt_number DESC LIMIT 1
                """,
                (item.id, item.reviewer_id),
            ).fetchone()
            if replace_snapshot_id is not None:
                source = connection.execute(
                    """
                    SELECT * FROM generation_snapshots
                    WHERE id = ? AND review_item_id = ? AND reviewer_id = ?
                    """,
                    (replace_snapshot_id, item.id, item.reviewer_id),
                ).fetchone()
                if source is None:
                    if existing is not None:
                        return int(existing["id"])
                    raise ValueError("The response pair selected for regeneration no longer exists.")
                if source["superseded_by"] is not None:
                    replacement = connection.execute(
                        "SELECT id FROM generation_snapshots WHERE id = ?",
                        (source["superseded_by"],),
                    ).fetchone()
                    if replacement is not None:
                        return int(replacement["id"])
                if existing is not None and int(existing["id"]) != replace_snapshot_id:
                    return int(existing["id"])
            elif existing is not None and existing["input_fingerprint"] == prompt.input_fingerprint:
                return int(existing["id"])
            previous_attempt = connection.execute(
                """
                SELECT COALESCE(MAX(attempt_number), 0)
                FROM generation_snapshots WHERE review_item_id = ? AND reviewer_id = ?
                """,
                (item.id, item.reviewer_id),
            ).fetchone()[0]
            attempt = int(previous_attempt) + 1
            now = utc_now()
            if existing is not None:
                connection.execute(
                    "UPDATE generation_snapshots SET status = 'superseded', updated_at = ? WHERE id = ?",
                    (now, existing["id"]),
                )
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
                    options_json, seed_1, seed_2, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'generating', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.id, item.reviewer_id, attempt, prompt.input_fingerprint,
                    exact_code_label, int(study["context_before"]), int(study["context_after"]),
                    int(study["symmetric_context"]), PROMPT_VERSION, CATEGORY_CONTRACT_VERSION,
                    prompt.prompt, prompt.prompt_sha256, study["model_name"], study["model_digest"],
                    study["ollama_base_url"], _json(options), seed_1, seed_2, now, now,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            if existing is not None:
                connection.execute(
                    "UPDATE generation_snapshots SET superseded_by = ? WHERE id = ?",
                    (snapshot_id, existing["id"]),
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
            if candidate_number in existing_numbers:
                continue
            seed = int(snapshot[f"seed_{candidate_number}"])
            self._generate_candidate(snapshot, candidate_number, seed, options)
        with self.store.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM candidates WHERE snapshot_id = ? ORDER BY candidate_number",
                (snapshot_id,),
            ).fetchall()
            if len(rows) != 2:
                return
            assignments = connection.execute(
                "SELECT COUNT(*) FROM ab_assignments WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()[0]
            if not assignments:
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
            connection.execute(
                "UPDATE generation_snapshots SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, snapshot_id),
            )
            connection.execute(
                "UPDATE review_items SET status = ?, updated_at = ? WHERE id = ?",
                ("generated" if status == "ready" else "invalid", now, snapshot["review_item_id"]),
            )
            if status == "ready":
                connection.execute(
                    """
                    DELETE FROM generation_snapshots
                    WHERE review_item_id = ? AND reviewer_id = ? AND id <> ?
                    """,
                    (snapshot["review_item_id"], snapshot["reviewer_id"], snapshot_id),
                )

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
                    int(not errors), _json(errors),
                    response.model if response else None,
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
            "model": snapshot["model_name"], "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "format": candidate_json_schema(),
            "options": {**options, "seed": seed},
        }
        try:
            response = self.ollama.generate_structured(
                model=str(snapshot["model_name"]), prompt=prompt, schema=candidate_json_schema(),
                options=options, seed=seed,
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

def decision_idempotency_key(item_id: int, decision: str, snapshot_id: int | None) -> str:
    return hashlib.sha256(f"{item_id}\0{decision}\0{snapshot_id or ''}".encode("utf-8")).hexdigest()


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

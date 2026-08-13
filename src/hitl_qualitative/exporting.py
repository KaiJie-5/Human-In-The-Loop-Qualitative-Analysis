from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .candidates import historical_candidate_fields, render_response
from .categories import CATEGORY_CONTRACT_VERSION
from .database import SQLiteStore, utc_now


@dataclass(frozen=True, slots=True)
class ExportPreview:
    eligible_count: int
    exclusion_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ExportResult:
    jsonl_path: Path
    manifest_path: Path
    row_count: int
    sha256: str
    validation_result: str
    exclusion_counts: dict[str, int]


class ExportMapper:
    @staticmethod
    def row(prompt: str, chosen: str, rejected: str) -> dict[str, Any]:
        row = {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        }
        validate_conversation_row(row)
        return row


def validate_conversation_row(row: dict[str, Any]) -> None:
    if set(row) != {"prompt", "chosen", "rejected"}:
        raise ValueError("Row must contain exactly prompt, chosen, rejected.")
    expected_roles = {"prompt": "user", "chosen": "assistant", "rejected": "assistant"}
    for field, role in expected_roles.items():
        messages = row.get(field)
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError(f"{field} must contain exactly one message.")
        message = messages[0]
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message.get("role") != role
            or not isinstance(message.get("content"), str)
            or not message["content"]
        ):
            raise ValueError(f"Invalid {field} message.")


class PreferenceExporter:
    def __init__(
        self,
        store: SQLiteStore,
        export_directory: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.export_directory = export_directory
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def preview(self, dataset_id: int) -> ExportPreview:
        rows, exclusions = self._eligible_rows(dataset_id)
        return ExportPreview(len(rows), exclusions)

    def export(self, dataset_id: int) -> ExportResult:
        dataset = self.store.get_dataset(dataset_id)
        if dataset["split"] != "adaptation":
            raise ValueError("Only adaptation datasets may be exported for DPO training.")
        rows, exclusions = self._eligible_rows(dataset_id)
        if not rows:
            raise ValueError("No eligible preference pairs are available for export.")
        self.export_directory.mkdir(parents=True, exist_ok=True)
        timestamp = self.clock().astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        stem = f"preference_pairs_category_evidence_{timestamp}"
        final_jsonl = self.export_directory / f"{stem}.jsonl"
        final_manifest = self.export_directory / f"{stem}_manifest.json"
        if final_jsonl.exists() or final_manifest.exists():
            raise FileExistsError("Timestamped export target already exists; nothing was overwritten.")
        temporary_jsonl = self.export_directory / f".{stem}.{uuid.uuid4().hex}.tmp"
        temporary_manifest = self.export_directory / f".{stem}_manifest.{uuid.uuid4().hex}.tmp"
        digest = hashlib.sha256()
        try:
            with temporary_jsonl.open("xb") as handle:
                for row in rows:
                    validate_conversation_row(row)
                    encoded = (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    handle.write(encoded)
                    digest.update(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._validate_file(temporary_jsonl)
            jsonl_sha = digest.hexdigest()
            manifest = {
                "schema_version": "hitl_preference_export_manifest_v1",
                "created_at_utc": utc_now(),
                "dataset": {
                    "name": dataset["name"],
                    "split": dataset["split"],
                    "source_sha256": dataset["source_sha256"],
                },
                "output": {
                    "filename": final_jsonl.name,
                    "sha256": jsonl_sha,
                    "row_count": len(rows),
                    "loader_validation": "passed_exact_conversation_row_v1",
                },
                "counts": {
                    "eligible": len(rows),
                    "exported": len(rows),
                    "excluded_by_reason": exclusions,
                },
            }
            with temporary_manifest.open("xb") as handle:
                encoded_manifest = (
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
                ).encode("utf-8")
                handle.write(encoded_manifest)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(temporary_jsonl, final_jsonl)
            try:
                os.rename(temporary_manifest, final_manifest)
            except Exception:
                final_jsonl.unlink(missing_ok=True)
                raise
            with self.store.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO exports(
                        dataset_id, jsonl_path, manifest_path, jsonl_sha256,
                        row_count, exclusion_counts_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id, str(final_jsonl), str(final_manifest), jsonl_sha,
                        len(rows), json.dumps(exclusions, sort_keys=True), utc_now(),
                    ),
                )
            return ExportResult(
                jsonl_path=final_jsonl, manifest_path=final_manifest, row_count=len(rows),
                sha256=jsonl_sha, validation_result="passed_exact_conversation_row_v1",
                exclusion_counts=exclusions,
            )
        finally:
            temporary_jsonl.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    @staticmethod
    def _validate_file(path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Export line {line_number} is invalid JSON: {exc}.") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Export line {line_number} must be an object.")
                validate_conversation_row(row)

    def _eligible_rows(self, dataset_id: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
        dataset = self.store.get_dataset(dataset_id)
        exclusions: dict[str, int] = {}
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, str, str]] = set()
        with self.store.connection() as connection:
            items = connection.execute(
                """
                SELECT ri.id AS item_id, ri.source_order, sc.outcome
                FROM review_items ri
                LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
                WHERE ri.dataset_id = ? ORDER BY ri.source_order
                """,
                (dataset_id,),
            ).fetchall()
            if dataset["split"] != "adaptation":
                return [], {"non_adaptation_split": len(items)}
            for item in items:
                if item["outcome"] == "skipped":
                    _increment(exclusions, "segment_skip")
                    continue
                code_rows = connection.execute(
                    """
                    SELECT cr.id AS code_review_id, cr.status AS code_status,
                           cd.id AS decision_id, cd.decision, cd.snapshot_id,
                           cd.preferred_candidate_id, cd.reviewer_id
                    FROM code_reviews cr
                    LEFT JOIN code_decisions cd ON cd.code_review_id = cr.id
                    WHERE cr.review_item_id = ? AND cr.status <> 'abandoned'
                    ORDER BY cr.ordinal
                    """,
                    (item["item_id"],),
                ).fetchall()
                if not code_rows:
                    _increment(exclusions, "no_code_review")
                    continue
                for code in code_rows:
                    if code["decision_id"] is None:
                        reason = (
                            "invalid_candidate" if code["code_status"] == "invalid"
                            else "no_decision"
                        )
                        _increment(exclusions, reason)
                        continue
                    decision = str(code["decision"])
                    if decision not in {"prefer_a", "prefer_b"}:
                        _increment(exclusions, decision)
                        continue
                    snapshot = connection.execute(
                        "SELECT * FROM generation_snapshots WHERE id = ?", (code["snapshot_id"],)
                    ).fetchone()
                    if (
                        snapshot is None
                        or snapshot["status"] != "ready"
                        or snapshot["superseded_by"]
                    ):
                        _increment(exclusions, "invalid_or_superseded_snapshot")
                        continue
                    if (
                        hashlib.sha256(snapshot["exact_prompt"].encode("utf-8")).hexdigest()
                        != snapshot["prompt_sha256"]
                    ):
                        _increment(exclusions, "unstable_prompt")
                        continue
                    identity = (
                        int(code["code_review_id"]),
                        str(code["reviewer_id"]),
                        str(snapshot["prompt_sha256"]),
                    )
                    if identity in seen:
                        _increment(exclusions, "duplicate_identity")
                        continue
                    candidate_rows = connection.execute(
                        """
                        SELECT c.*, ab.display_label, cdc.category_id AS effective_category_id
                        FROM candidates c
                        JOIN ab_assignments ab
                          ON ab.candidate_id = c.id AND ab.snapshot_id = c.snapshot_id
                        LEFT JOIN code_decision_categories cdc
                          ON cdc.decision_id = ? AND cdc.candidate_id = c.id
                        WHERE c.snapshot_id = ? ORDER BY ab.display_label
                        """,
                        (code["decision_id"], snapshot["id"]),
                    ).fetchall()
                    if len(candidate_rows) != 2 or any(
                        not candidate["valid"] or not candidate["rendered_text"]
                        for candidate in candidate_rows
                    ):
                        _increment(exclusions, "invalid_candidate")
                        continue
                    rendered: dict[int, str] = {}
                    stable = True
                    for candidate in candidate_rows:
                        fields = _stable_candidate_fields(
                            candidate, str(snapshot["category_version"])
                        )
                        category_id = candidate["effective_category_id"]
                        if fields is None or not category_id:
                            stable = False
                            break
                        rendered[int(candidate["id"])] = render_response(
                            str(category_id), fields[1]
                        )
                    if not stable:
                        _increment(exclusions, "unstable_code_or_rendering")
                        continue
                    chosen_id = int(code["preferred_candidate_id"])
                    if chosen_id not in rendered:
                        _increment(exclusions, "invalid_display_mapping")
                        continue
                    rejected_id = next(value for value in rendered if value != chosen_id)
                    rows.append(
                        ExportMapper.row(
                            str(snapshot["exact_prompt"]),
                            rendered[chosen_id],
                            rendered[rejected_id],
                        )
                    )
                    seen.add(identity)
        return rows, dict(sorted(exclusions.items()))


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _stable_candidate_fields(
    candidate: Any, category_version: str
) -> tuple[str, str] | None:
    text = str(candidate["rendered_text"] or "")
    hash_is_stable = bool(
        candidate["rendered_sha256"]
        and hashlib.sha256(text.encode("utf-8")).hexdigest() == candidate["rendered_sha256"]
    )
    if not hash_is_stable:
        return None
    try:
        parsed = json.loads(str(candidate["parsed_json"] or "{}"))
    except Exception:
        parsed = None
    fields = historical_candidate_fields(parsed if isinstance(parsed, dict) else None, text)
    if fields is None:
        return None
    if category_version == CATEGORY_CONTRACT_VERSION and text != render_response(*fields):
        return None
    return fields

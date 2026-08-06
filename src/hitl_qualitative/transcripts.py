from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SEGMENT_REQUIRED_FIELDS = {
    "record_id",
    "text",
    "interview_id",
    "segment_id",
    "speaker",
    "turn_index",
    "previous_context",
    "next_context",
    "source_html_path",
}
TURN_REQUIRED_FIELDS = {"turn_index", "speaker", "text", "paragraph_index"}


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    turn_index: int
    speaker: str
    text: str
    paragraph_index: int
    speaker_label: str | None = None


@dataclass(frozen=True, slots=True)
class TargetSegment:
    record_id: str
    transcript_id: str
    segment_id: str
    speaker: str
    text: str
    turn_index: int
    target_turn_indexes: tuple[int, ...]
    source_order: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TranscriptImport:
    transcript_id: str
    turns: tuple[TranscriptTurn, ...]
    targets: tuple[TargetSegment, ...]


@dataclass(frozen=True, slots=True)
class ImportBundle:
    source_sha256: str
    source_locator: str
    source_files: tuple[str, ...]
    transcripts: tuple[TranscriptImport, ...]

    @property
    def target_count(self) -> int:
        return sum(len(transcript.targets) for transcript in self.transcripts)


class TranscriptAdapter:
    def from_path(self, source: Path) -> ImportBundle:
        files = self._segment_files(source)
        digest = hashlib.sha256()
        payloads: list[tuple[dict[str, Any], str, int]] = []
        for path in files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as binary:
                data = binary.read()
            digest.update(data)
            payloads.extend(self._decode_lines(data, str(path)))
        return self._build(
            payloads,
            source_sha256=digest.hexdigest(),
            source_locator=str(source.resolve()),
            source_files=tuple(str(path.resolve()) for path in files),
        )

    def from_upload(self, filename: str, data: bytes) -> ImportBundle:
        if Path(filename).suffix.lower() != ".jsonl":
            raise ValueError("Uploaded segment input must be a .jsonl file.")
        return self._build(
            self._decode_lines(data, filename),
            source_sha256=hashlib.sha256(data).hexdigest(),
            source_locator=f"upload:{filename}",
            source_files=(filename,),
        )

    @staticmethod
    def _segment_files(source: Path) -> list[Path]:
        if source.is_file():
            if source.suffix.lower() != ".jsonl":
                raise ValueError(f"Segment input file must be .jsonl: {source}")
            return [source]
        if not source.is_dir():
            raise FileNotFoundError(f"Segment path does not exist: {source}")
        files = sorted(source.glob("*_segments.jsonl")) or sorted(source.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No segment JSONL files found in {source}")
        return files

    @staticmethod
    def _decode_lines(data: bytes, label: str) -> list[tuple[dict[str, Any], str, int]]:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{label} is not valid UTF-8: {exc}.") from exc
        values: list[tuple[dict[str, Any], str, int]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {label}:{line_number}: {exc}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{label}:{line_number} must contain a JSON object.")
            values.append((payload, label, line_number))
        if not values:
            raise ValueError(f"No segment records found in {label}.")
        return values

    def _build(
        self,
        payloads: list[tuple[dict[str, Any], str, int]],
        *,
        source_sha256: str,
        source_locator: str,
        source_files: tuple[str, ...],
    ) -> ImportBundle:
        turns_by_transcript: dict[str, tuple[TranscriptTurn, ...]] = {}
        targets_by_transcript: dict[str, list[TargetSegment]] = {}
        seen_records: set[str] = set()
        for source_order, (payload, label, line_number) in enumerate(payloads, 1):
            location = f"{label}:{line_number}"
            missing = sorted(SEGMENT_REQUIRED_FIELDS - set(payload))
            if missing:
                raise ValueError(f"{location} is missing required fields: {missing}.")
            if payload["speaker"] != "participant":
                raise ValueError(f"{location} target speaker must be 'participant'.")
            record_id = _nonempty(payload["record_id"], location, "record_id")
            if record_id in seen_records:
                raise ValueError(f"Duplicate record_id {record_id!r} at {location}.")
            seen_records.add(record_id)
            transcript_id = _nonempty(payload["interview_id"], location, "interview_id")
            text = _nonempty(payload["text"], location, "text")
            turns = _parse_turns(payload.get("interview_turns"), location)
            existing = turns_by_transcript.setdefault(transcript_id, turns)
            if existing != turns:
                raise ValueError(
                    f"Transcript {transcript_id!r} has inconsistent interview_turns at {location}."
                )
            turn_index = _integer(payload["turn_index"], location, "turn_index")
            raw_indexes = payload.get("target_turn_indexes", [turn_index])
            target_indexes = _target_indexes(raw_indexes, location)
            _validate_target(text, target_indexes, turns, location)
            metadata = {key: value for key, value in payload.items() if key != "interview_turns"}
            targets_by_transcript.setdefault(transcript_id, []).append(
                TargetSegment(
                    record_id=record_id,
                    transcript_id=transcript_id,
                    segment_id=_nonempty(payload["segment_id"], location, "segment_id"),
                    speaker="participant",
                    text=text,
                    turn_index=turn_index,
                    target_turn_indexes=target_indexes,
                    source_order=source_order,
                    metadata=metadata,
                )
            )
        transcripts = tuple(
            TranscriptImport(
                transcript_id=transcript_id,
                turns=turns_by_transcript[transcript_id],
                targets=tuple(sorted(targets, key=lambda item: item.source_order)),
            )
            for transcript_id, targets in sorted(
                targets_by_transcript.items(), key=lambda item: min(t.source_order for t in item[1])
            )
        )
        return ImportBundle(source_sha256, source_locator, source_files, transcripts)


def select_context(
    turns: Iterable[TranscriptTurn],
    target_turn_indexes: Iterable[int],
    turns_before: int,
    turns_after: int,
) -> tuple[tuple[TranscriptTurn, ...], tuple[TranscriptTurn, ...]]:
    if isinstance(turns_before, bool) or turns_before < 0:
        raise ValueError("turns_before must be a non-negative integer.")
    if isinstance(turns_after, bool) or turns_after < 0:
        raise ValueError("turns_after must be a non-negative integer.")
    values = tuple(turns)
    positions = {turn.turn_index: index for index, turn in enumerate(values)}
    wanted = tuple(target_turn_indexes)
    if not wanted or any(index not in positions for index in wanted):
        raise ValueError("Target turn indexes are missing from the transcript.")
    first = min(positions[index] for index in wanted)
    last = max(positions[index] for index in wanted)
    return (
        values[max(0, first - turns_before) : first],
        values[last + 1 : last + 1 + turns_after],
    )


def _parse_turns(raw: Any, location: str) -> tuple[TranscriptTurn, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{location} interview_turns must be a non-empty list.")
    turns: list[TranscriptTurn] = []
    previous = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{location} interview_turns[{index}] must be an object.")
        missing = sorted(TURN_REQUIRED_FIELDS - set(item))
        if missing:
            raise ValueError(f"{location} interview_turns[{index}] missing {missing}.")
        turn_index = _integer(item["turn_index"], location, "turn_index")
        if turn_index <= previous:
            raise ValueError(f"{location} interview turn indexes must strictly increase.")
        speaker = _nonempty(item["speaker"], location, "speaker")
        if speaker not in {"interviewer", "participant"}:
            raise ValueError(f"{location} has unsupported turn speaker {speaker!r}.")
        text = _nonempty(item["text"], location, "text")
        paragraph = _integer(item["paragraph_index"], location, "paragraph_index")
        label = item.get("speaker_label")
        turns.append(
            TranscriptTurn(
                turn_index=turn_index,
                speaker=speaker,
                text=text,
                paragraph_index=paragraph,
                speaker_label=str(label).strip() if isinstance(label, str) and label.strip() else None,
            )
        )
        previous = turn_index
    return tuple(turns)


def _target_indexes(raw: Any, location: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{location} target_turn_indexes must be a non-empty list.")
    values = tuple(_integer(value, location, "target_turn_indexes") for value in raw)
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{location} target_turn_indexes must be unique and increasing.")
    return values


def _validate_target(
    target_text: str,
    indexes: tuple[int, ...],
    turns: tuple[TranscriptTurn, ...],
    location: str,
) -> None:
    by_index = {turn.turn_index: turn for turn in turns}
    selected = [by_index[index] for index in indexes if index in by_index]
    if len(selected) != len(indexes):
        raise ValueError(f"{location} target turns are missing from interview_turns.")
    if any(turn.speaker != "participant" for turn in selected):
        raise ValueError(f"{location} target turns must all be participant turns.")
    if len(selected) == 1:
        composed = selected[0].text
    else:
        composed = "\n".join(
            f"{turn.speaker_label or turn.speaker.capitalize()}: {turn.text}" for turn in selected
        )
    if composed != target_text:
        raise ValueError(f"{location} target text does not match its target turn(s).")


def _nonempty(value: Any, location: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} {field} must be a non-empty string.")
    return value


def _integer(value: Any, location: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} {field} must be an integer.")
    return value


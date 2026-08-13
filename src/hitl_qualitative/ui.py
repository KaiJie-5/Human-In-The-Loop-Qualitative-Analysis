from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

from .candidates import ResponseSection, historical_candidate_fields, response_sections
from .categories import CATEGORY_BY_ID, CATEGORY_SPECS
from .config import AppConfig, load_config
from .database import QuestionDraft, SQLiteStore
from .exporting import PreferenceExporter
from .ollama_client import HttpOllamaClient, OllamaError
from .transcripts import TranscriptAdapter, TranscriptTurn, select_context
from .workflow import ISSUE_TAGS, CodeReviewView, ReviewService, segment_idempotency_key


DECISION_LABELS = {
    "Choose a decision…": None,
    "Prefer A": "prefer_a",
    "Prefer B": "prefer_b",
    "Both poor": "both_poor",
    "Too similar": "too_similar",
    "Skip": "skip",
}
DECISION_BY_VALUE = {value: label for label, value in DECISION_LABELS.items()}


def setup_page() -> None:
    st.title("Setup")
    st.caption("Configure the local reviewer, Ollama model, research questions, and transcript data.")
    config, store = _environment()
    try:
        study = store.get_singleton_study()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if study is None:
        st.subheader("Initialize local review")
        st.caption(
            "Enter the staff or student ID used to recover your local review progress. "
            "It remains in SQLite and is never included in DPO exports."
        )
        with st.form("initialize_singleton"):
            reviewer_id = st.text_input("Staff or student ID")
            initialize = st.form_submit_button("Continue", type="primary")
        if initialize:
            try:
                store.create_singleton_study(
                    reviewer_id=reviewer_id,
                    ollama_base_url=config.ollama_base_url,
                    context_before=config.default_context_before,
                    context_after=config.default_context_after,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    output_tokens=config.output_tokens,
                    context_tokens=config.context_tokens,
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        return

    study_id = int(study["id"])
    _model_and_context_setup(config, store, study)
    _research_question_setup(store, study_id)
    _dataset_setup(store, study_id)


def _model_and_context_setup(
    config: AppConfig, store: SQLiteStore, study: dict[str, Any]
) -> None:
    study_id = int(study["id"])
    st.subheader("Reviewer, model, and context")
    base_url = st.text_input(
        "Ollama base URL", value=str(study["ollama_base_url"]), key=f"base_url_{study_id}"
    )
    if st.button("Refresh local models", key=f"models_{study_id}"):
        try:
            models = HttpOllamaClient(
                base_url,
                health_timeout_seconds=config.health_timeout_seconds,
                generation_timeout_seconds=config.generation_timeout_seconds,
            ).list_models()
            st.session_state[f"model_list_{study_id}"] = [
                {"name": model.name, "digest": model.digest} for model in models
            ]
            st.success(f"Connected to Ollama; found {len(models)} local model(s).")
        except OllamaError as exc:
            st.session_state.pop(f"model_list_{study_id}", None)
            st.error(str(exc))
    discovered = st.session_state.get(f"model_list_{study_id}", [])
    discovered_names = [entry["name"] for entry in discovered]
    if discovered_names:
        initial = study.get("model_name")
        options = discovered_names + ["Enter manually…"]
        selected = st.selectbox(
            "Local Ollama model",
            options,
            index=discovered_names.index(initial) if initial in discovered_names else 0,
            key=f"model_select_{study_id}",
        )
        model_name = (
            st.text_input("Manual model name/tag", key=f"manual_model_{study_id}")
            if selected == "Enter manually…" else selected
        )
    else:
        model_name = st.text_input(
            "Model name/tag",
            value=str(study.get("model_name") or ""),
            help="Refresh models when Ollama is available, or enter a local model tag manually.",
            key=f"manual_model_{study_id}",
        )

    with st.form(f"study_settings_{study_id}"):
        reviewer_id = st.text_input(
            "Staff or student ID",
            value=str(study["reviewer_id"]),
            disabled=bool(study["reviewer_locked"]),
            help=(
                "Stored only in local SQLite, omitted from exports, and locked after "
                "generation or a segment decision."
            ),
        )
        symmetric = st.toggle(
            "Use a symmetric context window", value=bool(study["symmetric_context"])
        )
        context_before = st.number_input(
            "Previous turns", min_value=0, max_value=config.maximum_context_turns,
            value=int(study["context_before"]), step=1,
        )
        context_after = st.number_input(
            "Next turns", min_value=0, max_value=config.maximum_context_turns,
            value=int(study["context_after"]), step=1, disabled=symmetric,
        )
        with st.expander("Advanced generation settings"):
            temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=2.0,
                value=float(study["temperature"]), step=0.1,
            )
            top_p = st.number_input(
                "Top-p", min_value=0.01, max_value=1.0,
                value=float(study["top_p"]), step=0.05,
            )
            output_tokens = st.number_input(
                "Maximum output tokens", min_value=1, max_value=32768,
                value=int(study["output_tokens"]), step=100,
            )
            context_tokens = st.number_input(
                "Ollama context length (tokens)", min_value=8192, max_value=1048576,
                value=int(study["context_tokens"]), step=8192,
                help="Larger context windows require more GPU or system memory.",
            )
        save_settings = st.form_submit_button("Verify model and save settings", type="primary")
    if save_settings:
        try:
            if not model_name.strip():
                raise ValueError("Enter or select an Ollama model.")
            client = HttpOllamaClient(
                base_url,
                health_timeout_seconds=config.health_timeout_seconds,
                generation_timeout_seconds=config.generation_timeout_seconds,
            )
            model = client.show_model(model_name)
            if model.context_length is not None and int(context_tokens) > model.context_length:
                raise ValueError(
                    f"The requested context length is {int(context_tokens):,} tokens, but "
                    f"{model_name} reports a maximum of {model.context_length:,}."
                )
            if int(output_tokens) >= int(context_tokens):
                raise ValueError("Maximum output tokens must be smaller than the context length.")
            digest = next(
                (entry["digest"] for entry in discovered if entry["name"] == model_name), ""
            ) or model.digest
            if not digest:
                raise ValueError("Ollama did not report a stable digest for the selected model.")
            store.update_study(
                study_id,
                reviewer_id=reviewer_id,
                ollama_base_url=base_url,
                model_name=model_name,
                model_digest=digest,
                context_before=int(context_before),
                context_after=int(context_before) if symmetric else int(context_after),
                symmetric_context=symmetric,
                temperature=float(temperature),
                top_p=float(top_p),
                output_tokens=int(output_tokens),
                context_tokens=int(context_tokens),
            )
            st.session_state[f"health_ok_{study_id}"] = {
                "base_url": base_url.rstrip("/"), "model": model_name
            }
            st.success(f"Model {model_name} is available and the settings were saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _research_question_setup(store: SQLiteStore, study_id: int) -> None:
    st.subheader("Research questions")
    st.caption(
        "Use one row per question. Selected questions are snapshotted into every code prompt."
    )
    current = store.get_questions(study_id)
    editor_key = f"question_editor_{study_id}"
    data = [
        {
            "id": row["id"], "order": row["display_order"],
            "selected": bool(row["selected"]), "question": row["text"],
        }
        for row in current
    ] or [{"id": None, "order": 1, "selected": True, "question": ""}]
    edited = st.data_editor(
        data,
        num_rows="dynamic",
        key=editor_key,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "order": st.column_config.NumberColumn("Order", min_value=1, step=1, required=True),
            "selected": st.column_config.CheckboxColumn("Use", default=True),
            "question": st.column_config.TextColumn("Research question", required=True, width="large"),
        },
        use_container_width=True,
    )
    if st.button("Save research questions", key=f"save_questions_{study_id}"):
        try:
            records = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
            records = [
                row for row in records
                if _optional_int(row.get("id")) is not None
                or str(row.get("question") or "").strip()
            ]
            if not records:
                raise ValueError("Enter and select at least one research question.")
            drafts = [
                QuestionDraft(
                    id=_optional_int(row.get("id")),
                    text=str(row.get("question") or ""),
                    selected=bool(row.get("selected", True)),
                )
                for row in sorted(records, key=lambda row: int(row.get("order") or 0))
            ]
            store.save_questions(study_id, drafts)
            st.session_state.pop(editor_key, None)
            st.success("Research-question versions and ordering were saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _dataset_setup(store: SQLiteStore, study_id: int) -> None:
    st.subheader("Interview dataset")
    st.caption(
        "New imports are adaptation data: preferred pairs may be exported for another DPO round."
    )
    datasets = store.list_datasets(study_id)
    if datasets:
        st.dataframe(
            [
                {
                    "Name": row["name"], "Transcripts": row["transcript_count"],
                    "Targets": row["target_count"],
                    "SHA-256": str(row["source_sha256"])[:12] + "…",
                }
                for row in datasets
            ],
            hide_index=True,
            use_container_width=True,
        )
    path_tab, upload_tab = st.tabs(["Import local path", "Upload one JSONL file"])
    with path_tab:
        with st.form(f"path_import_{study_id}"):
            name = st.text_input("Dataset name", key=f"path_dataset_name_{study_id}")
            source_path = st.text_input("Segment JSONL file or directory path")
            submit = st.form_submit_button("Validate and import path")
        if submit:
            _import_dataset(
                store, study_id, name, "path",
                lambda: TranscriptAdapter().from_path(Path(source_path)),
            )
    with upload_tab:
        with st.form(f"upload_import_{study_id}"):
            name = st.text_input("Dataset name", key=f"upload_dataset_name_{study_id}")
            uploaded = st.file_uploader("Segment JSONL", type=["jsonl"])
            submit = st.form_submit_button("Validate and import upload")
        if submit:
            if uploaded is None:
                st.error("Choose a JSONL file.")
            else:
                _import_dataset(
                    store, study_id, name, "upload",
                    lambda: TranscriptAdapter().from_upload(uploaded.name, uploaded.getvalue()),
                )

    datasets = store.list_datasets(study_id)
    if datasets:
        labels = {
            int(row["id"]): f"{row['name']} ({row['target_count']} targets)" for row in datasets
        }
        dataset_id = st.selectbox(
            "Dataset to review", list(labels), format_func=labels.get, key=f"dataset_{study_id}"
        )
        if st.button("Start or resume review", type="primary"):
            if not store.get_questions(study_id, selected_only=True):
                st.error("Save at least one selected research question first.")
            elif not store.get_study(study_id).get("model_digest"):
                st.error("Verify and save an Ollama model first.")
            else:
                store.set_active_dataset(study_id, int(dataset_id))
                st.session_state.active_dataset_id = int(dataset_id)
                st.success("Review is ready. Open Review from the navigation menu.")


def review_page() -> None:
    st.title("Review")
    config, store = _environment()
    try:
        store.get_singleton_study()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    dataset_id = _active_dataset_id(store)
    if not dataset_id:
        st.info("Import and select a dataset on Setup first.")
        return
    item = store.get_next_item(int(dataset_id))
    progress = store.progress(int(dataset_id))
    if item is None:
        st.success(f"All {progress['total']} target segments are complete.")
        return
    study = store.get_study(item.study_id)
    client = HttpOllamaClient(
        str(study["ollama_base_url"]),
        health_timeout_seconds=config.health_timeout_seconds,
        generation_timeout_seconds=config.generation_timeout_seconds,
    )
    service = ReviewService(store, client)
    health_ok = _ollama_health(item.study_id, study, client)

    previous, following = select_context(
        item.turns, item.target_turn_indexes,
        int(study["context_before"]), int(study["context_after"]),
    )
    target_set = set(item.target_turn_indexes)
    target_turns = tuple(turn for turn in item.turns if turn.turn_index in target_set)
    questions = tuple(
        str(row["text"]) for row in store.get_questions(item.study_id, selected_only=True)
    )
    _sticky_reference(item, target_turns, questions, progress)
    with st.expander(f"Previous context ({len(previous)} turns)", expanded=False):
        _show_turns(previous)
    with st.expander(f"Next context ({len(following)} turns)", expanded=False):
        _show_turns(following)

    st.subheader("Qualitative codes")
    codes = service.list_code_reviews(item)
    for code in codes:
        _editable_code_row(service, item, code)
    with st.form(f"add_code_{item.id}", clear_on_submit=True):
        new_code = st.text_input("Add another qualitative code")
        add = st.form_submit_button("Add code")
    if add:
        try:
            service.add_code(item, new_code)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    codes = service.list_code_reviews(item)
    pending = [code for code in codes if code.snapshot is None or code.status == "draft"]
    if pending:
        st.caption(
            f"{len(pending)} code(s) await generation; each code produces two sequential Ollama calls."
        )
        if st.button(
            f"Generate responses for {len(pending)} code(s)",
            type="primary",
            disabled=not health_ok,
        ):
            try:
                with st.spinner("Analysing and Generating"):
                    service.generate_pending_codes(item)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    codes = service.list_code_reviews(item)
    all_draft_decisions = bool(codes)
    for code in codes:
        if code.snapshot is None:
            all_draft_decisions = False
            continue
        chosen = _code_response_group(service, item, code, health_ok)
        all_draft_decisions = all_draft_decisions and chosen

    if st.button(
        "Finish segment and next",
        type="primary",
        disabled=not all_draft_decisions,
        use_container_width=True,
    ):
        try:
            service.finalize_segment(
                item,
                idempotency_key=segment_idempotency_key(
                    item.id, item.reviewer_id, "complete"
                ),
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if not any(code.locked for code in codes):
        with st.expander("Skip this target segment"):
            with st.form(f"skip_segment_{item.id}"):
                reason = st.text_area("Optional skip reason", height=70)
                tags = st.multiselect("Optional issue tags", ISSUE_TAGS)
                skip = st.form_submit_button("Skip this segment")
            if skip:
                try:
                    service.skip_segment(
                        item,
                        reason=reason,
                        issue_tags=tags,
                        idempotency_key=segment_idempotency_key(
                            item.id, item.reviewer_id, "skip"
                        ),
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def _editable_code_row(service: ReviewService, item: Any, code: CodeReviewView) -> None:
    if code.locked:
        st.markdown(f"**Code {code.ordinal}:** {code.code_label}")
        return
    with st.form(f"edit_code_{code.id}"):
        value = st.text_input(f"Code {code.ordinal}", value=code.code_label)
        left, right = st.columns(2)
        update = left.form_submit_button("Update code", use_container_width=True)
        remove = right.form_submit_button("Remove code", use_container_width=True)
    if update:
        try:
            service.update_code(item, code.id, value)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if remove:
        try:
            service.remove_code(item, code.id)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _code_response_group(
    service: ReviewService,
    item: Any,
    code: CodeReviewView,
    health_ok: bool,
) -> bool:
    snapshot = code.snapshot
    assert snapshot is not None
    st.markdown(f"### Code {code.ordinal}: {html.escape(code.code_label)}", unsafe_allow_html=True)
    if code.replacement_in_progress:
        if code.draft.snapshot_id is None:
            st.warning("Generation was interrupted. Any completed model call remains saved.")
        else:
            st.warning(
                "A replacement pair was interrupted. The previous pair and draft remain saved."
            )
        if st.button("Resume generation", key=f"resume_{code.id}", disabled=not health_ok):
            try:
                with st.spinner("Resuming the saved generation…"):
                    service.resume_pending_generation(item, code.id)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    elif st.button(
        "Regenerate this code pair", key=f"regen_{code.id}", disabled=not health_ok
    ):
        try:
            with st.spinner("Analysing and Generating"):
                service.regenerate_code(item, code.id)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    candidates = {candidate.display_label: candidate for candidate in snapshot.candidates}
    columns = st.columns(2, gap="large")
    effective: dict[str, str | None] = {}
    for column, label in zip(columns, ("A", "B")):
        candidate = candidates.get(label)
        with column:
            st.markdown(f"#### Response {label}")
            if candidate is None or not candidate.valid or not candidate.reflective_question:
                st.error("This response remained invalid after one repair attempt.")
                if candidate:
                    for error in candidate.validation_errors:
                        st.caption(error)
                effective[label] = None
                continue
            default_category = (
                code.draft.category_a_id if label == "A" else code.draft.category_b_id
            ) or candidate.model_category_id
            category_ids = [spec.id for spec in CATEGORY_SPECS]
            selected = st.selectbox(
                "Code category",
                category_ids,
                index=category_ids.index(default_category) if default_category in category_ids else 0,
                format_func=lambda value: CATEGORY_BY_ID[value].display_label,
                key=f"category_{code.id}_{snapshot.id}_{label}",
            )
            effective[label] = selected
            st.markdown(
                _response_card_html(
                    (ResponseSection("Reflective question", candidate.reflective_question),)
                ),
                unsafe_allow_html=True,
            )

    all_valid = len(candidates) == 2 and all(
        candidate.valid and candidate.reflective_question for candidate in candidates.values()
    )
    allowed = (
        ["Choose a decision…", "Prefer A", "Prefer B", "Both poor", "Too similar", "Skip"]
        if all_valid else ["Choose a decision…", "Both poor", "Skip"]
    )
    initial_label = DECISION_BY_VALUE.get(code.draft.decision, "Choose a decision…")
    if initial_label not in allowed:
        initial_label = "Choose a decision…"
    decision_label = st.selectbox(
        "Decision",
        allowed,
        index=allowed.index(initial_label),
        key=f"decision_{code.id}_{snapshot.id}",
    )
    reason = st.text_area(
        "Optional reason",
        value=code.draft.reason,
        height=70,
        key=f"reason_{code.id}_{snapshot.id}",
    )
    tags = st.multiselect(
        "Optional issue tags",
        ISSUE_TAGS,
        default=list(code.draft.issue_tags),
        key=f"tags_{code.id}_{snapshot.id}",
    )
    decision = DECISION_LABELS[decision_label]
    try:
        service.save_code_draft(
            item=item,
            code_review_id=code.id,
            snapshot_id=snapshot.id,
            decision=decision,
            category_a_id=effective.get("A"),
            category_b_id=effective.get("B"),
            reason=reason,
            issue_tags=tags,
        )
        st.caption("Draft saved automatically.")
    except Exception as exc:
        st.error(f"Draft was not saved: {exc}")
        return False
    return (
        decision is not None
        and code.status in {"ready", "invalid"}
        and not code.replacement_in_progress
    )


def progress_page() -> None:
    st.title("Progress and export")
    config, store = _environment()
    try:
        store.get_singleton_study()
    except RuntimeError as exc:
        st.error(str(exc))
        return
    dataset_id = _active_dataset_id(store)
    if not dataset_id:
        st.info("Choose a dataset on Setup first.")
        return
    dataset = store.get_dataset(int(dataset_id))
    progress = store.progress(int(dataset_id))
    exporter = PreferenceExporter(store, config.export_directory)
    preview = exporter.preview(int(dataset_id))

    st.subheader("Target segments")
    segment_metrics = [
        ("Total", "total"), ("Completed", "segment_completed"),
        ("Unfinished", "unreviewed"), ("Skipped", "segment_skipped"),
    ]
    columns = st.columns(4)
    for column, (label, key) in zip(columns, segment_metrics):
        column.metric(label, progress[key])

    st.subheader("Code reviews")
    code_metrics = [
        ("Total codes", "code_total"), ("Unfinished", "code_unfinished"),
        ("Generated", "generated"), ("Preferred", "preferred"),
        ("Both poor", "both_poor"), ("Too similar", "too_similar"),
        ("Skipped", "skipped"), ("Invalid", "invalid"),
    ]
    columns = st.columns(4)
    for index, (label, key) in enumerate(code_metrics):
        columns[index % 4].metric(label, progress[key])
    st.metric("Export eligible", preview.eligible_count)

    records = store.list_records(int(dataset_id))
    record_filter = st.selectbox("Filter target segments", ["All", "Unfinished", "Completed", "Skipped"])
    if record_filter == "Unfinished":
        filtered = [row for row in records if not row.get("segment_outcome")]
    elif record_filter == "Completed":
        filtered = [row for row in records if row.get("segment_outcome") == "completed"]
    elif record_filter == "Skipped":
        filtered = [row for row in records if row.get("segment_outcome") == "skipped"]
    else:
        filtered = records
    st.dataframe(
        [
            {
                "Transcript": row["transcript_id"], "Segment": row["segment_id"],
                "Record": row["record_id"], "Segment status": row["segment_outcome"] or "unfinished",
                "Codes": row["code_count"], "Code decisions": row["code_decision_count"],
                "Preferred": row["preferred_count"] or 0, "Invalid": row["invalid_count"] or 0,
            }
            for row in filtered
        ],
        hide_index=True,
        use_container_width=True,
    )
    if records:
        inspect_id = st.selectbox(
            "Inspect one saved record",
            [int(row["id"]) for row in records],
            format_func=lambda value: next(
                f"{row['transcript_id']} / {row['segment_id']}"
                for row in records if int(row["id"]) == value
            ),
        )
        with st.expander("Protected record details", expanded=False):
            _show_record_details(store, inspect_id)

    st.subheader("DPO export")
    if preview.exclusion_counts:
        st.json(preview.exclusion_counts)
    disabled = dataset["split"] != "adaptation" or preview.eligible_count == 0
    if dataset["split"] != "adaptation":
        st.warning(f"This historical {dataset['split']} dataset remains blocked from training export.")
    if st.button("Export eligible preference pairs", type="primary", disabled=disabled):
        try:
            result = exporter.export(int(dataset_id))
            st.success(f"Exported {result.row_count} validated rows to {result.jsonl_path}")
            st.code(str(result.manifest_path))
            st.caption(f"Loader validation: {result.validation_result}; SHA-256: {result.sha256}")
        except Exception as exc:
            st.error(str(exc))


def _show_record_details(store: SQLiteStore, item_id: int) -> None:
    with store.connection() as connection:
        item = connection.execute(
            """
            SELECT ri.target_text, t.transcript_id, ri.segment_id, ri.record_id,
                   sc.outcome, sc.reason AS segment_reason
            FROM review_items ri JOIN transcripts t ON t.id = ri.transcript_pk
            LEFT JOIN segment_completions sc ON sc.review_item_id = ri.id
            WHERE ri.id = ?
            """,
            (item_id,),
        ).fetchone()
        codes = connection.execute(
            """
            SELECT cr.id, cr.ordinal, cr.code_label, cr.status, cd.id AS decision_id,
                   cd.decision, cd.reason, cd.issue_tags_json,
                   COALESCE(cd.snapshot_id, draft.snapshot_id) AS snapshot_id,
                   draft.category_a_id AS draft_category_a_id,
                   draft.category_b_id AS draft_category_b_id
            FROM code_reviews cr
            LEFT JOIN code_decisions cd ON cd.code_review_id = cr.id
            LEFT JOIN code_review_drafts draft ON draft.code_review_id = cr.id
            WHERE cr.review_item_id = ? AND cr.status <> 'abandoned'
            ORDER BY cr.ordinal
            """,
            (item_id,),
        ).fetchall()
    if item:
        st.markdown(f"**{item['transcript_id']} / {item['segment_id']} / {item['record_id']}**")
        st.caption(f"Segment outcome: {item['outcome'] or 'unfinished'}")
        st.text(str(item["target_text"]))
    for code in codes:
        st.markdown(f"### Code {code['ordinal']}: {code['code_label']}")
        st.caption(f"Status: {code['status']}; decision: {code['decision'] or 'draft'}")
        if code["snapshot_id"] is None:
            continue
        with store.connection() as connection:
            candidates = connection.execute(
                """
                SELECT ab.display_label, c.parsed_json, c.rendered_text,
                       c.validation_errors_json, c.valid, cdc.category_id
                FROM ab_assignments ab JOIN candidates c ON c.id = ab.candidate_id
                LEFT JOIN code_decision_categories cdc
                  ON cdc.decision_id = ? AND cdc.candidate_id = c.id
                WHERE ab.snapshot_id = ? ORDER BY ab.display_label
                """,
                (code["decision_id"], code["snapshot_id"]),
            ).fetchall()
        columns = st.columns(2)
        for column, candidate in zip(columns, candidates):
            with column:
                st.markdown(f"**Response {candidate['display_label']}**")
                parsed = _parsed_object(candidate["parsed_json"])
                fields = historical_candidate_fields(parsed, candidate["rendered_text"])
                draft_category = (
                    code["draft_category_a_id"]
                    if candidate["display_label"] == "A"
                    else code["draft_category_b_id"]
                )
                effective = candidate["category_id"] or draft_category or (fields[0] if fields else None)
                if candidate["valid"] and fields and effective:
                    _show_response(parsed, str(candidate["rendered_text"]), effective_category_id=effective)
                else:
                    st.json(json.loads(candidate["validation_errors_json"]))


def _ollama_health(study_id: int, study: dict[str, Any], client: HttpOllamaClient) -> bool:
    key = f"health_ok_{study_id}"
    expected = {
        "base_url": str(study["ollama_base_url"]).rstrip("/"),
        "model": str(study.get("model_name") or ""),
    }
    health_ok = st.session_state.get(key) == expected
    check_column, status_column = st.columns([1, 3])
    with check_column:
        check = st.button("Check Ollama", use_container_width=True)
    if check:
        try:
            client.show_model(expected["model"])
            st.session_state[key] = expected
            health_ok = True
        except Exception as exc:
            st.session_state.pop(key, None)
            health_ok = False
            st.error(str(exc))
    with status_column:
        if health_ok:
            st.success(f"Ollama model available: {expected['model']}")
        else:
            st.warning("Check Ollama before generating. Saved review state is unaffected.")
    return health_ok


def _sticky_reference(
    item: Any,
    target_turns: tuple[TranscriptTurn, ...],
    questions: tuple[str, ...],
    progress: dict[str, int],
) -> None:
    turns = ", ".join(str(turn.turn_index) for turn in target_turns)
    # Streamlit wraps Markdown in a same-height element container. The CSS therefore locates
    # this enclosing layout block through the reference card and pins that outer block.
    with st.container(key="review_reference_panel"):
        st.markdown(
            sticky_reference_html(
                transcript_id=item.transcript_id,
                segment_id=item.segment_id,
                split=item.split,
                target_text=item.target_text,
                turn_labels=turns,
                questions=questions,
                reviewed=progress["reviewed"],
                total=progress["total"],
            ),
            unsafe_allow_html=True,
        )


def sticky_reference_html(
    *,
    transcript_id: str,
    segment_id: str,
    split: str,
    target_text: str,
    turn_labels: str,
    questions: tuple[str, ...],
    reviewed: int,
    total: int,
) -> str:
    question_html = "".join(f"<li>{html.escape(question)}</li>" for question in questions)
    target = html.escape(target_text).replace("\n", "<br>")
    metadata = html.escape(
        f"Transcript {transcript_id} · Segment {segment_id} · Split {split} · "
        f"{reviewed} of {total} segments finalized"
    )
    return f"""
        <div class="sticky-reference">
          <div class="sticky-metadata">{metadata}</div>
          <div class="sticky-grid">
            <div><div class="sticky-label">TARGET SEGMENT · TURN(S) {html.escape(turn_labels)}</div>
                 <div class="sticky-target">{target}</div></div>
            <div><div class="sticky-label">SELECTED RESEARCH QUESTIONS</div>
                 <ol class="sticky-questions">{question_html}</ol></div>
          </div>
        </div>
        """


def _show_response(
    parsed: dict[str, Any] | None,
    rendered_text: str,
    *,
    effective_category_id: str | None = None,
) -> None:
    fields = historical_candidate_fields(parsed, rendered_text)
    if not fields:
        st.text(rendered_text)
        return
    payload = {"category_id": fields[0], "reflective_question": fields[1]}
    sections = response_sections(payload, effective_category_id=effective_category_id)
    st.markdown(_response_card_html(sections), unsafe_allow_html=True)


def _response_card_html(sections: tuple[ResponseSection, ...]) -> str:
    fields: list[str] = []
    for section in sections:
        label = html.escape(section.label)
        value = html.escape(section.value).replace("\n", "<br>")
        fields.append(
            f'<div class="response-field"><div class="response-field-label">{label}</div>'
            f'<div class="response-field-value">{value}</div></div>'
        )
    return '<div class="response-card">' + "".join(fields) + "</div>"


def _show_turns(turns: tuple[TranscriptTurn, ...]) -> None:
    if not turns:
        st.caption("No context turns in this direction.")
        return
    for turn in turns:
        st.markdown(
            f"**Turn {turn.turn_index} · {turn.speaker_label or turn.speaker.capitalize()}**  \n"
            f"{turn.text}"
        )


def _environment() -> tuple[AppConfig, SQLiteStore]:
    config = load_config()
    store = SQLiteStore(config.database_path)
    store.initialize()
    return config, store


def _active_dataset_id(store: SQLiteStore) -> int | None:
    value = st.session_state.get("active_dataset_id")
    if value:
        return int(value)
    recovered = store.recover_active_dataset()
    if recovered is not None:
        st.session_state.active_dataset_id = recovered
    return recovered


def _import_dataset(
    store: SQLiteStore,
    study_id: int,
    name: str,
    source_kind: str,
    loader: Any,
) -> None:
    try:
        bundle = loader()
        dataset_id, created = store.import_adaptation_dataset(
            study_id=study_id,
            name=name,
            source_kind=source_kind,
            bundle=bundle,
        )
        st.session_state.active_dataset_id = dataset_id
        store.set_active_dataset(study_id, dataset_id)
        if created:
            st.success(
                f"Imported {len(bundle.transcripts)} transcript(s) and {bundle.target_count} target(s)."
            )
        else:
            st.info("This adaptation dataset is already imported; existing progress was resumed.")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))


def _optional_int(value: Any) -> int | None:
    if value in (None, "") or value != value:
        return None
    return int(value)


def _parsed_object(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        div[data-testid="stLayoutWrapper"]:has(.sticky-reference) {
            position: sticky !important;
            top: 3.75rem;
            z-index: 999991;
            align-self: stretch;
        }
        .sticky-reference {
            border: 1px solid #77a88d;
            border-left: 6px solid #2d6a4f;
            border-radius: 0.65rem;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem;
            background: #ffffff;
            box-shadow: 0 0.25rem 0.8rem rgba(0, 0, 0, 0.08);
            max-height: 46vh;
            overflow-y: auto;
        }
        .sticky-grid {display: grid; grid-template-columns: 1.35fr 1fr; gap: 1.2rem;}
        .sticky-metadata {font-size: 0.78rem; color: #5f6368; margin-bottom: 0.55rem;}
        .sticky-label {font-size: 0.76rem; font-weight: 750; letter-spacing: 0.04em; margin-bottom: 0.3rem;}
        .sticky-target {line-height: 1.5;}
        .sticky-questions {margin: 0.15rem 0 0 1.2rem; padding: 0; line-height: 1.45;}
        .response-card {
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 0.7rem;
            padding: 0.35rem 1rem;
            background: rgba(248, 249, 251, 0.72);
        }
        .response-field {padding: 0.8rem 0; border-bottom: 1px solid rgba(49, 51, 63, 0.10);}
        .response-field:last-child {border-bottom: 0;}
        .response-field-label {
            display: inline-block;
            margin-bottom: 0.38rem;
            padding: 0.16rem 0.48rem;
            border-radius: 0.35rem;
            background: rgba(45, 106, 79, 0.13);
            color: #24553f;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .response-field-value {line-height: 1.55; overflow-wrap: anywhere;}
        @media (max-width: 760px) {
            div[data-testid="stLayoutWrapper"]:has(.sticky-reference) {top: 3.5rem;}
            .sticky-reference {max-height: 55vh;}
            .sticky-grid {grid-template-columns: 1fr;}
        }
        @media (prefers-color-scheme: dark) {
            .sticky-reference {background: #0e1117;}
            .response-card {background: rgba(28, 31, 36, 0.72);}
            .response-field-label {color: #b7e4c7;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

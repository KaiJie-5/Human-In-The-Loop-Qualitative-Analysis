from __future__ import annotations

import html
import json
import secrets
from pathlib import Path
from typing import Any

import streamlit as st

from .candidates import without_redundant_response_fields
from .config import AppConfig, load_config
from .database import QuestionDraft, SQLiteStore
from .exporting import PreferenceExporter
from .ollama_client import HttpOllamaClient, OllamaError
from .transcripts import TranscriptAdapter, TranscriptTurn, select_context
from .workflow import ISSUE_TAGS, ReviewService, decision_idempotency_key


def setup_page() -> None:
    st.title("Study setup")
    st.caption("Configure a local study, its transcript data, research questions, and Ollama model.")
    config, store = _environment()

    studies = store.list_studies()
    with st.expander("Create a new study", expanded=not studies):
        with st.form("create_study"):
            name = st.text_input("Study name")
            reviewer = st.text_input(
                "Reviewer ID (staff or student ID)",
                help=(
                    "Enter your institutional staff/student ID or a study-specific "
                    "pseudonym. It is stored only in the local SQLite database and "
                    "omitted from exports."
                ),
            )
            create = st.form_submit_button("Create study", type="primary")
        if create:
            try:
                study_id = store.create_study(
                    name=name,
                    reviewer_id=reviewer,
                    ollama_base_url=config.ollama_base_url,
                    context_before=config.default_context_before,
                    context_after=config.default_context_after,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    output_tokens=config.output_tokens,
                    context_tokens=config.context_tokens,
                )
                st.session_state.active_study_id = study_id
                st.success("Study created.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    studies = store.list_studies()
    if not studies:
        st.info("Create a study to continue.")
        return
    study_labels = {int(value["id"]): str(value["name"]) for value in studies}
    default_study = int(st.session_state.get("active_study_id", studies[0]["id"]))
    if default_study not in study_labels:
        default_study = next(iter(study_labels))
    study_id = st.selectbox(
        "Current study",
        options=list(study_labels),
        format_func=study_labels.get,
        index=list(study_labels).index(default_study),
    )
    st.session_state.active_study_id = study_id
    study = store.get_study(study_id)

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
    discovered_names = [item["name"] for item in discovered]
    if discovered_names:
        initial_model = study.get("model_name")
        model_name = st.selectbox(
            "Local Ollama model",
            discovered_names + ["Enter manually…"],
            index=(discovered_names.index(initial_model) if initial_model in discovered_names else 0),
            key=f"model_select_{study_id}",
        )
        if model_name == "Enter manually…":
            model_name = st.text_input("Manual model name/tag", key=f"manual_model_{study_id}")
    else:
        model_name = st.text_input(
            "Model name/tag",
            value=str(study.get("model_name") or ""),
            help="Refresh models when Ollama is available, or enter a local model tag manually.",
            key=f"manual_model_{study_id}",
        )

    with st.form(f"study_settings_{study_id}"):
        reviewer_id = st.text_input(
            "Reviewer ID (staff or student ID)",
            value=str(study["reviewer_id"]),
            disabled=bool(study["reviewer_locked"]),
            help=(
                "Enter your institutional staff/student ID or a study-specific pseudonym. "
                "The value locks after the first generation or decision and is never "
                "included in DPO exports."
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
                "Ollama context length (tokens)",
                min_value=8192,
                max_value=1048576,
                value=int(study["context_tokens"]),
                step=8192,
                help=(
                    "Maximum combined working context allocated by Ollama. The application "
                    "sends this value explicitly as num_ctx for every generation. Larger "
                    "values require more GPU or system memory."
                ),
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
            actual_after = int(context_before) if symmetric else int(context_after)
            store.update_study(
                study_id,
                reviewer_id=reviewer_id,
                ollama_base_url=base_url,
                model_name=model_name,
                model_digest=digest,
                context_before=int(context_before),
                context_after=actual_after,
                symmetric_context=symmetric,
                temperature=float(temperature),
                top_p=float(top_p),
                output_tokens=int(output_tokens),
                context_tokens=int(context_tokens),
            )
            st.session_state[f"health_ok_{study_id}"] = {
                "base_url": base_url.rstrip("/"), "model": model_name
            }
            maximum_note = (
                f" (model maximum: {model.context_length:,} tokens)"
                if model.context_length is not None else ""
            )
            st.success(
                f"Model {model_name} is available; context length {int(context_tokens):,} "
                f"tokens was saved{maximum_note}."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Research questions")
    st.caption(
        "Enter the questions your qualitative analysis aims to answer. Use one row per "
        "question, select the questions to include in model assessments, and use the + "
        "control to add more rows. The Order value controls how questions appear in prompts."
    )
    current_questions = store.get_questions(study_id)
    editor_key = f"question_editor_{study_id}"
    editor_data = [
        {
            "id": row["id"], "order": row["display_order"], "selected": bool(row["selected"]),
            "question": row["text"],
        }
        for row in current_questions
    ]
    if not editor_data:
        editor_data = [{"id": None, "order": 1, "selected": True, "question": ""}]
    edited = st.data_editor(
        editor_data,
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
                row
                for row in records
                if _optional_int(row.get("id")) is not None
                or str(row.get("question") or "").strip()
            ]
            if not records:
                raise ValueError("Enter and select at least one research question.")
            ordered = sorted(records, key=lambda row: int(row.get("order") or 0))
            drafts = [
                QuestionDraft(
                    id=_optional_int(row.get("id")),
                    text=str(row.get("question") or ""),
                    selected=bool(row.get("selected", True)),
                )
                for row in ordered
            ]
            store.save_questions(study_id, drafts)
            st.success("Research-question versions and ordering were saved.")
            st.session_state.pop(editor_key, None)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Interview dataset")
    datasets = store.list_datasets(study_id)
    if datasets:
        st.dataframe(
            [
                {
                    "Name": row["name"], "Split": row["split"],
                    "Transcripts": row["transcript_count"], "Targets": row["target_count"],
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
            dataset_name = st.text_input("Dataset name", key=f"path_dataset_name_{study_id}")
            split = st.selectbox("Data split", ["adaptation", "validation", "test"])
            source_path = st.text_input("Segment JSONL file or directory path")
            import_path = st.form_submit_button("Validate and import path")
        if import_path:
            _import_dataset(
                store, study_id, dataset_name, split, "path",
                lambda: TranscriptAdapter().from_path(Path(source_path)),
            )
    with upload_tab:
        with st.form(f"upload_import_{study_id}"):
            upload_name = st.text_input("Dataset name", key=f"upload_dataset_name_{study_id}")
            upload_split = st.selectbox(
                "Data split", ["adaptation", "validation", "test"], key=f"upload_split_{study_id}"
            )
            uploaded = st.file_uploader("Segment JSONL", type=["jsonl"])
            import_upload = st.form_submit_button("Validate and import upload")
        if import_upload:
            if uploaded is None:
                st.error("Choose a JSONL file.")
            else:
                _import_dataset(
                    store, study_id, upload_name, upload_split, "upload",
                    lambda: TranscriptAdapter().from_upload(uploaded.name, uploaded.getvalue()),
                )

    datasets = store.list_datasets(study_id)
    if datasets:
        labels = {
            int(row["id"]): f"{row['name']} — {row['split']} ({row['target_count']} targets)"
            for row in datasets
        }
        selected_dataset = st.selectbox(
            "Dataset to review", list(labels), format_func=labels.get, key=f"dataset_{study_id}"
        )
        selected = next(row for row in datasets if int(row["id"]) == selected_dataset)
        if selected["split"] != "adaptation":
            st.warning(
                f"{selected['split'].title()} preferences are frozen from training export."
            )
        if st.button("Start or resume review", type="primary"):
            if not store.get_questions(study_id, selected_only=True):
                st.error("Save at least one selected research question first.")
            elif not store.get_study(study_id).get("model_digest"):
                st.error("Verify and save an Ollama model first.")
            else:
                store.set_active_dataset(study_id, selected_dataset)
                st.session_state.active_dataset_id = selected_dataset
                st.success("Review is ready. Open Review from the navigation menu.")


def review_page() -> None:
    st.title("Review")
    config, store = _environment()
    dataset_id = _active_dataset_id(store)
    if not dataset_id:
        st.info("Choose a study and dataset on Setup, then select Start or resume review.")
        return
    try:
        dataset = store.get_dataset(int(dataset_id))
    except KeyError:
        st.session_state.pop("active_dataset_id", None)
        st.error("The selected dataset no longer exists.")
        return
    item = store.get_next_item(int(dataset_id))
    progress = store.progress(int(dataset_id))
    if item is None:
        st.success(f"All {progress['total']} review items have final decisions.")
        return
    study = store.get_study(item.study_id)
    client = HttpOllamaClient(
        str(study["ollama_base_url"]),
        health_timeout_seconds=config.health_timeout_seconds,
        generation_timeout_seconds=config.generation_timeout_seconds,
    )
    service = ReviewService(store, client, maximum_pair_attempts=config.maximum_pair_attempts)
    snapshot = service.active_snapshot(item.id)

    health_key = f"health_ok_{item.study_id}"
    expected_health = {
        "base_url": str(study["ollama_base_url"]).rstrip("/"),
        "model": str(study.get("model_name") or ""),
    }
    health_ok = st.session_state.get(health_key) == expected_health
    health_column, status_column = st.columns([1, 3])
    with health_column:
        check_health = st.button("Check Ollama", use_container_width=True)
    if check_health:
        try:
            client.show_model(expected_health["model"])
            st.session_state[health_key] = expected_health
            health_ok = True
        except Exception as exc:
            st.session_state.pop(health_key, None)
            health_ok = False
            st.error(str(exc))
    with status_column:
        if health_ok:
            st.success(f"Ollama model available: {expected_health['model']}")
        else:
            st.warning("Check Ollama before generating. Existing review state is unaffected.")

    st.caption(
        f"Transcript {item.transcript_id} · Segment {item.segment_id} · "
        f"{item.split.title()} · {progress['reviewed']} of {progress['total']} reviewed"
    )
    if snapshot:
        previous, target_turns, following = snapshot.previous, snapshot.target, snapshot.following
    else:
        previous, following = select_context(
            item.turns, item.target_turn_indexes,
            int(study["context_before"]), int(study["context_after"]),
        )
        target_set = set(item.target_turn_indexes)
        target_turns = tuple(turn for turn in item.turns if turn.turn_index in target_set)
    with st.expander(f"Previous context ({len(previous)} turns)", expanded=False):
        _show_turns(previous)
    _target_card(item.target_text, target_turns)
    with st.expander(f"Next context ({len(following)} turns)", expanded=False):
        _show_turns(following)

    questions = snapshot.questions if snapshot else tuple(
        _question_view(row) for row in store.get_questions(item.study_id, selected_only=True)
    )
    st.markdown("**Selected research questions**")
    for question in questions:
        st.write(f"{question.text}")

    with st.form(f"generate_{item.id}"):
        code_label = st.text_area(
            "Researcher qualitative code",
            value=snapshot.code_label if snapshot else "",
            height=100,
            help="The exact submitted value is frozen into the generation snapshot and renderer.",
        )
        generate = st.form_submit_button(
            "Generate two responses", type="primary",
            disabled=not bool(study.get("model_digest")) or not health_ok,
        )
    if generate:
        try:
            with st.spinner("Analysing and Generating"):
                snapshot = service.generate_pair(item, code_label)
            st.success("Both responses and their A/B assignment were saved.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    if snapshot:
        if snapshot.status == "generating":
            if st.button("Resume interrupted generation"):
                try:
                    with st.spinner("Resuming the saved generation snapshot…"):
                        service.generate_pair(item, snapshot.code_label)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            with st.form(f"interrupted_skip_{item.id}_{snapshot.id}"):
                interrupted_reason = st.text_area("Optional skip reason", height=70)
                interrupted_skip = st.form_submit_button("Abandon generation and skip")
            if interrupted_skip:
                try:
                    service.save_decision(
                        item=item,
                        decision="skip",
                        reason=interrupted_reason,
                        idempotency_key=decision_idempotency_key(item.id, "skip", snapshot.id),
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        elif snapshot.candidates:
            columns = st.columns(2, gap="large")
            for column, candidate in zip(columns, snapshot.candidates):
                with column:
                    st.subheader(f"Response {candidate.display_label}")
                    if candidate.valid and candidate.rendered_text:
                        st.text(without_redundant_response_fields(candidate.rendered_text))
                    else:
                        st.error("This response remained invalid after one repair attempt.")
                        for error in candidate.validation_errors:
                            st.caption(error)
            all_valid = len(snapshot.candidates) == 2 and all(
                candidate.valid and candidate.rendered_text for candidate in snapshot.candidates
            )
            decision_options = (
                ["Prefer A", "Prefer B", "Both poor", "Too similar", "Skip"]
                if all_valid else ["Both poor", "Skip"]
            )
            with st.form(f"decision_{item.id}_{snapshot.id}"):
                decision_label = st.radio("Decision", decision_options, horizontal=True)
                reason = st.text_area("Optional reason", height=80)
                tags = st.multiselect("Optional issue tags", ISSUE_TAGS)
                save = st.form_submit_button("Save and next", type="primary")
            if save:
                decision = {
                    "Prefer A": "prefer_a", "Prefer B": "prefer_b",
                    "Both poor": "both_poor", "Too similar": "too_similar", "Skip": "skip",
                }[decision_label]
                try:
                    service.save_decision(
                        item=item, decision=decision, reason=reason, issue_tags=tags,
                        idempotency_key=decision_idempotency_key(item.id, decision, snapshot.id),
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if snapshot is None:
        with st.form(f"pregen_skip_{item.id}"):
            skip_reason = st.text_area("Optional skip reason", height=70)
            skip_tags = st.multiselect("Optional issue tags", ISSUE_TAGS, key=f"skip_tags_{item.id}")
            skip = st.form_submit_button("Skip this segment")
        if skip:
            try:
                service.save_decision(
                    item=item, decision="skip", reason=skip_reason, issue_tags=skip_tags,
                    idempotency_key=decision_idempotency_key(item.id, "skip", None),
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def progress_page() -> None:
    st.title("Progress and export")
    config, store = _environment()
    dataset_id = _active_dataset_id(store)
    if not dataset_id:
        st.info("Choose a dataset on Setup first.")
        return
    dataset = store.get_dataset(int(dataset_id))
    progress = store.progress(int(dataset_id))
    labels = [
        ("Total", "total"), ("Unreviewed", "unreviewed"), ("Generated", "generated"),
        ("Preferred", "preferred"), ("Both poor", "both_poor"),
        ("Too similar", "too_similar"), ("Skipped", "skipped"), ("Latest invalid", "invalid"),
    ]
    columns = st.columns(4)
    for index, (label, key) in enumerate(labels):
        columns[index % 4].metric(label, progress[key])
    st.caption("Generated and latest-invalid counts may overlap final decision counts.")

    records = store.list_records(int(dataset_id))
    decision_filter = st.selectbox(
        "Filter saved records",
        ["All", "Unreviewed", "prefer_a", "prefer_b", "both_poor", "too_similar", "skip"],
    )
    filtered = records
    if decision_filter == "Unreviewed":
        filtered = [row for row in records if not row.get("decision")]
    elif decision_filter != "All":
        filtered = [row for row in records if row.get("decision") == decision_filter]
    st.dataframe(
        [
            {
                "Transcript": row["transcript_id"], "Segment": row["segment_id"],
                "Record": row["record_id"], "Status": row["status"],
                "Generation": row["generation_status"], "Decision": row["decision"],
                "Reason": row["reason"],
            }
            for row in filtered
        ],
        hide_index=True,
        use_container_width=True,
    )
    if records:
        inspect_id = st.selectbox(
            "Inspect one saved record", [int(row["id"]) for row in records],
            format_func=lambda value: next(
                f"{row['transcript_id']} / {row['segment_id']}" for row in records if row["id"] == value
            ),
        )
        with st.expander("Protected record details", expanded=False):
            _show_record_details(store, inspect_id)

    st.subheader("DPO export")
    exporter = PreferenceExporter(store, config.export_directory)
    preview = exporter.preview(int(dataset_id))
    st.metric("Eligible adaptation preferences", preview.eligible_count)
    if preview.exclusion_counts:
        st.json(preview.exclusion_counts)
    disabled = dataset["split"] != "adaptation" or preview.eligible_count == 0
    if dataset["split"] != "adaptation":
        st.warning(f"Export is disabled for the immutable {dataset['split']} split.")
    if st.button("Export eligible preference pairs", type="primary", disabled=disabled):
        try:
            result = exporter.export(int(dataset_id))
            st.success(
                f"Exported {result.row_count} validated rows to {result.jsonl_path}"
            )
            st.code(str(result.manifest_path))
            st.caption(f"Loader validation: {result.validation_result}; SHA-256: {result.sha256}")
        except Exception as exc:
            st.error(str(exc))


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
    split: str,
    source_kind: str,
    loader: Any,
) -> None:
    try:
        bundle = loader()
        dataset_id, created = store.import_dataset(
            study_id=study_id, name=name, split=split,
            source_kind=source_kind, bundle=bundle,
        )
        st.session_state.active_dataset_id = dataset_id
        store.set_active_dataset(study_id, dataset_id)
        if created:
            st.success(
                f"Imported {len(bundle.transcripts)} transcript(s) and {bundle.target_count} target(s)."
            )
        else:
            st.info("This dataset checksum and split already exist; existing progress was resumed.")
        st.rerun()
    except Exception as exc:
        st.error(str(exc))


def _target_card(target_text: str, target_turns: tuple[TranscriptTurn, ...]) -> None:
    turn_labels = ", ".join(str(turn.turn_index) for turn in target_turns)
    st.markdown(
        "<div class='target-card'><div class='target-label'>TARGET SEGMENT · TURN(S) "
        + html.escape(turn_labels)
        + "</div><div>"
        + html.escape(target_text).replace("\n", "<br>")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def _show_turns(turns: tuple[TranscriptTurn, ...]) -> None:
    if not turns:
        st.caption("No context turns in this direction.")
        return
    for turn in turns:
        st.markdown(
            f"**Turn {turn.turn_index} · {turn.speaker_label or turn.speaker.capitalize()}**  \n"
            f"{turn.text}"
        )


def _question_view(row: dict[str, Any]) -> Any:
    from .prompting import QuestionSnapshot

    return QuestionSnapshot(id=int(row["id"]), version=int(row["version"]), text=str(row["text"]))


def _optional_int(value: Any) -> int | None:
    if value in (None, "") or value != value:  # NaN from a newly added data-editor row
        return None
    return int(value)


def _show_record_details(store: SQLiteStore, item_id: int) -> None:
    with store.connection() as connection:
        item = connection.execute(
            """
            SELECT ri.target_text, t.transcript_id, ri.segment_id, ri.record_id
            FROM review_items ri JOIN transcripts t ON t.id = ri.transcript_pk
            WHERE ri.id = ?
            """,
            (item_id,),
        ).fetchone()
        decision = connection.execute(
            "SELECT decision, reason, issue_tags_json, created_at FROM decisions WHERE review_item_id = ?",
            (item_id,),
        ).fetchone()
        candidates = connection.execute(
            """
            SELECT ab.display_label, c.valid, c.rendered_text, c.validation_errors_json
            FROM generation_snapshots gs
            JOIN ab_assignments ab ON ab.snapshot_id = gs.id
            JOIN candidates c ON c.id = ab.candidate_id
            WHERE gs.review_item_id = ? AND gs.status IN ('ready', 'invalid')
            ORDER BY gs.attempt_number DESC, ab.display_label
            """,
            (item_id,),
        ).fetchall()
    if item:
        st.markdown(f"**{item['transcript_id']} / {item['segment_id']} / {item['record_id']}**")
        st.text(str(item["target_text"]))
    if decision:
        st.json(dict(decision))
    for candidate in candidates[:2]:
        st.markdown(f"**Response {candidate['display_label']}**")
        if candidate["rendered_text"]:
            st.text(without_redundant_response_fields(str(candidate["rendered_text"])))
        else:
            st.json(json.loads(candidate["validation_errors_json"]))


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        .target-card {
            border: 1px solid #77a88d;
            border-left: 6px solid #2d6a4f;
            border-radius: 0.55rem;
            padding: 1rem 1.15rem;
            margin: 0.8rem 0;
            background: rgba(82, 183, 136, 0.10);
            line-height: 1.6;
        }
        .target-label {font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

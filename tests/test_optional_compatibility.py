from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from hitl_qualitative.exporting import ExportMapper, validate_conversation_row


@pytest.mark.upstream_compat
def test_export_row_with_configured_upstream_loader() -> None:
    root_value = os.environ.get("DPO_REPOSITORY_PATH")
    if not root_value:
        pytest.skip("Set DPO_REPOSITORY_PATH to the upstream checkout.")
    source = Path(root_value) / "src"
    if not source.is_dir():
        pytest.skip("Configured DPO_REPOSITORY_PATH has no src directory.")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("dpo_training.data")
        row = ExportMapper.row("saved prompt", "preferred response", "other response")
        module._validate_conversation_row(row, "category_evidence", 1)
    finally:
        sys.path.remove(str(source))


@pytest.mark.real_data_compat
def test_first_twenty_real_rows_include_all_heading_contracts() -> None:
    path_value = os.environ.get("DPO_REFERENCE_JSONL")
    if not path_value:
        pytest.skip("Set DPO_REFERENCE_JSONL to preference_pairs_category_evidence.jsonl.")
    path = Path(path_value)
    if not path.is_file():
        pytest.skip(f"Configured DPO_REFERENCE_JSONL is not accessible: {path}")
    categories: set[str] = set()
    inspected = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_conversation_row(row)
            inspected += 1
            for field in ("chosen", "rejected"):
                first = row[field][0]["content"].splitlines()[0]
                if first.startswith("Code category: "):
                    categories.add(first.removeprefix("Code category: "))
            if inspected == 20:
                break
    assert inspected == 20
    assert categories == {
        "Wrong code",
        "Descriptive but not answering the research question",
        "Too broad code",
        "Useful analytical code",
    }

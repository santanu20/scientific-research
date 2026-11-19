"""Regression tests for provenance manifest durability and validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _provenance import (  # noqa: E402
    PROVENANCE_FILENAME,
    ProvenanceManifestError,
    load_manifest,
    record_stage,
)


def test_missing_manifest_starts_new_history(tmp_path: Path) -> None:
    record = record_stage("discovery", {}, {}, {}, tmp_path)
    assert record["stage"] == "discovery"
    assert load_manifest(tmp_path) == [record]


def test_valid_manifest_is_appended_without_mutating_prior_records(tmp_path: Path) -> None:
    first = record_stage("discovery", {"query": "x"}, {}, {}, tmp_path)
    second = record_stage("verification", {}, {}, {}, tmp_path)
    assert load_manifest(tmp_path) == [first, second]


@pytest.mark.parametrize(
    "contents",
    [
        "{invalid json",
        "null",
        '[{"stage": "ok"}, null]',
        '"unexpected scalar"',
    ],
)
def test_corrupt_or_invalid_manifest_fails_closed_and_is_preserved(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / PROVENANCE_FILENAME
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ProvenanceManifestError):
        record_stage("synthesis", {}, {}, {}, tmp_path)

    assert path.read_text(encoding="utf-8") == contents


def test_load_manifest_distinguishes_corruption_from_absence(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == []
    path = tmp_path / PROVENANCE_FILENAME
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ProvenanceManifestError):
        load_manifest(tmp_path)


def test_manifest_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    record_stage("discovery", {}, {}, {}, tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []

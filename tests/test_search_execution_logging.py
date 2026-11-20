from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import discover
from _sources import PaperRecord

from research_core import SearchExecutionStatus


def test_search_multi_source_records_nonempty_adapter_results(monkeypatch) -> None:
    monkeypatch.setattr(
        discover,
        "crossref_search",
        lambda query, **kwargs: [PaperRecord(title="A returned paper")],
    )
    executions = []
    papers = discover.search_multi_source(
        "test query",
        sources=["crossref"],
        max_per_source=7,
        execution_records=executions,
    )

    assert len(papers) == 1
    assert len(executions) == 1
    execution = executions[0]
    assert execution.provider == "crossref"
    assert execution.exact_query == "test query"
    assert execution.result_count == 1
    assert execution.status is SearchExecutionStatus.SUCCEEDED
    assert execution.elapsed_ms is not None
    assert dict(execution.parameters)["max_results"] == "7"


def test_search_multi_source_records_adapter_empty_result(monkeypatch) -> None:
    monkeypatch.setattr(discover, "crossref_search", lambda query, **kwargs: [])
    executions = []

    papers = discover.search_multi_source(
        "no-hit query",
        sources=["crossref"],
        execution_records=executions,
    )

    assert papers == []
    assert len(executions) == 1
    assert executions[0].status is SearchExecutionStatus.EMPTY
    assert executions[0].result_count == 0
    assert executions[0].error_type is None


def test_search_multi_source_records_raised_provider_error_without_message(monkeypatch) -> None:
    def fail(query, **kwargs):
        raise RuntimeError("token=private-secret")

    monkeypatch.setattr(discover, "crossref_search", fail)
    executions = []

    papers = discover.search_multi_source(
        "failing query",
        sources=["crossref"],
        execution_records=executions,
    )

    assert papers == []
    assert len(executions) == 1
    assert executions[0].status is SearchExecutionStatus.FAILED
    assert executions[0].error_type == "RuntimeError"
    assert not hasattr(executions[0], "error_message")


def test_execution_collector_is_optional_and_return_type_is_unchanged(monkeypatch) -> None:
    expected = [PaperRecord(title="Existing caller result")]
    monkeypatch.setattr(discover, "crossref_search", lambda query, **kwargs: expected)

    result = discover.search_multi_source("test query", sources=["crossref"])

    assert isinstance(result, list)
    assert result == expected



def test_search_multi_source_records_timeout(monkeypatch) -> None:
    def slow(query, **kwargs):
        time.sleep(0.15)
        return []

    monkeypatch.setattr(discover, "crossref_search", slow)
    executions = []

    papers = discover.search_multi_source(
        "slow query",
        sources=["crossref"],
        per_source_timeout=0.01,
        execution_records=executions,
    )

    assert papers == []
    assert len(executions) == 1
    assert executions[0].status is SearchExecutionStatus.TIMED_OUT
    assert executions[0].error_type == "TimeoutError"

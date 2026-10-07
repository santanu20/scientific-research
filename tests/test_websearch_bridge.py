#!/usr/bin/env python3
"""Unit pins for the web-pro bridge (_websearch.run_pro_synthesis).

The bridge shells out to the sibling web-search skill CLI; these pins mock
the subprocess boundary and verify the adaptation contract the pipeline
supplement site consumes: {synthesis, confidence, facets, coverage,
sources} — plus loud-skip (None, never raise) on every failure mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# Make scripts/ importable (matches tests/test_skill.py pattern)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import _websearch as ws  # noqa: E402

FAKE_PAYLOAD = {
    "synthesis": "Answer text with [0] marker.",
    "confidence": 0.55,
    "coverage_detail": {
        "methane": ["https://a", "https://b"],
        "hydrate": ["https://a"],
        "slope": ["https://a", "https://c", "https://d"],
    },
    "sources": [
        {"url": "https://a", "title": "Paper A", "host": "a.org", "tier": 1},
        {"url": "https://b", "title": "Paper B", "host": "b.org", "tier": 2},
    ],
    "n_sources": 2,
    "citation_grounding": [{"grounded": True}],
}


def _fake_proc(stdout: str = "", rc: int = 0) -> SimpleNamespace:
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr="log line\n")


def _skill_root(tmp_path: Path) -> Path:
    root = tmp_path / "web-search"
    root.mkdir()
    (root / "web_search.py").write_text("# stub\n", encoding="utf-8")
    return root


def test_unit__bridge__missing_skill_dir_returns_none(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("WEB_SEARCH_SKILL_DIR", str(tmp_path / "nowhere"))
    with caplog.at_level("WARNING", logger="scientific_research.websearch"):
        result = ws.run_pro_synthesis("query")
    assert result is None
    assert any("not found" in r.message for r in caplog.records)


def test_unit__bridge__happy_path_maps_contract(tmp_path, monkeypatch) -> None:
    root = _skill_root(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_SKILL_DIR", str(root))
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return _fake_proc(stdout=json.dumps(FAKE_PAYLOAD))

    monkeypatch.setattr(ws, "subprocess", SimpleNamespace(run=fake_run))
    result = ws.run_pro_synthesis("methane hydrate slope failure")
    assert result is not None
    # CLI invocation shape: sibling root cwd, pro subcommand, bounded rounds
    assert seen["cwd"] == root
    assert seen["cmd"][1:][:3] == ["web_search.py", "pro", "methane hydrate slope failure"]
    assert "--max-rounds" in seen["cmd"]
    assert "--no-llm" in seen["cmd"]
    # Contract mapping
    assert result["confidence"] == 0.55
    assert sorted(result["facets"]) == ["hydrate", "methane", "slope"]
    # Coverage mirrors pro's own >= 2 distinct-source rule
    assert result["coverage"] == {"methane": True, "hydrate": False, "slope": True}
    assert result["n_sources"] == 2
    # Self-contained section: [n] resolution list appended (0-based markers)
    assert "Web sources:" in result["synthesis"]
    assert "[0] Paper A — https://a" in result["synthesis"]
    assert "[1] Paper B — https://b" in result["synthesis"]


def test_unit__bridge__failure_modes_return_none(tmp_path, monkeypatch) -> None:
    root = _skill_root(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_SKILL_DIR", str(root))

    def boom_timeout(cmd, **kwargs):
        raise _TE(cmd, 1.0)

    import subprocess as _sp

    _TE = _sp.TimeoutExpired

    def _ns(run):
        return SimpleNamespace(run=run, TimeoutExpired=_TE)

    for fake in (
        _ns(lambda c, **k: _fake_proc(rc=1)),
        _ns(lambda c, **k: _fake_proc(stdout="not json{")),
        _ns(boom_timeout),
        _ns(lambda c, **k: _fake_proc(stdout=json.dumps({"synthesis": ""}))),
    ):
        monkeypatch.setattr(ws, "subprocess", fake)
        assert ws.run_pro_synthesis("q") is None


def test_unit__bridge__venv_python_preferred(tmp_path, monkeypatch) -> None:
    root = _skill_root(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_SKILL_DIR", str(root))
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _fake_proc(stdout=json.dumps(FAKE_PAYLOAD))

    monkeypatch.setattr(ws, "subprocess", SimpleNamespace(run=fake_run))
    ws.run_pro_synthesis("q")
    assert seen["cmd"][0] == str(root / ".venv" / "bin" / "python")

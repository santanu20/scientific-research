"""Characterization tests for the pipeline CLI entry (P0-1).

Pins the CLI surface: flag -> ResearchConfig mapping, --json output shape,
exit codes, and the timestamped default output dir. No network, no Ollama:
run_pipeline is monkeypatched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pipeline  # noqa: E402
from config import ResearchConfig  # noqa: E402


def _capture_run(monkeypatch, results=None):
    calls: list[ResearchConfig] = []

    def fake_run(cfg, progress=None):
        calls.append(cfg)
        if isinstance(results, Exception):
            raise results
        return results or {}

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    return calls


def test_unit__cli_help__exits_zero_and_lists_core_flags():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPTS / "pipeline.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=_SKILL_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0
    for flag in ("--max", "--sources", "--use-llm", "--reflect", "--json", "--out-dir", "--budget"):
        assert flag in proc.stdout, f"missing flag in --help: {flag}"


def test_unit__cli_flags__map_onto_research_config(monkeypatch, tmp_path):
    calls = _capture_run(monkeypatch)
    rc = pipeline.main(
        [
            "magma ocean thermal evolution",
            "--max",
            "7",
            "--sources",
            "crossref,openalex",
            "--use-llm",
            "--llm-model",
            "qwen3.5:9b",
            "--llm-quality",
            "quality",
            "--from-year",
            "2020",
            "--to-year",
            "2024",
            "--open-access-only",
            "--type",
            "journal-article",
            "--reflect",
            "2",
            "--no-web-pro",
            "--agentic-web",
            "--no-local-pdfs",
            "--max-pdf",
            "3",
            "--budget",
            "600",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    cfg = calls[0]
    assert cfg.query == "magma ocean thermal evolution"
    assert cfg.max_papers == 7
    assert cfg.sources == ["crossref", "openalex"]
    assert cfg.use_llm is True
    assert cfg.llm_model == "qwen3.5:9b"
    assert cfg.llm_quality == "quality"
    assert cfg.year_from == 2020 and cfg.year_to == 2024
    assert cfg.open_access_only is True
    assert cfg.publication_type == "journal-article"
    assert cfg.reflect_iterations == 2
    assert cfg.use_web_pro is False
    assert cfg.use_web_search_agentic is True
    assert cfg.match_local_pdfs is False
    assert cfg.max_pdf_downloads == 3
    assert cfg.wall_clock_budget_s == 600.0
    assert str(cfg.output_dir) == str(tmp_path / "out")


def test_unit__cli_defaults__timestamped_out_dir_and_default_sources(monkeypatch):
    calls = _capture_run(monkeypatch)
    rc = pipeline.main(["climate feedbacks"])
    assert rc == 0
    cfg = calls[0]
    assert cfg.max_papers == 30
    assert cfg.reflect_iterations == 1
    assert cfg.use_web_pro is True
    assert "crossref" in cfg.sources and "openalex" in cfg.sources
    assert cfg.output_dir is not None
    out = Path(cfg.output_dir)
    assert "research_outputs" in str(out)
    assert out.name.startswith("pipeline-")  # pipeline-<ts>-<slug>, never clobbers
    assert out.is_dir()


def test_unit__cli_json__prints_results_without_brief_by_default(monkeypatch, capsys):
    canned = {
        "query": "q",
        "research_type": "exploratory",
        "n_papers": 5,
        "n_fulltext_matched": 2,
        "paths": {"brief": Path("/tmp/x/brief.md"), "corpus": Path("/tmp/x/corpus.json")},
        "brief_text": "# " + "x" * 5000,
        "elapsed": 12.3,
    }
    _capture_run(monkeypatch, results=canned)
    rc = pipeline.main(["q", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_papers"] == 5
    assert payload["paths"]["brief"] == "/tmp/x/brief.md"  # Path -> str via default=str
    assert "brief_text" not in payload  # 5 KB brief excluded unless --with-brief

    _capture_run(monkeypatch, results=canned)
    rc = pipeline.main(["q", "--json", "--with-brief"])
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["brief_text"].startswith("# ")


def test_unit__cli_busy__exits_two(monkeypatch, capsys):
    _capture_run(monkeypatch, results=pipeline.PipelineBusyError("busy"))
    rc = pipeline.main(["q"])
    assert rc == 2


def test_unit__cli_human_summary__prints_stats_and_paths(monkeypatch, capsys):
    canned = {
        "research_type": "exploratory",
        "n_papers": 5,
        "n_fulltext_matched": 2,
        "paths": {"brief": Path("/tmp/b.md")},
        "elapsed": 4.2,
    }
    _capture_run(monkeypatch, results=canned)
    rc = pipeline.main(["q"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "exploratory" in out and "/tmp/b.md" in out

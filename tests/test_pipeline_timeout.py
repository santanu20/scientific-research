"""Characterization tests for the wall-clock timeout contract.

Pin doctrine: a pipeline run that exceeds --budget must return a partial
result built from on-disk artifacts (never a raw traceback), release the
pipeline lock, and the CLI must exit 3. Found live 2026-10-07: the
PipelineTimeoutError docstring promised run_pipeline catches it — nothing
did (zero `except PipelineTimeout` in the tree).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _SKILL_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import pipeline  # noqa: E402
from config import ResearchConfig  # noqa: E402


def _cfg(tmp_path: Path) -> ResearchConfig:
    return ResearchConfig(query="timeout pin query", output_dir=tmp_path / "run")


def test_unit__timeout_partial_result__empty_dir_reports_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    exc = pipeline.PipelineTimeoutError(phase=2, elapsed=41.0, budget=30.0)
    partial = pipeline._timeout_partial_result(cfg, exc)

    assert partial["timed_out"] is True
    assert partial["timeout_phase"] == 2
    assert partial["query"] == cfg.query
    assert partial["research_type"] == "unknown"
    assert partial["n_papers"] == 0
    assert partial["brief_text"] is None
    assert all(p is None for p in partial["paths"].values())
    assert partial["elapsed"] == 41.0


def test_unit__timeout_partial_result__disk_truth_only(tmp_path):
    cfg = _cfg(tmp_path)
    rd = cfg.results_dir
    rd.mkdir(parents=True)
    (rd / "extracted.json").write_text(json.dumps({"extractions": [{}, {}, {}]}))
    (rd / "verified.json").write_text(json.dumps([{}, {}]))
    (rd / "research_brief.md").write_text("# partial brief")
    (rd / "research_context.json").write_text(json.dumps({"research_type": "verification"}))

    partial = pipeline._timeout_partial_result(cfg, pipeline.PipelineTimeoutError(5, 643.0, 600.0))

    assert partial["n_papers"] == 3
    assert partial["n_verified"] == 2
    assert partial["research_type"] == "verification"
    assert partial["brief_text"] == "# partial brief"
    assert partial["paths"]["brief"] == str(rd / "research_brief.md")
    assert partial["paths"]["corpus"] is None  # never written -> never reported
    assert partial["paths"]["extracted"] == str(rd / "extracted.json")


def test_unit__run_pipeline__timeout_returns_partial_and_releases_lock(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    def raise_timeout(config, progress=None):
        raise pipeline.PipelineTimeoutError(5, 643.0, 600.0)

    monkeypatch.setattr(pipeline, "_run_pipeline_impl", raise_timeout)
    partial = pipeline.run_pipeline(cfg)

    assert partial["timed_out"] is True
    assert partial["timeout_phase"] == 5

    # Lock must be released: an immediate second run succeeds.
    monkeypatch.setattr(pipeline, "_run_pipeline_impl", lambda config, progress=None: {"ok": True})
    assert pipeline.run_pipeline(cfg) == {"ok": True}


def test_unit__run_pipeline__timeout_fires_completion_callback(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    seen: list[dict] = []
    monkeypatch.setattr(
        pipeline,
        "_run_pipeline_impl",
        lambda config, progress=None: (_ for _ in ()).throw(pipeline.PipelineTimeoutError(3, 99.0, 90.0)),
    )
    monkeypatch.setattr(pipeline, "_notify_research_complete", seen.append)

    pipeline.run_pipeline(cfg)

    assert len(seen) == 1 and seen[0]["timed_out"] is True


def test_unit__cli__timeout_exits_3_and_prints_partial_json(monkeypatch, tmp_path, capsys):
    def fake_run(config, progress=None):
        return {
            "query": config.query,
            "research_type": "unknown",
            "n_papers": 0,
            "paths": {"brief": str(config.results_dir / "research_brief.md")},
            "elapsed": 643.0,
            "timed_out": True,
            "timeout_phase": 5,
        }

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)

    rc = pipeline.main(["timeout question", "--json", "--out-dir", str(tmp_path / "run")])

    out = json.loads(capsys.readouterr().out)
    assert rc == 3
    assert out["timed_out"] is True
    assert out["timeout_phase"] == 5

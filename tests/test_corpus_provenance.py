"""Pins for corpus provenance (2026-09-03): discovery-route accounting.

Discovery is nondeterministic (web backends rotate; identical queries
yield different corpora run-to-run) — the route each paper took
(primary discovery, typo-repair rediscovery, rescue, KB) must be
recorded on the record and rendered deterministically in the brief.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_unit__provenance__record_field_defaults_and_serializes() -> None:
    from _sources import PaperRecord

    rec = PaperRecord(doi="10.1/x", title="T")
    assert rec.discovery_provenance == ""
    d = rec.to_dict()
    assert "discovery_provenance" in d
    assert PaperRecord.from_dict(d).discovery_provenance == ""


def test_unit__provenance__renderer_route_and_fallback() -> None:
    from _narrative import render_corpus_provenance

    md = render_corpus_provenance(
        [
            {
                "title": "Gondwana sedimentation review",
                "year": 2003,
                "authors": [{"name": "Sengupta, S."}],
                "discovery_provenance": "discovery:typo-repair-rediscovery",
            },
            {
                "title": "An untagged legacy record",
                "year": 2015,
                "authors": [],
                "source": "openalex",
            },
        ]
    )
    assert "## Corpus provenance" in md
    assert "`discovery:typo-repair-rediscovery`" in md
    assert "`source:openalex`" in md  # fallback when no route recorded
    assert "(2003) Gondwana sedimentation review" in md
    assert "Anonymous (2015)" in md


def test_unit__provenance__set_sites_wired() -> None:
    """All six route-tagging sites exist in the pipeline source."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "pipeline.py").read_text(encoding="utf-8")
    for route in (
        "discovery:primary",
        "discovery:typo-repair-rediscovery",
        "supplement:web-sparse",
        "escalation:agentic-crawl",
        "rescue:web-relevance",
        "kb:local",
    ):
        assert f'"{route}"' in src, route

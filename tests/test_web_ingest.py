"""Pins for web-search supplement ingestion hygiene (B1-F4)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from discover import _clean_harvested_doi, _content_tokens, _query_overlap_ok


def _rec(title: str, abstract: str = ""):
    return type("R", (), {"title": title, "abstract": abstract})()


def test__doi_cleanup_strips_trailing_junk():
    assert _clean_harvested_doi('10.1007/s123-456.,;)') == "10.1007/s123-456"
    assert _clean_harvested_doi("10.1038/NATURE.PDF") == "10.1038/nature.pdf"


def test__citation_noise_rejected_by_overlap_gate():
    toks = _content_tokens("Groundwater lineament analysis for exploration")
    noise = _rec("Quantum chromodynamics on the lattice")
    relevant = _rec(
        "Lineament-based groundwater prospects in hard rock",
        "Remote sensing lineament mapping for groundwater.",
    )
    assert not _query_overlap_ok(noise, toks)
    assert _query_overlap_ok(relevant, toks)


def test__gate_passes_when_no_tokens():
    # Degenerate query must never hard-block the supplement.
    assert _query_overlap_ok(_rec("anything"), set()) is True


def test__h20_gate_blocks_unresolved_in_synthesis_merge():
    from _artifact import _extract_papers

    verified = {
        "papers": [
            {"doi": "10.1/a", "title": "Resolved paper", "verification_status": "resolved"},
            {"doi": "10.2/b", "title": "Unresolved paper", "verification_status": "unresolved"},
        ]
    }
    papers = _extract_papers(None, verified)
    titles = [p["title"] for p in papers]
    assert "Resolved paper" in titles
    assert "Unresolved paper" not in titles


def test__paperrecord_default_status_empty():
    from _sources import PaperRecord

    assert PaperRecord(title="x").verification_status == ""

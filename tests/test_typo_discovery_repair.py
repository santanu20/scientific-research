"""Regression pins for the typo-discovery repair batch (2026-09-03).

Live case: query "prahnita godawari sedimentary chandrapur" (two typos)
returned 20 off-topic candidates and the honesty gate correctly refused
synthesis — because every SEARCH string carried the typos verbatim. No
offline dictionary knew the basin name (geodict + 231 local-library
titles), the Wikipedia corrector scored whole titles only ('prahnita'
vs 'Pranhita River' = 0.73 < 0.75 while the head token is DL1 = 0.875),
and no discovery-feedback loop existed to repair a zero-match term.

Four fixes pinned here (conventions of test_honesty_gate_fixes.py /
test_websearch_and_fixes.py):
  A1  _query._wikipedia_correct_proper_nouns — head-token DL1 repair
  A2  pipeline._phase_discovery — zero-match web-vocab repair + one
      corrected rediscovery round
  A3  context built from the typo-repaired audit_query; rescue paths
      never re-search the raw typo'd string
  A4  discover._query_overlap_ok — fuzzy DL1 tier so typo'd tokens still
      count as overlap for web-harvested papers
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SRC = _SCRIPTS

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_LIVE_QUERY = "prahnita godawari sedimentary chandrapur"
_CORRECTED_QUERY = "pranhita godavari sedimentary chandrapur"
_OFFTOPIC_TITLE = "Sedimentary ripple marks from pavements of forts in Jaipur, Rajasthan"
_TARGET_TITLE = (
    "Sedimentology of a Proterozoic erg: the Venkatpur Sandstone, "
    "Pranhita\u2010Godavari Valley, south India"  # U+2010 hyphen, as deposited
)
_TARGET_ABSTRACT = "aeolian facies and strata of the Pranhita-Godavari Gondwana basin"


# ── A1: Wikipedia head-token repair ───────────────────────────────────────

_TYPO_TOKEN = "prahnita"  # p-r-a-h-n-i-t-a: 'pranhita' with nh/hn transposed


class _FakeResp:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _opensearch(titles: list[str]) -> object:
    return ["ctx", titles, [""] * len(titles), [""] * len(titles)]


def test_unit__wiki_head_repair__river_titles_fix_both_typos(monkeypatch) -> None:
    """'prahnita'/'godawari' must repair against 'Pranhita River'/
    'Godavari River' heads — the live failure mode (whole-title ratio
    0.73 < 0.75 could never fire; head token is DL1)."""
    import urllib.request

    from _query import _wikipedia_correct_proper_nouns

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeResp(_opensearch(["Pranhita River", "Godavari River"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    corrected, notes = _wikipedia_correct_proper_nouns("prahnita godawari")
    assert corrected == "pranhita godavari"
    assert "prahnita → pranhita" in notes
    assert "godawari → godavari" in notes


def test_unit__wiki_head_repair__unrelated_titles_no_op(monkeypatch) -> None:
    import urllib.request

    from _query import _wikipedia_correct_proper_nouns

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeResp(_opensearch(["Nagaur Tertiary Interface", "Bishnumati River"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    corrected, notes = _wikipedia_correct_proper_nouns("prahnita godawari")
    assert corrected == "prahnita godawari"
    assert notes == []


def test_unit__wiki_head_repair__already_correct_word_unchanged(monkeypatch) -> None:
    """Head equal to the word (must-differ guard) — 'chandrapur' against
    'Chandrapur'/'Chandrapur district' must not churn the query."""
    import urllib.request

    from _query import _wikipedia_correct_proper_nouns

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeResp(_opensearch(["Chandrapur", "Chandrapur district"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    corrected, notes = _wikipedia_correct_proper_nouns("chandrapur sedimentary")
    assert corrected == "chandrapur sedimentary"
    assert notes == []


def test_unit__wiki_head_repair__ambiguous_forms_skipped(monkeypatch) -> None:
    """Two DISTINCT DL1 heads = ambiguous — no repair (unique-candidate
    guard, same contract as spellcorrect_query). Multi-word titles keep
    the legacy whole-title ratio below its 0.90 direct-accept, so this
    pins the head-path guard in isolation."""
    import urllib.request

    from _query import _wikipedia_correct_proper_nouns

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        # 'chandrpur' is DL1 from BOTH heads, but they differ from each other
        return _FakeResp(
            _opensearch(["Chandrapur District Gazetteer", "Chandarpur Tehsil Records"])
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    corrected, notes = _wikipedia_correct_proper_nouns("chandrpur dyke")
    assert corrected == "chandrpur dyke"
    assert notes == []


def test_unit__wiki_head_repair__identity_head_vetoes_near_neighbor(monkeypatch) -> None:
    """Live false positive (2026-09-03): OpenSearch returned 'Chandrapur'
    AND 'Chandrapura' (a different town, DL1) — the word existing verbatim
    as a head must veto repair, not just be excluded from candidates."""
    import urllib.request

    from _query import _wikipedia_correct_proper_nouns

    def fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeResp(_opensearch(["Chandrapur", "Chandrapura"]))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    corrected, notes = _wikipedia_correct_proper_nouns("chandrapur dyke")
    assert corrected == "chandrapur dyke"
    assert notes == []


def test_unit__spellcorrect__web_vocab_repairs_basin_typos() -> None:
    """The web-vocab repair mechanism in isolation: result titles/bodies
    carry the intended spellings ('Pranhita-Godavari Basin') and DL1
    unique-candidate correction recovers both terms."""
    from _query import spellcorrect_query

    web_vocab = [
        "Pranhita-Godavari Basin gondwana sedimentary chandrapur geology",
        "proterozoic strata of the pranhita godavari valley",
    ]
    corrected, notes = spellcorrect_query(_LIVE_QUERY, extra_vocab=web_vocab)
    assert corrected == _CORRECTED_QUERY
    assert sorted(notes) == ["godawari -> godavari", "prahnita -> pranhita"]


# ── A2/A3: pipeline wiring (source-level, repo convention) ───────────────


def test_unit__pipeline__zero_match_repair_wired() -> None:
    src = (_SCRIPTS / "pipeline.py").read_text(encoding="utf-8")
    assert "Zero-match term repair" in src
    assert "from _websearch import search_text" in src  # dual-mode shim import
    # corrected tokens replace the raw query inside the retry round
    assert "qs = qs.with_token_fixes(" in src
    # one retry round, bounded budget, fail-open
    assert "call_budget=40" in src


def test_unit__pipeline__context_built_from_repaired_query() -> None:
    """Strategies/aliases must derive from the typo-repaired audit_query,
    not the raw config.query (split-brain regression pin)."""
    src = (_SCRIPTS / "pipeline.py").read_text(encoding="utf-8")
    assert "build_research_context(qs.audit)" in src


def test_unit__pipeline__rescues_use_repaired_query() -> None:
    """Sparse supplement / agentic escalation / relevance-emptiness rescue
    must never re-search the raw typo'd string."""
    src = (_SCRIPTS / "pipeline.py").read_text(encoding="utf-8")
    assert re.search(r"web_search_paper_discovery\(\s*qs.audit", src)
    assert re.search(r"web_search_agentic_discovery\(\s*qs.audit", src)
    assert "rescue, qs.audit, alias_map=alias_map" in src


# ── A4: fuzzy overlap gate for web-harvested papers ──────────────────────


def test_unit__query_overlap__typo_token_counts_as_overlap() -> None:
    """'godawari' (1-edit typo) must overlap 'Godavari basin' text — the
    hard-drop site for correctly-harvested papers."""
    from discover import _query_overlap_ok

    rec = SimpleNamespace(
        title="Sedimentation history of the Godavari basin",
        abstract="",
    )
    assert _query_overlap_ok(rec, {"godawari", "xyzzyy"}) is True


def test_unit__query_overlap__unrelated_token_still_rejected() -> None:
    from discover import _query_overlap_ok

    rec = SimpleNamespace(
        title="Sedimentation history of the Godavari basin",
        abstract="",
    )
    assert _query_overlap_ok(rec, {"khammam"}) is False


# ── A2: end-to-end _phase_discovery repair (mocked IO, real logic) ───────


def test_unit__phase_discovery__zero_match_repairs_and_rediscovers(
    monkeypatch, tmp_path
) -> None:
    """Full Phase-1 story: typo'd round returns an off-topic single-term
    candidate; the zero-match terms trigger web-vocab repair; the
    corrected retry round discovers the Pranhita-Godavari paper."""
    import _context
    import _query
    import _ranking
    import discover
    from _sources import PaperRecord

    from config import ResearchConfig
    from pipeline import _phase_discovery

    search_calls: list[str] = []
    emits: list[str] = []

    def fake_search_multi_source(query, **kwargs):  # noqa: ANN001, ANN003
        search_calls.append(query)
        if _TYPO_TOKEN in query:  # typo spelling present = unrepaired round
            # typo'd round -> the LIVE failure shape: one abstract-less
            # paper whose only pranhita mention the ranker will discard,
            # plus an off-topic single-term paper with an abstract.
            return [
                PaperRecord(
                    doi="10.9999/noabs",
                    title="Pranhita valley stratigraphy notes",
                    abstract="",  # ranker discards abstract-less papers
                    source="crossref",
                ),
                PaperRecord(
                    doi="10.9999/offtopic",
                    title=_OFFTOPIC_TITLE,
                    abstract="ripple marks on fort pavements",
                    source="crossref",
                ),
            ]
        return [
            PaperRecord(
                doi="10.9999/target",
                title=_TARGET_TITLE,
                abstract=_TARGET_ABSTRACT,
                source="crossref",
            )
        ]

    def fake_web_text(query, **kwargs):  # noqa: ANN001, ANN003
        return 0, {
            "results": [
                {
                    "title": "Pranhita-Godavari Basin gondwana sedimentary chandrapur",
                    "body": "proterozoic strata of the pranhita godavari valley",
                }
            ]
        }

    def fake_ctx(query, **kwargs):  # noqa: ANN001, ANN003
        return SimpleNamespace(
            intent="subtopic",
            domain="sedimentary",
            niche=False,
            entities=[SimpleNamespace(term=t, is_primary=False) for t in query.split()],
            search_strategies=[query],
            alias_map={},
            screening_terms=[],
            source_notes=[],
            to_dict=lambda: {},
        )

    import _sources

    monkeypatch.setattr(discover, "search_multi_source", fake_search_multi_source)
    monkeypatch.setattr(discover, "_web_text_engine", fake_web_text)
    monkeypatch.setattr(_context, "build_research_context", fake_ctx)
    monkeypatch.setattr(_sources, "openalex_get_by_doi", lambda doi: None)
    monkeypatch.setattr(
        _query, "normalize_query", lambda q, **k: (q, [], [])  # noqa: ARG005
    )
    monkeypatch.setattr(
        _ranking,
        "semantic_relevance_scores",
        lambda q, papers: __import__("numpy").zeros(len(papers)),
    )

    # max_papers=1 forces the ranker to keep ONLY the with-abstract
    # off-topic paper — the pranhita mention is discarded, so the kept
    # pool has zero pranhita and the (relocated) repair gate must fire.
    cfg = ResearchConfig(query=_LIVE_QUERY, max_papers=1)
    papers, qs, rtype, ctx, notes = _phase_discovery(
        cfg, tmp_path, lambda n, m: emits.append(m)
    )

    # First round searched the typo verbatim; retry round searched corrected
    assert any("prahnita" in c for c in search_calls), search_calls
    assert any(
        "pranhita" in c and "godavari" in c for c in search_calls
    ), search_calls
    # The corrected retry must actually re-search, not just re-screen
    assert sum(1 for c in search_calls if "pranhita" in c) >= 1

    # Typo repair was loud
    assert any("Query typo repaired via web vocabulary" in m for m in emits), emits
    assert any("prahnita → pranhita" in m and "godawari → godavari" in m for m in emits), emits

    # QueryState contract: audit/search repaired, raw preserved verbatim,
    # fixes carry the provenance notes
    assert qs.raw == _LIVE_QUERY
    assert qs.audit == _CORRECTED_QUERY
    assert "pranhita" in qs.search and "prahnita" not in qs.search
    assert any("prahnita → pranhita" in f for f in qs.fixes), qs.fixes
    assert any("godawari → godavari" in f for f in qs.fixes), qs.fixes
    # The target paper is in the final candidate pool

    # Corrected-rediscovery papers carry the provenance route
    rediscovered = [p for p in papers if "Pranhita" in (p.title or "")]
    assert rediscovered, titles
    assert all(
        p.discovery_provenance == "discovery:typo-repair-rediscovery"
        for p in rediscovered
    ), [p.discovery_provenance for p in rediscovered]
    titles = [p.title or "" for p in papers]
    assert any("Pranhita" in t for t in titles), titles
    assert any("Godavari" in t or "Godāvari" in t for t in titles), titles


def test_unit__query_state__frozen_and_token_fix_propagation() -> None:
    """QueryState is immutable; with_token_fixes repairs audit+search,
    preserves raw verbatim, accumulates provenance notes."""
    from _query import QueryState

    qs = QueryState(raw=_LIVE_QUERY, audit=_LIVE_QUERY, search="expanded " + _LIVE_QUERY)
    fixed = qs.with_token_fixes(
        {"prahnita": "pranhita", "godawari": "godavari"},
        notes=["prahnita → pranhita (test)"],
    )
    assert qs.audit == _LIVE_QUERY  # original untouched
    assert fixed.raw == _LIVE_QUERY  # provenance: raw never repaired
    assert fixed.audit == _CORRECTED_QUERY
    assert fixed.search == "expanded " + _CORRECTED_QUERY
    assert fixed.fixes == ("prahnita → pranhita (test)",)
    with pytest.raises(Exception):  # frozen dataclass
        fixed.audit = "mutated"  # type: ignore[misc]


# ── A5: pdf_ocr page-range clamp (full-text path, found live) ────────────


def test_unit__zero_match_recall__correctly_spelled_term_round(monkeypatch, tmp_path) -> None:
    """Correctly-spelled zero-match term (gadchiroli — not a typo): the
    recall round searches term+subject+domain-anchor and merges keepers
    through the paired gate, tagged 'discovery:zero-term-recall'."""
    import _context
    import _query as _q
    import _ranking
    import discover
    from _sources import PaperRecord
    from config import ResearchConfig

    from pipeline import _phase_discovery

    calls: list[str] = []

    def fake_sms(query, **kwargs):
        calls.append(query)
        if "geochemistry" in query:  # recall query carries the domain anchor
            return [
                PaperRecord(
                    doi="10.9999/wairagarh",
                    title="Geochemical Characteristics of Mafic Dykes from Wairagarh Area",
                    abstract="Mafic dykes of the Gadchiroli district, Bastar craton.",
                    source="openalex",
                )
            ]
        return [
            PaperRecord(
                doi="10.9999/off",
                title="Sedimentary ripple marks in Jaipur",
                abstract="ripple marks",
                source="crossref",
            )
        ]

    monkeypatch.setattr(discover, "search_multi_source", fake_sms)
    monkeypatch.setattr(
        _context, "build_research_context",
        lambda q, **k: SimpleNamespace(
            intent="subtopic", domain="igneous", niche=True, entities=[],
            search_strategies=[q], alias_map={}, screening_terms=[],
            source_notes=[], to_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(_q, "normalize_query", lambda q, **k: (q, [], []))
    import _sources as _so

    monkeypatch.setattr(_so, "openalex_get_by_doi", lambda doi: None)
    monkeypatch.setattr(
        _ranking, "semantic_relevance_scores",
        lambda q, papers: __import__("numpy").zeros(len(papers)),
    )

    cfg = ResearchConfig(query="dykes age exposed in gadchiroli", max_papers=10)
    papers, qs, *_ = _phase_discovery(cfg, tmp_path, lambda n, m: None)
    recalled = [p for p in papers if p.discovery_provenance == "discovery:zero-term-recall"]
    assert recalled, [p.title for p in papers]
    assert "Wairagarh" in recalled[0].title
    assert any("gadchiroli" in c and "dyke" in c for c in calls), calls

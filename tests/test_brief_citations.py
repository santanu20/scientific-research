"""Contract test: brief body [N] citations must align with References [N].

Diagnosed bug: synthesized brief uses theme-offset numbering in the body
(``ref_offset + len(theme_cited) + 1``) but the References section uses
simple linear iteration. They CAN'T align by construction.

This contract test:
  1. Fails (xfail strict) on the current amphibole thermometery brief
     (demonstrating the bug)
  2. Will pass once the synthesis is fixed to use a single citation index
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_DATA = Path(__file__).resolve().parents[3] / "data" / "research"


# =============================================================================
# Helper unit tests — verify_citation_alignment on synthetic briefs
# =============================================================================


class TestVerifyCitationAlignment:
    def test_well_formed_brief_passes(self):
        from _citation_check import verify_citation_alignment

        brief = """# Brief

Body cites [1] Smith 2020 and [2] Jones 2021 for their findings.

## References

[1] Smith, J. (2020). Amphibole thermometry of Adirondack rocks. DOI:10.1000/smith

[2] Jones, K. (2021). Garnet-pyroxene thermobarometry. DOI:10.1000/jones
"""
        mismatches = verify_citation_alignment(brief)
        assert mismatches == [], f"Expected no mismatches, got: {mismatches}"

    def test_missing_reference_detected(self):
        from _citation_check import verify_citation_alignment

        brief = """# Brief

Body cites [3] which has no reference entry.

## References

[1] Smith, J. (2020). Some paper.
"""
        mismatches = verify_citation_alignment(brief)
        assert len(mismatches) == 1
        assert mismatches[0].ref_num == 3
        assert "no entry" in mismatches[0].reason

    def test_misaligned_citation_detected(self):
        from _citation_check import verify_citation_alignment

        # Body [1] discusses amphibole, ref [1] discusses blockchain — mismatch
        # Requires strict_word_overlap=True because structural check (existence)
        # alone can't detect topical misalignment.
        brief = """# Brief

We find [1] amphibole thermometry results consistent with prior work.

## References

[1] Nakamoto, S. (2008). Bitcoin: A peer-to-peer electronic cash system.
"""
        mismatches = verify_citation_alignment(brief, strict_word_overlap=True)
        assert len(mismatches) == 1
        assert mismatches[0].ref_num == 1
        assert "different papers" in mismatches[0].reason

    def test_missing_references_section_detected(self):
        from _citation_check import verify_citation_alignment

        brief = """# Brief

No references section here.
"""
        mismatches = verify_citation_alignment(brief)
        assert len(mismatches) == 1
        assert "References" in mismatches[0].reason

    def test_permissive_with_title_overlap(self):
        from _citation_check import verify_citation_alignment

        # Body [1] fragment contains "amphibole" which is in ref title — pass
        # even with strict_word_overlap=True.
        brief = """# Brief

The [1] amphibole data shows clear trends.

## References

[1] Smith, J. (2020). Amphibole thermometry of Adirondack rocks.
"""
        mismatches = verify_citation_alignment(brief, strict_word_overlap=True)
        assert mismatches == []

    def test_strict_doi_mismatch(self):
        from _citation_check import verify_citation_alignment

        brief = """# Brief

The [1] amphibole data (DOI:10.1000/wrong) shows clear trends.

## References

[1] Smith, J. (2020). Amphibole thermometry. DOI:10.1000/right
"""
        mismatches = verify_citation_alignment(brief, require_doi_match=True)
        assert len(mismatches) == 1
        assert "DOI differs" in mismatches[0].reason


# =============================================================================
# Contract test on the actual amphibole brief — REGRESSION GUARD
# =============================================================================


class TestAmphiboleBriefContract:
    """Verify the actual saved amphibole brief is well-formed.

    The saved brief was regenerated on 2026-07-14 after the synthesis
    code was fixed to use a unified citation index. Earlier versions
    had body [N] numbering that diverged from References [N] (body
    went up to [98], References only to [76]).
    """

    BRIEF_PATH = _DATA / "1d59ece550032c91" / "research_brief.md"

    def test_amphibole_brief_citations_align(self):
        """Every [N] in body must map to same paper in References [N]."""
        if not self.BRIEF_PATH.exists():
            pytest.skip(f"brief fixture missing: {self.BRIEF_PATH}")
        from _citation_check import verify_citation_alignment

        brief = self.BRIEF_PATH.read_text(encoding="utf-8")
        mismatches = verify_citation_alignment(brief)
        assert mismatches == [], (
            f"Found {len(mismatches)} citation misalignments in amphibole brief. "
            f"First 3:\n" + "\n".join(str(m) for m in mismatches[:3])
        )

    def test_amphibole_brief_body_max_within_references_range(self):
        """body max [N] must not exceed References max [N]."""
        import re

        if not self.BRIEF_PATH.exists():
            pytest.skip(f"brief fixture missing: {self.BRIEF_PATH}")
        brief = self.BRIEF_PATH.read_text(encoding="utf-8")
        parts = re.split(r"^##\s+References\s*$", brief, maxsplit=1, flags=re.MULTILINE)
        if len(parts) != 2:
            pytest.skip("brief has no References section")
        body, refs = parts
        body_max = max(int(n) for n in re.findall(r"\[(\d+)\]", body))
        ref_max = max(int(n) for n in re.findall(r"^\[(\d+)\]", refs, re.MULTILINE))
        assert body_max <= ref_max, (
            f"body cites [{body_max}] but References only goes to [{ref_max}] — "
            f"misalignment present"
        )


# =============================================================================
# CI-friendly contract test: synthesize a brief in-test from a small fixture
# corpus. No dependency on saved artifacts (which can drift). Verifies the
# synthesis code ITSELF produces aligned citations.
# =============================================================================


def _make_fixture_corpus() -> list[dict]:
    """Tiny fixture corpus — 5 synthetic papers with realistic structure.

    Designed to exercise the theme-grouping + integrative synthesis path
    without requiring network access or large fixtures.
    """
    return [
        {
            "paper_id": "doi:10.1/amphibole-1",
            "doi": "10.1/amphibole-1",
            "title": "Hornblende geothermometry of Adirondack amphibolites",
            "abstract": "We report amphibole thermometry on 12 samples from the Adirondack Lowlands, yielding temperatures of 600-750 degrees Celsius.",
            "key_finding": "Amphibole thermometry yields temperatures of 600-750 °C for Adirondack amphibolites.",
            "year": 2020,
            "authors": [{"name": "Smith, J."}, {"name": "Jones, K."}],
            "measurements": [
                {"value": 600, "unit": "°C", "measurement": "temperature", "raw_text": "600 °C"},
                {"value": 750, "unit": "°C", "measurement": "temperature", "raw_text": "750 °C"},
            ],
        },
        {
            "paper_id": "doi:10.2/amphibole-2",
            "doi": "10.2/amphibole-2",
            "title": "Amphibole endmember geothermobarometry in metabasites",
            "abstract": "We calibrate a new amphibole-plagioclase thermometer and apply it to metabasite samples, finding pressures of 6-8 kbar.",
            "key_finding": "New amphibole-plagioclase calibration gives P=6-8 kbar for metabasites.",
            "year": 2022,
            "authors": [{"name": "Lee, H."}, {"name": "Brown, M."}],
            "measurements": [
                {"value": 6, "unit": "kbar", "measurement": "pressure", "raw_text": "6 kbar"},
                {"value": 8, "unit": "kbar", "measurement": "pressure", "raw_text": "8 kbar"},
            ],
        },
        {
            "paper_id": "doi:10.3/cpx-1",
            "doi": "10.3/cpx-1",
            "title": "Clinopyroxene-garnet Fe-Mg exchange thermometry",
            "abstract": "We apply garnet-clinopyroxene Fe-Mg exchange thermometry to eclogite samples, finding temperatures of 700-900 degrees Celsius.",
            "key_finding": "Garnet-clinopyroxene thermometry on eclogites yields T=700-900 °C.",
            "year": 2021,
            "authors": [{"name": "Wang, L."}, {"name": "Chen, Y."}],
            "measurements": [
                {"value": 700, "unit": "°C", "measurement": "temperature", "raw_text": "700 °C"},
                {"value": 900, "unit": "°C", "measurement": "temperature", "raw_text": "900 °C"},
            ],
        },
        {
            "paper_id": "doi:10.4/cpx-2",
            "doi": "10.4/cpx-2",
            "title": "Revised activity-composition models for clinopyroxene",
            "abstract": "We present revised a-x models for clinopyroxene applicable to mafic assemblages.",
            "key_finding": "Revised clinopyroxene a-x models improve phase-equilibrium calculations.",
            "year": 2019,
            "authors": [{"name": "Powell, R."}, {"name": "Holland, T."}],
        },
        {
            "paper_id": "doi:10.5/method-1",
            "doi": "10.5/method-1",
            "title": "Uncertainty analysis in non-equilibrium geothermobarometry",
            "abstract": "We present a statistical framework for propagating uncertainty through thermobarometric calculations.",
            "key_finding": "Statistical propagation of thermobarometric uncertainty requires Monte Carlo methods.",
            "year": 2023,
            "authors": [{"name": "Carlson, W."}],
        },
    ]


class TestFreshSynthesisAlignment:
    """Contract test: freshly synthesized brief must have aligned citations.

    No dependency on saved artifacts — builds a tiny fixture corpus in-test,
    runs the actual synthesis, verifies alignment. Catches regressions in
    the synthesis code itself.
    """

    def test_fresh_brief_has_no_structural_mismatches(self):
        """Every [N] in body must have a References [N] entry."""
        from _citation_check import verify_citation_alignment
        from _narrative import build_chronological_narrative, format_citation_list

        papers = _make_fixture_corpus()
        narrative, cited = build_chronological_narrative(papers, "amphibole thermometry", "survey")
        refs = format_citation_list(cited)
        brief = f"{narrative}\n\n## References\n\n{refs}"
        mismatches = verify_citation_alignment(brief)
        assert mismatches == [], (
            f"Fresh synthesis has {len(mismatches)} structural mismatches:\n"
            + "\n".join(str(m) for m in mismatches[:3])
        )

    def test_fresh_brief_body_max_within_references_range(self):
        """body max [N] ≤ References max [N]."""
        import re

        from _narrative import build_chronological_narrative, format_citation_list

        papers = _make_fixture_corpus()
        narrative, cited = build_chronological_narrative(papers, "amphibole thermometry", "survey")
        refs = format_citation_list(cited)
        brief = f"{narrative}\n\n## References\n\n{refs}"
        parts = re.split(r"^##\s+References\s*$", brief, maxsplit=1, flags=re.MULTILINE)
        assert len(parts) == 2, "References section missing"
        body, ref_section = parts
        body_cites = [int(n) for n in re.findall(r"\[(\d+)\]", body)]
        ref_nums = [int(n) for n in re.findall(r"^\[(\d+)\]", ref_section, re.MULTILINE)]
        if body_cites and ref_nums:
            assert max(body_cites) <= max(ref_nums), (
                f"body max [{max(body_cites)}] > references max [{max(ref_nums)}]"
            )

    def test_fresh_brief_references_are_continuous(self):
        """References must be [1], [2], [3], ... with no gaps."""
        import re

        from _narrative import build_chronological_narrative, format_citation_list

        papers = _make_fixture_corpus()
        narrative, cited = build_chronological_narrative(papers, "amphibole thermometry", "survey")
        refs = format_citation_list(cited)
        ref_nums = sorted(int(n) for n in re.findall(r"^\[(\d+)\]", refs, re.MULTILINE))
        if ref_nums:
            expected = list(range(1, ref_nums[-1] + 1))
            assert ref_nums == expected, (
                f"References numbering has gaps: {ref_nums} (expected {expected})"
            )

    def test_fresh_brief_cited_count_matches_references_count(self):
        """Number of cited papers must equal number of reference entries."""
        import re

        from _narrative import build_chronological_narrative, format_citation_list

        papers = _make_fixture_corpus()
        _narrative, cited = build_chronological_narrative(papers, "amphibole thermometry", "survey")
        refs = format_citation_list(cited)
        ref_count = len(re.findall(r"^\[\d+\]", refs, re.MULTILINE))
        assert ref_count == len(cited), (
            f"cited list has {len(cited)} papers but References has {ref_count} entries"
        )


class TestCitationHonestyFixes:
    """Pins for the 2026-09-03 PG-basin brief defects: fabricated author
    fallback ('Sedimentary (1998)' — title word mined as a surname) and
    comma-list citations ('[2, 10]') leaking verbatim past both [N]
    sweeps."""

    def test_empty_authors_render_anonymous_never_title_mined(self) -> None:
        from _narrative import format_citation_list

        refs = format_citation_list(
            [
                {
                    "title": "Sedimentary environmental model for Lower Gondwana "
                    "sediments around Bellampalli",
                    "year": 1998,
                    "authors": [],
                    "doi": "10.1016/s0899-5362(97)83493-8",
                }
            ]
        )
        assert "Anonymous (1998)" in refs
        assert "Sedimentary (" not in refs

    def test_comma_list_bracket_regex_at_both_sweeps(self) -> None:
        """Pass-4a grounding and the final _humanize sweep must both match
        citation GROUPS ('[2, 10]'), not only single [N]."""
        src = (_SCRIPTS / "_ollama_extract.py").read_text(encoding="utf-8")
        assert src.count(r"\[(\d+(?:\s*,\s*\d+)*)\]") == 2

    def test_title_mining_fallback_deleted(self) -> None:
        """_extract_author (capitalized-title-word 'surname' mining) is
        gone; the fallback is the honest sentinel."""
        src = (_SCRIPTS / "_ollama_extract.py").read_text(encoding="utf-8")
        assert "_extract_author" not in src
        assert 'or "Anonymous"' in src
        assert '"Unknown", "", "Anonymous"' in src

    def test_authorless_paper_with_signal_kept_and_numbered(self) -> None:
        """Anonymous papers with abstract/key_finding stay in the corpus
        (no data loss) and ref_ids stay continuous after any skip."""
        from _ollama_extract import _build_papers_data

        extractions = [
            {
                "title": "A real study of basin evolution",
                "doi": "10.1/a",
                "year": 1998,
                "authors": [],
                "abstract": "We describe sedimentary environments in detail.",
                "key_finding": "Facies shift recorded.",
            },
            {  # junk: no author, no signal -> skipped, must not eat [2]
                "title": "Journal of Nothing volume metadata",
                "doi": "10.1/junk",
                "year": 2001,
                "authors": [],
            },
            {
                "title": "Another real study of provenance",
                "doi": "10.1/b",
                "year": 2015,
                "first_author": "Amarasinghe",
                "authors": [{"name": "Amarasinghe, A."}],
                "abstract": "Zircon ages described.",
            },
        ]
        papers, refs, _vals = _build_papers_data(extractions, "basin evolution")
        ids = [p["ref_id"] for p in papers]
        assert ids == [1, 2], ids
        assert papers[0]["author"] == "Anonymous"
        assert refs[0].startswith("[1] Anonymous (1998)")
        assert refs[1].startswith("[2] Amarasinghe")

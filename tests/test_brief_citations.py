#!/usr/bin/env python3
"""Contract tests for brief citation alignment (`_citation_check.py`).

Verifies that `verify_citation_alignment()` correctly detects:
  1. Well-formed brief → empty mismatch list
  2. DOI mismatch between body [N] and References [N] → CitationMismatch
  3. Missing `## References` section → single mismatch with ref_num=-1
  4. Body cites [N] not in References → mismatch with reason
  5. Loose word-overlap heuristic (strict_word_overlap=True) flags disjoint papers

Run with:
    cd ~/.config/opencode/skills/scientific-research
    uv run pytest tests/test_brief_citations.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable (matches tests/test_skill.py pattern)
SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from _citation_check import (
    CitationMismatch,
    verify_citation_alignment,
)
from _narrative import build_chronological_narrative, format_citation_list

# =============================================================================
# Fixtures
# =============================================================================

ALIGNED_BRIEF = """\
# Research Brief: Nanometre-scale thermometry

Recent work demonstrates nanoscale temperature sensing [1].

## References

[1] G. Kucsko et al. (2013). Nanometre-scale thermometry in a living cell. Nature. DOI: 10.1038/nature12373
"""

DOI_MISMATCH_BRIEF = """\
# Research Brief: Diamond thermometry

Recent work [1] (DOI: 10.1038/nature12373) shows unexpected results.

## References

[1] Different Author (2020). Unrelated paper. DOI: 10.1000/fake1234
"""

MISSING_REFS_BRIEF = """\
# Research Brief: No references section

Body cites [1] but there is no References heading.
"""

BODY_CITES_MISSING_REF = """\
# Research Brief: Orphan citation

Body cites [1] and [2] but only [1] is in references.

## References

[1] A. Author (2020). Real paper. DOI: 10.1000/real1
"""

DISJOINT_WORD_OVERLAP_BRIEF = """\
# Research Brief: Disjoint topics

The study [1] reports zirconium isotope ratios showing unexpected fractionation patterns.

## References

[1] B. Biologist (2019). Mitochondrial DNA analysis of marine invertebrates.
"""


# =============================================================================
# Tests
# =============================================================================


class TestVerifyCitationAlignment:
    """Contract: every [N] in body must align with References [N]."""

    def test_aligned_brief_returns_no_mismatches(self):
        """Well-formed brief → empty list (no mismatches)."""
        mismatches = verify_citation_alignment(ALIGNED_BRIEF)
        assert mismatches == [], f"Expected no mismatches, got: {mismatches}"

    def test_doi_mismatch_detected_when_required(self):
        """Body DOI differs from References DOI → mismatch (require_doi_match=True)."""
        mismatches = verify_citation_alignment(DOI_MISMATCH_BRIEF, require_doi_match=True)
        assert len(mismatches) == 1
        assert mismatches[0].ref_num == 1
        assert "DOI differs" in mismatches[0].reason
        assert "10.1038/nature12373" in mismatches[0].reason
        assert "10.1000/fake1234" in mismatches[0].reason

    def test_doi_mismatch_ignored_when_not_required(self):
        """Without require_doi_match, structural alignment passes (no DOI check)."""
        mismatches = verify_citation_alignment(DOI_MISMATCH_BRIEF)
        # Only structural check runs — body [1] exists in References → no mismatch
        assert mismatches == []

    def test_missing_references_section_returns_single_mismatch(self):
        """Brief without `## References` → one mismatch with ref_num=-1."""
        mismatches = verify_citation_alignment(MISSING_REFS_BRIEF)
        assert len(mismatches) == 1
        assert mismatches[0].ref_num == -1
        assert "missing" in mismatches[0].reason.lower()

    def test_body_cites_missing_reference_detected(self):
        """Body [2] not in References → mismatch with 'no entry' reason."""
        mismatches = verify_citation_alignment(BODY_CITES_MISSING_REF)
        # [1] aligns; [2] is missing from References
        missing = [m for m in mismatches if m.ref_num == 2]
        assert len(missing) == 1
        assert "no entry" in missing[0].reason
        # [1] should NOT be flagged
        ref_one = [m for m in mismatches if m.ref_num == 1]
        assert ref_one == []

    @pytest.mark.parametrize("strict", [False, True])
    def test_disjoint_word_overlap(self, strict):
        """strict_word_overlap=True flags disjoint papers; False does not."""
        mismatches = verify_citation_alignment(
            DISJOINT_WORD_OVERLAP_BRIEF, strict_word_overlap=strict
        )
        if strict:
            # Body says "zirconium isotope"; ref says "mitochondrial DNA" — no overlap
            assert any("word overlap" in m.reason for m in mismatches), (
                f"Expected word-overlap mismatch, got: {mismatches}"
            )
        else:
            # Without strict check, structural alignment passes
            assert mismatches == [], f"Expected no mismatches, got: {mismatches}"


class TestCitationMismatchDataclass:
    """CitationMismatch is frozen + slots + has __str__."""

    def test_frozen_dataclass(self):
        m = CitationMismatch(
            ref_num=1,
            body_context="some context",
            ref_text="some ref",
            reason="test",
        )
        with pytest.raises((AttributeError, Exception)):
            m.ref_num = 2  # type: ignore[misc]

    def test_str_format(self):
        m = CitationMismatch(
            ref_num=3,
            body_context="ctx",
            ref_text="ref text",
            reason="because",
        )
        s = str(m)
        assert "[3]" in s
        assert "MISMATCH" in s
        assert "because" in s


class TestNarrativeCitationAlignment:
    """Contract: build_thematic_narrative output must pass citation alignment."""

    def test_multitheme_brief_citations_align(self):
        """Multi-theme brief with a duplicate paper → body [N] aligns with References [N]."""
        # 4 papers, 2 themes. Paper B appears in both themes (duplicate).
        # Without the renumber fix, body [N] would misalign with References [N].
        papers = [
            {
                "paper_id": "A",
                "doi": "10.1/a",
                "title": "Paper A on isotope geochemistry",
                "year": 2020,
                "authors": [{"name": "A. Author"}],
                "key_finding": "Isotope ratios show fractionation",
                "discipline": "geochemistry",
            },
            {
                "paper_id": "B",
                "doi": "10.1/b",
                "title": "Paper B on method comparison",
                "year": 2021,
                "authors": [{"name": "B. Author"}],
                "key_finding": "Methods produce consistent results",
                "discipline": "methodology",
            },
            {
                "paper_id": "C",
                "doi": "10.1/c",
                "title": "Paper C on field study",
                "year": 2022,
                "authors": [{"name": "C. Author"}],
                "key_finding": "Field samples confirm the model",
                "discipline": "field_study",
            },
            {
                "paper_id": "D",
                "doi": "10.1/d",
                "title": "Paper D on new technique",
                "year": 2023,
                "authors": [{"name": "D. Author"}],
                "key_finding": "New technique improves precision",
                "discipline": "methodology",
            },
        ]
        topic = "isotope fractionation methods"
        narrative, cited = build_chronological_narrative(papers, topic, "survey", correlation=None)
        # Build full brief with References
        brief = narrative + "\n\n## References\n\n" + format_citation_list(cited)
        mismatches = verify_citation_alignment(brief)
        assert mismatches == [], "Citation misalignment detected:\n" + "\n".join(
            str(m) for m in mismatches
        )

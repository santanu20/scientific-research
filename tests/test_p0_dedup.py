"""P0-8 blocked dedup tests — generic-title false merges + preprint merging."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _sources import PaperRecord, dedup_papers


def _p(title, *, doi=None, arxiv=None, year=None, first_author=None):
    authors = [{"name": first_author}] if first_author else []
    return PaperRecord(title=title, doi=doi, arxiv_id=arxiv, year=year, authors=authors)


class TestBlockedDedup:
    def test_generic_titles_different_authors_not_merged(self):
        """The D3 false-merge: identical generic title, different papers."""
        a = _p("Introduction to quantum field theory", year=2019, first_author="Jones")
        b = _p("Introduction to quantum field theory", year=2021, first_author="Smith")
        out = dedup_papers([a, b])
        assert len(out) == 2  # old code merged these (Jaccard=1.0)

    def test_generic_titles_same_year_different_authors_not_merged(self):
        a = _p("A review of machine learning", year=2020, first_author="Alice Chen")
        b = _p("A review of machine learning", year=2020, first_author="Bob Diaz")
        assert len(dedup_papers([a, b])) == 2  # chen vs diaz blocks

    def test_preprint_published_merged_via_block(self):
        """Preprint (arXiv, no DOI) + published version merge: same block + fuzzy title."""
        pre = _p(
            "A comprehensive review of deep learning methods for mineral mapping",
            arxiv="2401.00001",
            year=2024,
            first_author="Smith",
        )
        pub = _p(
            "A comprehensive review of deep learning methods for geological mineral mapping",
            doi="10.999/abc123",
            year=2024,
            first_author="Smith",
        )
        out = dedup_papers([pre, pub])
        assert len(out) == 1
        merged = out[0]
        assert merged.doi == "10.999/abc123" and merged.arxiv_id == "2401.00001"

    def test_doi_exact_still_merges_across_blocks(self):
        """Exact DOI is ground truth — merges even when author/year metadata disagrees."""
        a = _p("Title One", doi="10.1/x", year=2020, first_author="Smith")
        b = _p(
            "Completely Different Title", doi="10.1/X", year=2021, first_author="Jones"
        )
        out = dedup_papers([a, b])
        assert len(out) == 1

    def test_missing_author_uses_strict_threshold(self):
        """Author-less records: stricter threshold; identical title same year merges."""
        a = _p("Rare earth elements in arc magmas", year=2021)
        b = _p("Rare earth elements in arc magmas", year=2021, doi="10.2/r")
        assert len(dedup_papers([a, b])) == 1

    def test_missing_author_different_years_not_merged(self):
        a = _p("Rare earth elements in arc magmas", year=2019)
        b = _p("Rare earth elements in arc magmas", year=2023, doi="10.2/s")
        assert len(dedup_papers([a, b])) == 2

    def test_supplemental_doi_filtered(self):
        a = _p("Paper", doi="10.1/main")
        supp = _p("Supplementary Information", doi="10.1/main.s001")
        out = dedup_papers([a, supp])
        assert len(out) == 1

    def test_arxiv_exact_merge(self):
        a = _p("One title", arxiv="2301.11111", year=2023, first_author="Khan")
        b = _p("Another title", arxiv="2301.11111", year=2023, first_author="Khan")
        assert len(dedup_papers([a, b])) == 1

"""P1-13 PRISMA-S / PROSPERO renderer tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from report import render_prisma_s, render_prospero_draft  # noqa: E402


class TestPrismaS:
    def test_all_16_items_present(self):
        out = render_prisma_s("q", ["crossref", "epmc"], 42, ["2026-08-14T10:00:00Z"])
        for i in range(1, 17):
            assert f"{i}. " in out

    def test_databases_mapped_to_names(self):
        out = render_prisma_s("q", ["crossref", "epmc"], 5, ["d"])
        assert "Crossref (crossref.org)" in out
        assert "Europe PMC" in out

    def test_unknown_source_passed_through(self):
        out = render_prisma_s("q", ["mysterydb"], 5, ["d"])
        assert "mysterydb" in out

    def test_fill_placeholders_never_guessed(self):
        out = render_prisma_s("q", ["crossref"], 5, ["d"])
        assert "[FILL: peer reviewer" in out
        assert "[FILL: paste exact per-database" in out

    def test_record_count_reported(self):
        out = render_prisma_s("q", ["crossref"], 42, ["d"])
        assert "42 records" in out


class TestProspero:
    def test_draft_mentions_counts_and_registration_caveat(self):
        out = render_prospero_draft("q", n_records=42, n_verified=30)
        assert "**42**" in out and "**30**" in out
        assert "retrospective" in out

    def test_fill_fields_for_human_input(self):
        out = render_prospero_draft("q", 10, 8)
        assert out.count("[FILL") >= 5

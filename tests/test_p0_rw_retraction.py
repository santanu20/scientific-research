"""P0-4 Retraction Watch secondary cross-check tests (concern-only, fail-open)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import verify
from _sources import PaperRecord
from verify import (
    check_retraction_watch,
    load_rw_index,
)


@pytest.fixture(autouse=True)
def _reset_rw_cache():
    verify._RW_INDEX = None
    yield
    verify._RW_INDEX = None


def _rw_csv(tmp_path, rows):
    f = tmp_path / "retractions.csv"
    header = "Record ID,Title,OriginalPaperDOI,RetractionDate,Reason\n"
    f.write_text(header + "".join(rows))
    return f


class TestLoadRwIndex:
    def test_default_path_auto_discovery(self, tmp_path, monkeypatch):
        """No arg + no env → resolves _RW_DEFAULT_CSV (the git-clone location)."""
        f = _rw_csv(tmp_path, ["1,P,10.1/d,2024,Misconduct\n"])
        monkeypatch.delenv("SCIENTIFIC_RESEARCH_RW_CSV", raising=False)
        monkeypatch.setattr(verify, "_RW_DEFAULT_CSV", f)
        idx = load_rw_index(None)
        assert idx == {"10.1/d"}

    def test_env_overrides_default(self, tmp_path, monkeypatch):
        env_csv = tmp_path / "env.csv"
        env_csv.write_text("Record ID,Title,OriginalPaperDOI\n1,E,10.1/env\n")
        default_csv = tmp_path / "default.csv"
        default_csv.write_text("Record ID,Title,OriginalPaperDOI\n1,D,10.1/default\n")
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_RW_CSV", str(env_csv))
        monkeypatch.setattr(verify, "_RW_DEFAULT_CSV", default_csv)
        idx = load_rw_index(None)
        assert idx == {"10.1/env"}  # env wins over default

    def test_loads_dois_from_any_doi_column(self, tmp_path):
        f = _rw_csv(
            tmp_path,
            [
                "1,Some paper,10.1/a,2024-01-01,Misconduct\n",
                "2,Other,https://doi.org/10.1/b,2024-02-01,Error\n",
            ],
        )
        idx = load_rw_index(f)
        assert idx == {"10.1/a", "10.1/b"}  # https prefix stripped

    def test_missing_file_fail_open(self, tmp_path):
        idx = load_rw_index(tmp_path / "nonexistent.csv")
        assert idx == set()

    def test_no_doi_column_fail_open(self, tmp_path):
        f = tmp_path / "retractions.csv"
        f.write_text("Record ID,Title\n1,No doi here\n")
        assert load_rw_index(f) == set()

    def test_cached_after_first_load(self, tmp_path):
        f = _rw_csv(tmp_path, ["1,P,10.1/c,2024,Plagiarism\n"])
        first = load_rw_index(f)
        second = load_rw_index(None)  # cached — no env, no path
        assert second is first


class TestCheckRetractionWatch:
    def test_match_returns_concern(self):
        rec = PaperRecord(doi="10.1/a", title="Some paper")
        hit = check_retraction_watch(rec, {"10.1/a"})
        assert hit is not None
        status, source = hit
        assert status == "concern"  # never auto-'retracted' (secondary signal)
        assert "retraction-watch" in source

    def test_no_match_returns_none(self):
        rec = PaperRecord(doi="10.1/zzz", title="Clean paper")
        assert check_retraction_watch(rec, {"10.1/a"}) is None

    def test_doi_url_normalized(self):
        rec = PaperRecord(doi="https://doi.org/10.1/b", title="Prefixed")
        hit = check_retraction_watch(rec, {"10.1/b"})
        assert hit is not None

    def test_empty_index_returns_none(self):
        rec = PaperRecord(doi="10.1/a", title="x")
        assert check_retraction_watch(rec, set()) is None

    def test_paper_without_doi_returns_none(self):
        rec = PaperRecord(title="no doi")
        assert check_retraction_watch(rec, {"10.1/a"}) is None

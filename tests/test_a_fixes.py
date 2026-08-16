"""A1-A12 regression tests (nuclear-audit fix round, 2026-08-16).

Each test guards a finding from .audit/findings.md. Mutation spot-checks
M1/M4/M5 (pass-ledger) must fail these if reintroduced.
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _sources import PaperRecord


# ─── A1: screen.py keyword mode crashed (NameError: _math) ───────────────────
class TestA1ScreenKeywordMode:
    def test_screen_corpus_keyword_mode_runs(self):
        from screen import screen_corpus

        ps = [
            PaperRecord(
                doi="10.1/x", title="magma evolution", abstract="basalt petrology"
            ),
            PaperRecord(
                doi="10.1/y", title="mars volcanism", abstract="martian basalt"
            ),
        ]
        scored = screen_corpus(ps, include=["basalt"], exclude=["mars"])
        assert len(scored) == 2
        # the mars paper must score lower (exclude keyword hit)
        by_doi = {s.paper.doi: s.score for s in scored}
        assert by_doi["10.1/x"] > by_doi["10.1/y"]

    def test_set_keyword_idf_weights_rare_terms_higher(self):
        from screen import _score_paper_keywords, set_keyword_idf

        ps = [
            PaperRecord(doi=f"10.1/{i}", title="magma", abstract="basalt flow")
            for i in range(5)
        ]
        ps.append(
            PaperRecord(doi="10.1/rare", title="magma", abstract="eclogite exhumation")
        )
        set_keyword_idf(ps, include=["basalt", "eclogite"], exclude=["mars"])
        idf = _score_paper_keywords._batch_idf
        assert idf["eclogite"] > idf["basalt"]  # df=1 beats df=5
        # M5 mutation guard: score must be nonzero and signed correctly
        s_inc, mi, _ = _score_paper_keywords(
            PaperRecord(doi="10.1/z", title="eclogite", abstract=""),
            ["eclogite"],
            ["basalt"],
        )
        assert s_inc > 0 and mi == ["eclogite"]


# ─── A2: search_cache_put NameError (dead KB manifest write path) ────────────
class TestA2SearchCachePut:
    def test_put_get_roundtrip_and_filters_hash_keying(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_KB", str(tmp_path / "kb"))
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_CACHE", str(tmp_path / "cache"))
        import _search_cache as sc

        sc = importlib.reload(sc)
        sc.search_cache_put("query a", "openalex", 10, ["id1", "id2"])
        ids, age = sc.search_cache_get("query a", "openalex", 10)
        assert ids == ["id1", "id2"] and age == 0

        # distinct filters_hash → distinct manifest entry (never collide)
        sc.search_cache_put("query a", "openalex", 10, ["filtered"], filters_hash="f1")
        ids_f, _ = sc.search_cache_get("query a", "openalex", 10, filters_hash="f1")
        ids_plain, _ = sc.search_cache_get("query a", "openalex", 10)
        assert ids_f == ["filtered"] and ids_plain == ["id1", "id2"]

    def test_put_writes_manifest_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_KB", str(tmp_path / "kb"))
        monkeypatch.setenv("SCIENTIFIC_RESEARCH_CACHE", str(tmp_path / "cache"))
        import _search_cache as sc

        sc = importlib.reload(sc)
        sc.search_cache_put("q", "epmc", 5, ["pid"])
        manifest = tmp_path / "kb" / "search_manifest.json"
        assert manifest.exists()  # the dead path must actually persist


# ─── A4/M4: _ref_str format contract (renumbering blind zone) ────────────────
class TestA4RefStrFormat:
    def test_ref_str_exact_format(self):
        from _narrative import _ref_str

        assert _ref_str([]) == ""
        assert _ref_str([1]) == "1"
        assert _ref_str([1, 2]) == "1, 2"
        assert _ref_str([1, 2, 3, 4, 5]) == "1-5"  # exact, no +1 drift


# ─── A4/M1: DL heterogeneity c-constant golden (blind zone) ──────────────────
class TestA4DLHeterogeneityGolden:
    def test_q_and_i2_hand_computed(self):
        """Independent numpy reference; guards _stats.py c-constant skew."""
        import numpy as np
        from _stats import heterogeneity

        effects = [1.0, 2.0, 3.5]
        variances = [0.25, 0.5, 0.2]

        w = np.array([1.0 / v for v in variances])
        eff = np.array(effects)
        mu = (w * eff).sum() / w.sum()
        q_ref = float((w * (eff - mu) ** 2).sum())
        i2_ref = max(0.0, (q_ref - (len(effects) - 1)) / q_ref * 100.0)
        c_ref = w.sum() - (w**2).sum() / w.sum()
        tau2_ref = max((q_ref - (len(effects) - 1)) / c_ref, 0.0)

        h = heterogeneity(effects, variances)
        assert abs(h.q - q_ref) < 1e-9
        assert abs(h.i_squared - i2_ref) < 1e-9
        assert abs(h.tau_squared - tau2_ref) < 1e-9
        assert h.df == 2 and h.k == 3

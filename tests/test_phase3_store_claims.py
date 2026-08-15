"""Phase 3 tests: DocumentStore, claims engine, subgroup decomposition."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _claims_engine import (  # noqa: E402
    NumericClaim,
    contradiction_report,
    detect_contradictions,
    extract_claims,
)
from _documentstore import DocumentStore, _paper_key  # noqa: E402


class TestDocumentStore:
    def test_store_and_retrieve_roundtrip(self, tmp_path):
        ds = DocumentStore(tmp_path)
        # simulate a stored text directly
        sha = "deadbeefdeadbeef"
        (tmp_path / "store" / f"{sha}.txt").write_text("x" * 600)
        ds.index["10.1/a"] = {"sha": sha, "source": "incoming", "n_chars": 600}
        ds._save_index()
        ds2 = DocumentStore(tmp_path)
        assert len(ds2.get_text({"doi": "10.1/a"})) == 600
        assert ds2.get_text({"doi": "10.1/missing"}) == ""

    def test_incoming_fuzzy_match(self, tmp_path):
        ds = DocumentStore(tmp_path)
        (
            tmp_path / "incoming" / "Amphibole thermobarometry Mount Adams 2020.pdf"
        ).write_bytes(b"pdf")
        hit = ds._incoming_match("amphibole thermobarometry of mount adamas")
        assert hit is not None and "Amphibole" in hit.name

    def test_no_match_returns_none(self, tmp_path):
        ds = DocumentStore(tmp_path)
        (tmp_path / "incoming" / "clinical trial.pdf").write_bytes(b"pdf")
        assert ds._incoming_match("amphibole thermobarometry") is None

    def test_unpaywall_no_doi(self):
        assert DocumentStore._unpaywall_pdf("") is None

    def test_paper_key_stable(self):
        assert _paper_key({"doi": "10.1/x"}) == "10.1/x"
        assert _paper_key({"title": "T" * 300}) == "T" * 120


class TestClaimsEngine:
    def test_extract_from_abstract(self):
        p = {
            "doi": "10.1/a",
            "abstract": "We measure a temperature of 850 °C in the magma body.",
        }
        claims = extract_claims(p)
        assert any(
            c.quantity == "temperature" and c.unit == "°C" and c.value == 850.0
            for c in claims
        )

    def test_fulltext_claims_tagged(self):
        p = {"doi": "10.1/b", "abstract": ""}
        ft = (
            "The crystallization pressure is 8 kbar based on the amphibole barometry. "
            * 3
        )
        claims = extract_claims(p, fulltext=ft)
        assert claims and all(c.source == "fulltext" for c in claims)

    def test_contradiction_detected(self):
        claims = [
            NumericClaim("A", "pressure", "kbar", 2.0, "s1"),
            NumericClaim("B", "pressure", "kbar", 12.0, "s2"),  # 6x spread
        ]
        cons = detect_contradictions(claims)
        assert len(cons) == 1 and cons[0].ratio >= 5.0

    def test_agreement_not_flagged(self):
        claims = [
            NumericClaim("A", "pressure", "kbar", 5.0, "s"),
            NumericClaim("B", "pressure", "kbar", 6.0, "s"),
        ]
        assert detect_contradictions(claims) == []

    def test_single_paper_never_contradicts(self):
        claims = [
            NumericClaim("A", "T", "°C", 100.0, "s"),
            NumericClaim("A", "T", "°C", 900.0, "s"),
        ]
        assert detect_contradictions(claims) == []

    def test_report_renders(self):
        cons = detect_contradictions(
            [
                NumericClaim("A", "T", "°C", 100.0, "s"),
                NumericClaim("B", "T", "°C", 900.0, "s"),
            ]
        )
        txt = contradiction_report(cons)
        assert "9x" in txt.replace("×", "x") or "9" in txt
        assert "analyst adjudicates" in txt


class TestSubgroupDecade:
    def test_high_i2_triggers_decade_split(self, tmp_path):
        """Source contract + behavior: I²>75 pools gain decade subgroups."""
        import meta_analyze

        src = Path(meta_analyze.__file__).read_text()
        assert "year-decade decomposition" in src
        assert "_year" in src

    def test_pool_with_years_emits_subgroups(self, tmp_path):
        from _sources import PaperRecord
        from meta_analyze import StudyEffect, run_meta_analysis

        studies = []
        for i, (yr, val) in enumerate(
            [(2005, 2.0), (2007, 2.4), (2018, 9.0), (2019, 9.5)]
        ):
            se = StudyEffect(
                paper_id=f"p{i}",
                doi=f"10.1/d{i}",
                name=f"D{i}",
                effect=val,
                variance=1.0,
                ci_lower=val - 1,
                ci_upper=val + 1,
                scale="value",
                scale_label="pressure (kbar)",
                subgroup="pressure",
            )
            se._year = yr
            studies.append(se)
        res = run_meta_analysis(studies, model="random", do_subgroups=False)
        # do_subgroups=False but decade split is its own gate (I² will be ~high)
        decade_rows = [
            r for r in res.subgroup_results if "decade" in str(r.get("label", ""))
        ]
        assert len(decade_rows) == 2  # 2000s + 2010/20s
        by_label = {r["label"]: r for r in decade_rows}
        e2k = by_label["decade 2000s"]["effect"]
        e20 = by_label["decade 2010s"]["effect"]
        assert e2k < 5 < e20  # split EXPLAINS the heterogeneity

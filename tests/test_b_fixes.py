"""B1-B5 fix regression tests (2026-08-15 E2E validation findings)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _effect_parser import extract_effect_sizes
from _sources import PaperRecord, dedup_papers


# ── B5a: unit binding + family compatibility ─────────────────────────────
class TestB5aUnitBinding:
    def test_temperature_never_bound_to_gpa(self):
        """The observed failure: 'temperature' label leaked onto a GPa number."""
        text = (
            "We estimate temperature of 90 GPa and 5 kbar pressure, "
            "with melt composition of 2 °C."
        )
        singles = extract_effect_sizes(text)["single_measurements"]
        for s in singles:
            if s["unit"] == "GPa":
                assert s["measurement"] == "pressure", s
            if s["unit"] == "°C":
                assert s["measurement"] == "temperature", s
            if s["measurement"] == "melt composition":
                assert s["unit"] not in ("GPa", "°C"), s

    def test_unlabeled_unitless_numbers_dropped(self):
        text = "The samples were collected at site 3 and 47 others nearby."
        singles = extract_effect_sizes(text)["single_measurements"]
        assert all(s.get("unit") or s.get("measurement") for s in singles)

    def test_unit_family_default_label(self):
        text = "Crystallization occurred at 750 °C based on the models."
        singles = extract_effect_sizes(text)["single_measurements"]
        t = [s for s in singles if s["unit"] == "°C"]
        assert t and all(s["measurement"] == "temperature" for s in t)

    def test_normal_bindings_survive(self):
        text = "Amphibole crystallized at 850 °C and 6 kbar; age of 12 Ma."
        singles = extract_effect_sizes(text)["single_measurements"]
        units = {s["unit"] for s in singles}
        assert "°C" in units
        assert ("kbar" in units) or ("bar" in units)  # kbar via 'bar' substring
        ages = [s for s in singles if s["unit"] in ("Ma", "Ga")]
        assert ages and all(s["measurement"] == "age" for s in ages)


# ── B5b: unit-partition gate ───────────────────────────────────────────────
def _ext(pid, val, unit, meas):
    return {
        "paper_id": pid,
        "effect_sizes": {
            "single_measurements": [{"value": val, "unit": unit, "measurement": meas}]
        },
    }


class TestB5bUnitGate:
    def test_never_pools_across_units(self):
        from meta_analyze import collect_single_measurements

        papers = [PaperRecord(doi=f"10.1/g{i}", title=f"P{i}") for i in range(4)]
        {p.primary_id: p for p in papers}
        exts = [
            _ext(p.primary_id, 700 + i * 10, "°C", "temperature")
            for i, p in enumerate(papers[:3])
        ] + [_ext(papers[3].primary_id, 5, "kbar", "pressure")]
        effects = collect_single_measurements(exts, papers)
        # only the temperature group (k=3) pools; the kbar singleton is gated out
        labels = {e.scale_label for e in effects}
        assert labels == {"temperature (°C)"}, labels
        assert all(e.effect > 600 for e in effects)

    def test_singleton_groups_dropped(self):
        from meta_analyze import collect_single_measurements

        papers = [
            PaperRecord(doi="10.1/a", title="A"),
            PaperRecord(doi="10.1/b", title="B"),
        ]
        exts = [
            _ext(papers[0].primary_id, 700, "°C", "temperature"),
            _ext(papers[1].primary_id, 5, "kbar", "pressure"),
        ]
        assert collect_single_measurements(exts, papers) == []

    def test_unbound_rows_never_reach_pooling(self):
        from meta_analyze import collect_single_measurements

        papers = [
            PaperRecord(doi="10.1/a", title="A"),
            PaperRecord(doi="10.1/b", title="B"),
        ]
        exts = [
            _ext(papers[0].primary_id, 3, "", ""),
            _ext(papers[1].primary_id, 7, "", ""),
        ]
        assert collect_single_measurements(exts, papers) == []


# ── B1: co-occurrence threshold ────────────────────────────────────────────
class TestB1Cooccurrence:
    def test_multiterm_query_keeps_partial_matches(self):
        import discover

        # the exact observed failure: 6-term query → 1 paper kept of 128
        papers = [
            PaperRecord(
                title="Amphibole geothermobarometry of arc magmas",
                abstract="We constrain temperatures in subduction zones.",
            ),
            PaperRecord(title="Clinical trial outcomes", abstract="Drug study."),
        ]
        out = discover._enforce_cooccurrence(
            papers, "amphibole thermobarometry arc magma temperatures pressures"
        )
        assert len(out) == 1  # old code: 0 (missing exact 'thermobarometry')

    def test_off_topic_still_rejected(self):
        import discover

        papers = [
            PaperRecord(title="Drug trial results", abstract="Placebo controlled.")
        ]
        out = discover._enforce_cooccurrence(
            papers, "amphibole thermobarometry arc magma temperatures pressures"
        )
        assert out == []

    def test_single_term_passthrough(self):
        import discover

        papers = [PaperRecord(title="Anything", abstract="x")]
        assert discover._enforce_cooccurrence(papers, "thermobarometry") == papers


# ── B3: cross-year preprint repost dedup ──────────────────────────────────
class TestB3CrossYearRepost:
    def test_repost_different_year_merged(self):
        a = PaperRecord(
            title="Unique Amphibole Bearing Mantle Column Beneath the Leningrad Region",
            doi="10.20944/preprints202112.0444.v1",
            year=2021,
            authors=[{"name": "Ashchepkov IV"}],
        )
        b = PaperRecord(
            title="Unique Amphibole Bearing Mantle Column Beneath the Leningrad Region",
            doi="10.21203/rs.3.rs-1374345/v1",
            year=2022,
            authors=[{"name": "Ashchepkov IV"}],
        )
        out = dedup_papers([a, b])
        assert len(out) == 1
        dois = {out[0].doi}
        assert dois & {
            "10.20944/preprints202112.0444.v1",
            "10.21203/rs.3.rs-1374345/v1",
        }

    def test_same_surname_different_papers_not_merged(self):
        """Near-exact fallback must stay SURNAME-scoped and title-strict."""
        a = PaperRecord(
            title="Amphibole thermobarometry of the Aleutian arc",
            year=2019,
            authors=[{"name": "Smith J"}],
        )
        b = PaperRecord(
            title="Amphibole thermobarometry of the Cascades arc",
            year=2022,
            authors=[{"name": "Smith K"}],
        )
        assert len(dedup_papers([a, b])) == 2


# ── B4: semantic per_page clamp ────────────────────────────────────────────
class TestB4SemanticClamp:
    def test_semantic_per_page_never_exceeds_50(self):
        import re as _re

        src = (Path(__file__).parent.parent / "scripts" / "_sources.py").read_text()
        fn_block = _re.search(
            r"def openalex_semantic_search.*?(?=\ndef )", src, _re.DOTALL
        ).group(0)
        assert "per_page=min(max_results, 50)" in fn_block
        # keyword endpoint (cursor-paginated) legitimately allows 200; only
        # the .similar() semantic endpoint hard-rejects per_page > 50
        assert "200" not in fn_block

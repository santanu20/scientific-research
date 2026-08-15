"""Gap-fix tests: unit canonicalization (G1) + stopping advisory (G3)."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _units import BASE_MEASUREMENT, canonical_unit, to_canonical
from screen import stopping_advisory


class TestUnitConversion:
    def test_pressure_family_merges(self):
        assert to_canonical(1.0, "GPa") == ("kbar", 10.0, None)
        assert to_canonical(100.0, "MPa")[1] == pytest.approx(1.0)  # 100 MPa = 1 kbar
        assert to_canonical(1000.0, "bar")[1] == pytest.approx(1.0)
        assert to_canonical(5.0, "kbar") == ("kbar", 5.0, None)

    def test_age_and_length(self):
        assert to_canonical(1.0, "Ga") == ("Ma", 1000.0, None)
        assert to_canonical(1.0, "ka")[1] == pytest.approx(0.001)
        assert to_canonical(1.0, "m")[1] == pytest.approx(1e-3)
        assert to_canonical(1.0, "nm")[1] == pytest.approx(1e-12)

    def test_temperature_affine(self):
        base, v, u = to_canonical(1000.0, "K", 25.0)
        assert base == "°C"
        assert v == pytest.approx(726.85)
        assert u == pytest.approx(25.0)  # affine offset does not scale unc

    def test_fraction(self):
        assert to_canonical(5.0, "‰")[1] == pytest.approx(0.5)
        assert to_canonical(5.0, "wt%") == ("%", 5.0, None)

    def test_uncertainty_scales_multiplicatively(self):
        _, v, u = to_canonical(2.0, "GPa", 0.1)
        assert v == pytest.approx(20.0) and u == pytest.approx(1.0)

    def test_exotic_units_untouched(self):
        assert to_canonical(5.0, "‰ VSMOW") is None
        assert to_canonical(3.0, "fold") is None
        assert to_canonical(1.0, "ratio") is None

    def test_base_measurement_map(self):
        assert BASE_MEASUREMENT[canonical_unit("ka")] == "age"
        assert BASE_MEASUREMENT[canonical_unit("GPa")] == "pressure"


class TestCollectUnitMerge:
    def test_mixed_pressure_units_single_pool(self):
        from _sources import PaperRecord
        from meta_analyze import collect_single_measurements

        papers = [PaperRecord(doi=f"10.1/p{i}", title=f"P{i}") for i in range(3)]
        rows = [
            {"value": 5.0, "unit": "kbar", "measurement": "pressure"},
            {"value": 1.0, "unit": "GPa", "measurement": "pressure"},  # → 10 kbar
            {"value": 100.0, "unit": "MPa", "measurement": "pressure"},  # → 1 kbar
        ]
        exts = [
            {"paper_id": p.primary_id, "effect_sizes": {"single_measurements": [r]}}
            for p, r in zip(papers, rows, strict=True)
        ]
        effects = collect_single_measurements(exts, papers)
        assert len(effects) == 3
        assert {e.scale_label for e in effects} == {"pressure (kbar)"}
        vals = sorted(e.effect for e in effects)
        assert vals == [1.0, 5.0, 10.0]

    def test_ka_relabelled_age_and_converted(self):
        from _sources import PaperRecord
        from meta_analyze import collect_single_measurements

        papers = [PaperRecord(doi=f"10.1/a{i}", title=f"A{i}") for i in range(2)]
        rows = [
            {"value": 1000.0, "unit": "ka", "measurement": ""},  # → 1 Ma, age
            {"value": 2.0, "unit": "Ma", "measurement": "age"},
        ]
        exts = [
            {"paper_id": p.primary_id, "effect_sizes": {"single_measurements": [r]}}
            for p, r in zip(papers, rows, strict=True)
        ]
        effects = collect_single_measurements(exts, papers)
        assert len(effects) == 2
        assert {e.scale_label for e in effects} == {"age (Ma)"}
        assert sorted(e.effect for e in effects) == [1.0, 2.0]

    def test_kelvin_joins_celsius_pool(self):
        from _sources import PaperRecord
        from meta_analyze import collect_single_measurements

        papers = [PaperRecord(doi=f"10.1/t{i}", title=f"T{i}") for i in range(2)]
        rows = [
            {"value": 973.15, "unit": "K", "measurement": "temperature"},  # 700 °C
            {"value": 800.0, "unit": "°C", "measurement": "temperature"},
        ]
        exts = [
            {"paper_id": p.primary_id, "effect_sizes": {"single_measurements": [r]}}
            for p, r in zip(papers, rows, strict=True)
        ]
        effects = collect_single_measurements(exts, papers)
        assert {e.scale_label for e in effects} == {"temperature (°C)"}
        assert any(abs(e.effect - 700.0) < 1e-9 for e in effects)


class TestStoppingAdvisory:
    def test_rich_stream_high_expectation(self):
        adv = stopping_advisory(["include"] * 10 + ["exclude"] * 10, n_total=200)
        assert adv["p_include_posterior"] == pytest.approx(11 / 22)
        assert not adv["cal_stop"] and not adv["erv_stop"]

    def test_barren_stream_triggers_both(self):
        # beta(1,1) smoothing keeps p̂ ≥ 1/(n+2): ERV < 1 needs n ≥ 99
        adv = stopping_advisory(["exclude"] * 100, n_total=100)
        assert adv["cal_stop"] and adv["erv_stop"]

    def test_expected_remaining_math(self):
        adv = stopping_advisory(["exclude"] * 30 + ["include"] * 3, n_total=53)
        p = 4 / 35  # beta(1,1): (3+1)/(33+2)
        # function rounds to 2dp for report readability
        assert math.isclose(
            adv["expected_relevant_remaining"], round((53 - 33) * p, 2), rel_tol=1e-9
        )
        assert math.isclose(adv["erv_next_100"], round(100 * p, 2), rel_tol=1e-9)

    def test_maybe_counts_as_not_relevant(self):
        a = stopping_advisory(["maybe"] * 10, 50)
        b = stopping_advisory(["exclude"] * 10, 50)
        assert a["p_include_posterior"] == b["p_include_posterior"]

    def test_advisory_flag_present(self):
        adv = stopping_advisory(["include"], 10)
        assert "advisory" in adv["note"] and "never automatic" in adv["note"]

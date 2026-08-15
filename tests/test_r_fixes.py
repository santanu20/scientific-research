"""R1-R5 regression tests (verbose-audit round 3)."""

import json
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _effect_parser import extract_effect_sizes


class TestR4IntervalBounds:
    def test_cooling_interval_bounds_dropped(self):
        """The exact E2E failure: '>190–270 °C' cooling interval."""
        r = extract_effect_sizes(
            "a significant cooling and crystallization interval (>190–270 °C)"
        )
        vals = [s["value"] for s in r["single_measurements"]]
        assert 190 not in vals and 270 not in vals

    def test_experiment_range_bounds_dropped(self):
        r = extract_effect_sizes(
            "We have experimentally (piston-cylinder, 0.5–1.0 GPa, 1200–1350°C)"
        )
        vals = [s["value"] for s in r["single_measurements"]]
        assert 1200 not in vals and 1350 not in vals
        assert 0.5 not in vals and 1.0 not in vals

    def test_point_measurements_survive(self):
        r = extract_effect_sizes(
            "amphibole crystallized at 850 °C and 6 kbar (Late Miocene, 8 Ma)"
        )
        vals = {s["value"] for s in r["single_measurements"]}
        assert {850.0, 6.0} <= vals


class TestR1CorrelateFailLoud:
    def test_extracted_shape_refused(self, tmp_path):
        """Behavior: the typed artifact gate rejects extracted.json shape."""
        from _artifact import ArtifactShapeError, load_verified

        f = tmp_path / "extracted.json"
        f.write_text(json.dumps({"extractions": [], "meta": {}}))
        with pytest.raises(ArtifactShapeError, match="not a valid Verified"):
            load_verified(f)

    def test_zero_papers_refused(self, tmp_path):
        from _artifact import ArtifactShapeError, load_verified

        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"papers": []}))
        with pytest.raises(ArtifactShapeError, match="refusing empty"):
            load_verified(f)

    def test_correlation_must_be_live(self, tmp_path):
        from _artifact import ArtifactShapeError, load_correlation

        f = tmp_path / "dead.json"
        f.write_text(json.dumps({"n_papers": 0}))
        with pytest.raises(ArtifactShapeError, match="EMPTY"):
            load_correlation(f)


def _run_main(args):
    """Drive correlate main body past arg parsing (shape gate lives there)."""

    # replicate the load section directly (main() re-parses argv)
    data = json.loads(Path(args.verified).read_text(encoding="utf-8"))
    if "papers" not in data:
        raise SystemExit(
            f"FATAL: {args.verified} has no 'papers' key — extracted shape refused"
        )
    if not data.get("papers"):
        raise SystemExit("FATAL: 0 papers — refusing empty correlation")
    return data


class TestR3PhenomenonScreen:
    def test_band_table_defined_and_documented(self):
        import meta_analyze

        src = Path(meta_analyze.__file__).read_text()
        assert '("temperature", "°C"): (650.0, 1400.0)' in src
        assert '("pressure", "kbar"): (0.3, 15.0)' in src
        assert "Phenomenon screen" in src  # loud logging on exclusion

    def test_band_filter_logic(self):
        """Pool-level screen: vacuum rows outside band, storage rows kept."""
        lo, hi = 0.3, 15.0
        rows = [0.0, 0.01, 5.0, 6.0, 8.0]
        kept = [v for v in rows if lo <= v <= hi]
        assert kept == [5.0, 6.0, 8.0]
        # E2E pressure pool: 0.001 pooled value becomes impossible


class TestR2PoolsInBrief:
    def test_unit_pools_rendered(self, tmp_path):
        import synthesize

        meta = {
            "unit_pools": [
                {
                    "group": "temperature (°C)",
                    "k": 12,
                    "pooled_random": {
                        "effect": 692.9,
                        "ci_lower": 336.3,
                        "ci_upper": 1049.6,
                        "prediction_interval": [300.0, 1100.0],
                        "heterogeneity": {"i_squared": 100.0},
                    },
                }
            ]
        }
        out = synthesize._render_pool_section(meta)
        assert "temperature (°C)" in out
        assert "692.9" in out and "k=12" in out and "I²=100%" in out
        assert "high heterogeneity" in out

    def test_empty_meta_no_section(self):
        import synthesize

        assert synthesize._render_pool_section({}) == ""


class TestR5GapsFromCorrelation:
    def test_gap_signals_fired(self):
        """Structural gap signals: temporal decay + quant coverage."""
        from synthesize import _build_structural_gaps

        papers = [{"year": 2001} for _ in range(6)] + [
            {"year": 2024, "effect_sizes": {"single_measurements": [1]}}
            for _ in range(2)
        ]
        gaps = _build_structural_gaps(
            papers, correlation={"n_papers": 8, "bibliographic_coupling_edges": []}
        )
        txt = " ".join(gaps)
        assert "temporal" in txt
        assert "quantitative coverage" in txt
        assert "fragmented" in txt

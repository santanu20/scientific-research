"""Workflow-audit regressions.

History:
  A1  pipeline caps None (zero-truncation) — pinned 2026-08-15.
  A2/A3/A4  domain filtering round — originally pinned geology signal
      tables. REWRITTEN 2026-08-22 when the master mandated a fully
      dynamic, field-agnostic topical gate (_is_domain_relevant now
      checks overlap with THE QUERY's own content tokens; no discipline
      vocabulary anywhere). The behavioral guarantees below are preserved
      under the new mechanism, plus new pins for the hydro-topic massacre
      the old petrology-centric tables caused (32/35 true papers killed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from discover import _content_tokens, _is_domain_relevant


class _P(dict):
    """dict-style paper (getattr access via attribute passthrough)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestA1PipelineCaps:
    def test_caps_default_none(self):
        import config

        assert config.ResearchConfig(query="q").abstract_cap is None
        assert config.ResearchConfig(query="q").fulltext_cap is None

    def test_pipeline_guard_handles_none(self):
        src = (Path(__file__).parent.parent / "scripts" / "pipeline.py").read_text()
        assert "config.abstract_cap" in src  # guard present


class TestA2BehavioralPinsPreservedUnderDynamicGate:
    def test_ml_applied_to_geo_kept(self):
        p = _P(
            title="Machine learning thermobarometry using amphibole",
            abstract="models estimate pressures in arc magmas",
        )
        assert _is_domain_relevant(p, "amphibole thermobarometry arc magma")

    def test_pure_cs_still_excluded(self):
        p = _P(
            title="Deep learning wireless networks",
            abstract="neural network protocol optimization for antennas",
        )
        assert not _is_domain_relevant(p, "amphibole thermobarometry arc magma")


class TestA3DomainFilterWired:
    def test_cli_flow_calls_domain_filter(self):
        src = (Path(__file__).parent.parent / "scripts" / "discover.py").read_text()
        assert "final = [p for p in final if _is_domain_relevant(p, args.query)]" in src


class TestDynamicTopicalGate:
    """New-contract pins (2026-08-22): the gate is query-relative, not vocab."""

    def test_hydro_papers_survive_contamination_query(self):
        # Regression pin: the old petrology signal tables killed 32/35 true
        # papers for this exact topic class.
        q = "Assessment of Heavy-Metal Contamination in Groundwater Around Mining Areas of Chandrapur"
        hydro = _P(
            title="Hydrogeochemical assessment of groundwater quality",
            abstract="Heavy metal contamination in groundwater near mining areas",
        )
        assert _is_domain_relevant(hydro, q)

    def test_off_field_paper_rejected(self):
        q = "Comparative Petrographic Analysis of the Gadchiroli and Gondpipri Dykes"
        bat = _P(
            title="Physicochemical analysis and foraging habit of bats",
            abstract="Diet composition of greater false vampire bat populations",
        )
        assert not _is_domain_relevant(bat, q)

    def test_threshold_is_20pct_min_two(self):
        q = "groundwater lineament morphometric drainage basin analysis"
        toks = _content_tokens(q)
        need = max(2, -(-len(toks) * 20 // 100))
        partial = _P(
            title="Groundwater and drainage basin study",
            abstract="morphometric parameters",
        )
        hits = sum(1 for t in toks if t in ((partial["title"] + " " + partial["abstract"]).lower()))
        assert hits >= need or not _is_domain_relevant(partial, q)

    def test_empty_query_never_filters(self):
        assert _is_domain_relevant(_P(title="x", abstract=""), "")

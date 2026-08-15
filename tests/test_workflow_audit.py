"""Workflow-audit bug regressions (line-by-line audit round, 2026-08-15).

Bugs found and fixed:
  A1  pipeline.py abstract_cap=6000/fulltext_cap=10000 truncated content in
      the orchestrator path while CLI path was lossless (inconsistency)
  A2  _GEO_EXCLUSION hard-excluded ML/DL/NN terms → rejected legitimate
      ML-applied-to-geoscience papers
  A3  _is_domain_relevant never called from CLI flow (dead code path;
      only IDF co-occurrence caught off-domain junk)
  A4  _GEO_PAPER_SIGNALS plural-blind + missing core terms (magmas,
      amphibole, thermobarometry) — the E2E domain itself matched 0 signals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from discover import _GEO_PAPER_SIGNALS, _is_domain_relevant


class TestA1PipelineCaps:
    def test_caps_default_none(self):
        import config

        assert config.ResearchConfig(query="q").abstract_cap is None
        assert config.ResearchConfig(query="q").fulltext_cap is None

    def test_pipeline_guard_handles_none(self):
        src = (Path(__file__).parent.parent / "scripts" / "pipeline.py").read_text()
        assert "config.abstract_cap" in src  # guard present


class TestA2MLSoftExclusion:
    def test_ml_applied_to_geo_kept(self):
        p = {
            "title": "Machine learning thermobarometry using amphibole",
            "abstract": "models estimate pressures in arc magmas",
        }
        assert _is_domain_relevant(p, "amphibole thermobarometry arc magma")

    def test_pure_cs_still_excluded(self):
        p = {
            "title": "Deep learning wireless networks",
            "abstract": "neural network protocol optimization for antennas",
        }
        assert not _is_domain_relevant(p, "amphibole thermobarometry arc magma")

    def test_ml_with_thin_geo_evidence_excluded(self):
        p = {
            "title": "Neural network prediction of crystal hardness",
            "abstract": "neural network model predictions",
        }
        assert not _is_domain_relevant(p, "amphibole thermobarometry arc magma")


class TestA3DomainFilterWired:
    def test_cli_flow_calls_domain_filter(self):
        src = (Path(__file__).parent.parent / "scripts" / "discover.py").read_text()
        # wired call site (not just the definition)
        assert "final = [p for p in final if _is_domain_relevant(p, args.query)]" in src


class TestA4SignalRegex:
    def test_plurals_match(self):
        assert _GEO_PAPER_SIGNALS.search("in arc magmas")
        assert _GEO_PAPER_SIGNALS.search("melts in the crust")

    def test_core_petrology_terms(self):
        assert _GEO_PAPER_SIGNALS.search("amphibole compositions")
        assert _GEO_PAPER_SIGNALS.search("thermobarometry of plutons")
        assert _GEO_PAPER_SIGNALS.search("hornblende phenocrysts")

    def test_signal_count_thresholds_work(self):
        text = "amphibole thermobarometry in arc magmas"
        assert len(_GEO_PAPER_SIGNALS.findall(text)) >= 2

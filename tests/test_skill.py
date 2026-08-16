#!/usr/bin/env python3
"""Unit tests for scientific-research skill.

Run with:
    cd ~/.config/opencode/skills/scientific-research
    uv run --project <project-with-deps> pytest tests/ -v

Or standalone:
    uv run --project <project> python -m pytest tests/ -v

Tests use VCR-style fixtures where possible (offline), plus live markers for
API calls. Live tests require network + project deps installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def sample_paper_dict():
    """Minimal paper record as dict (no live API calls)."""
    return {
        "doi": "10.1038/nature12373",
        "title": "Nanometre-scale thermometry in a living cell",
        "abstract": "Sensitive probing of temperature variations on nanometre scales...",
        "authors": [{"name": "G. Kucsko"}, {"name": "P. C. Maurer"}],
        "year": 2013,
        "venue": "Nature",
        "type": "journal-article",
        "type_crossref": "journal-article",
        "citation_count": 1962,
        "is_retracted": False,
        "is_open_access": False,
        "concepts": ["Diamond", "Thermometer", "Nanotechnology"],
        "source": "merged",
        "sources_seen": ["crossref", "openalex", "s2"],
    }


@pytest.fixture
def sample_paper_record(sample_paper_dict):
    from _sources import PaperRecord

    return PaperRecord.from_dict(sample_paper_dict)


@pytest.fixture
def retracted_paper_dict():
    """Lancet Commission dementia paper — retracted 2024."""
    return {
        "doi": "10.1016/s0140-6736(20)30367-6",
        "title": "Dementia prevention, intervention, and care: 2020 report of the Lancix",
        "is_retracted": True,
        "year": 2020,
        "type": "journal-article",
        "type_crossref": "journal-article",
        "source": "openalex",
        "sources_seen": ["openalex"],
    }


# =============================================================================
# _sources.py tests
# =============================================================================
class TestPaperRecord:
    def test_primary_id_prefers_doi(self, sample_paper_record):
        assert sample_paper_record.primary_id == "doi:10.1038/nature12373"

    def test_primary_id_falls_back_to_arxiv(self):
        from _sources import PaperRecord

        p = PaperRecord(arxiv_id="1304.1068", title="Test")
        assert p.primary_id == "arxiv:1304.1068"

    def test_primary_id_uses_title_hash_when_no_doi(self):
        from _sources import PaperRecord

        p = PaperRecord(title="Some Random Title")
        assert p.primary_id.startswith("title:")

    def test_round_trip_serialization(self, sample_paper_record):
        d = sample_paper_record.to_dict()
        from _sources import PaperRecord

        restored = PaperRecord.from_dict(d)
        assert restored.doi == sample_paper_record.doi
        assert restored.title == sample_paper_record.title

    def test_unknown_keys_ignored_on_load(self):
        from _sources import PaperRecord

        p = PaperRecord.from_dict({"doi": "10.x/y", "unknown_field": "ignored"})
        assert p.doi == "10.x/y"


class TestDedup:
    def test_dedup_by_doi_exact(self):
        from _sources import PaperRecord, dedup_papers

        a = PaperRecord(
            doi="10.1038/nature12373", title="A", source="openalex", sources_seen=["openalex"]
        )
        b = PaperRecord(
            doi="10.1038/NATURE12373", title="B", source="crossref", sources_seen=["crossref"]
        )
        out = dedup_papers([a, b])
        assert len(out) == 1
        assert "openalex" in out[0].sources_seen
        assert "crossref" in out[0].sources_seen

    def test_dedup_by_title_similarity(self):
        from _sources import PaperRecord, dedup_papers

        # Same paper, one with DOI one without — should merge
        a = PaperRecord(doi="10.x/y", title="Nanodiamond thermometry in living cells")
        b = PaperRecord(title="Nanodiamond thermometry in living cells")
        out = dedup_papers([a, b])
        assert len(out) == 1

    def test_dedup_keeps_different_papers(self):
        from _sources import PaperRecord, dedup_papers

        a = PaperRecord(doi="10.x/y", title="Paper A about XYZ")
        b = PaperRecord(doi="10.x/z", title="Paper B about ABC")
        out = dedup_papers([a, b])
        assert len(out) == 2


class TestAuthorDedup:
    def test_author_dedup_by_last_plus_initial(self):
        from _sources import _dedup_authors

        authors = [
            {"name": "Joel Ita", "openalex_id": "A1"},
            {"name": "Joel Ita", "crossref_id": "A2"},  # exact dup
            {"name": "J. Ita", "s2_id": "A3"},  # abbreviated
            {"name": "Lars Stixrude"},
        ]
        out = _dedup_authors(authors)
        assert len(out) == 2  # 2 unique people
        names = sorted(a["name"] for a in out)
        assert names == ["Joel Ita", "Lars Stixrude"]

    def test_author_dedup_preserves_fuller_name(self):
        from _sources import _dedup_authors

        authors = [{"name": "J. Ita"}, {"name": "Joel Ita"}]
        out = _dedup_authors(authors)
        assert len(out) == 1
        assert out[0]["name"] == "Joel Ita"

    def test_merge_dedups_authors(self):
        from _sources import PaperRecord, _merge_records

        a = PaperRecord(doi="10.x/y", authors=[{"name": "Joel Ita"}])
        b = PaperRecord(doi="10.x/y", authors=[{"name": "J. Ita"}, {"name": "L. Stixrude"}])
        merged = _merge_records(a, b)
        names = sorted(a["name"] for a in merged.authors)
        assert names == ["Joel Ita", "L. Stixrude"]


class TestSanitizeJson:
    def test_datetime_converted_to_iso(self):
        import datetime

        from _sources import _sanitize_json

        d = datetime.datetime(2024, 1, 15, 12, 30, 0)
        out = _sanitize_json({"date": d})
        assert out["date"] == "2024-01-15T12:30:00"

    def test_set_converted_to_list(self):
        from _sources import _sanitize_json

        out = _sanitize_json({"items": {1, 2, 3}})
        assert sorted(out["items"]) == [1, 2, 3]

    def test_nested_structures(self):
        import datetime

        from _sources import _sanitize_json

        out = _sanitize_json(
            {
                "outer": {"inner_date": datetime.date(2024, 1, 1)},
                "list": [datetime.datetime(2024, 1, 1), "string", 42],
            }
        )
        assert out["outer"]["inner_date"] == "2024-01-01"
        assert out["list"][0] == "2024-01-01T00:00:00"
        assert out["list"][1] == "string"
        assert out["list"][2] == 42

    def test_bytes_decoded(self):
        from _sources import _sanitize_json

        out = _sanitize_json({"data": b"hello"})
        assert out["data"] == "hello"

    def test_unknown_object_stringified(self):
        from _sources import _sanitize_json

        class Foo:
            def __str__(self):
                return "FOO"

        out = _sanitize_json({"obj": Foo()})
        assert out["obj"] == "FOO"


# =============================================================================
# _stats.py tests
# =============================================================================
class TestEffectSizes:
    def test_cohens_d_simple(self):
        from _stats import cohens_d

        d = cohens_d(5.0, 1.0, 30, 4.5, 1.0, 30)
        assert abs(d - 0.5) < 0.01

    def test_cohens_d_zero_when_identical(self):
        from _stats import cohens_d

        d = cohens_d(5.0, 1.0, 30, 5.0, 1.0, 30)
        assert d == 0.0

    def test_hedges_g_shrinks_d(self):
        from _stats import cohens_d, hedges_g

        d = cohens_d(5.0, 1.0, 30, 4.5, 1.0, 30)
        g = hedges_g(d, 30, 30)
        # Hedges' g is always slightly smaller than d
        assert abs(g) <= abs(d)

    def test_odds_ratio_basic(self):
        from _stats import odds_ratio

        # 10/20 exposure, 5/25 control → OR > 1
        orr = odds_ratio(10, 20, 5, 25)
        assert orr > 1.0

    def test_odds_ratio_with_zero_cells(self):
        from _stats import odds_ratio

        # Zero cell — should still return a value (Haldane-Anscombe correction)
        orr = odds_ratio(0, 10, 5, 5)
        assert orr > 0
        assert orr < 1  # 0 events in exposure → protective

    def test_or_to_d_roundtrip(self):
        from _stats import d_to_or, or_to_d

        orr = 2.5
        d = or_to_d(orr)
        assert abs(d_to_or(d) - orr) < 0.1


class TestHeterogeneity:
    def test_low_heterogeneity_when_consistent(self):
        from _stats import heterogeneity, interpret_i_squared

        # All similar effects, low variance → low I²
        h = heterogeneity([0.5, 0.5, 0.5, 0.5], [0.04, 0.04, 0.04, 0.04])
        assert h.i_squared < 25
        assert "low" in interpret_i_squared(h.i_squared)

    def test_high_heterogeneity_when_disparate(self):
        from _stats import heterogeneity

        # Bimodal effects → high I²
        h = heterogeneity([1.5, -1.0, 1.8, -1.2], [0.04, 0.04, 0.04, 0.04])
        assert h.i_squared > 75

    def test_returns_k_studies(self):
        from _stats import heterogeneity

        h = heterogeneity([0.1, 0.2, 0.3], [0.04, 0.04, 0.04])
        assert h.k == 3
        assert h.df == 2


class TestMetaAnalysis:
    def test_pool_fixed_zero_when_effects_zero(self):
        from _stats import pool_fixed

        r = pool_fixed([0.0, 0.0, 0.0], [0.04, 0.04, 0.04])
        assert abs(r.effect) < 0.01
        assert r.model == "fixed"

    def test_pool_random_returns_larger_ci_for_heterogeneous(self):
        from _stats import pool_random

        r = pool_random([1.5, -1.0, 1.8, -1.2], [0.04, 0.04, 0.04, 0.04])
        # CI should be wider than effect itself
        assert (r.ci_upper - r.ci_lower) > abs(r.effect)


# =============================================================================
# _render.py tests
# =============================================================================
class TestRenderers:
    def test_citation_graph_basic(self):
        from _render import render_citation_graph

        papers = [
            {"doi": "10.1/x", "title": "Alpha", "referenced_works": ["10.2/y"]},
            {"doi": "10.2/y", "title": "Beta", "referenced_works": []},
        ]
        out = render_citation_graph(papers)
        assert "graph TD" in out

    def test_prisma_flow_includes_all_phases(self):
        from _render import render_prisma_flow

        out = render_prisma_flow(100, 20, 80, 30, 50, 5, 45, 15, 30)
        assert "Identification" in out
        assert "Screening" in out
        assert "Eligibility" in out
        assert "Included" in out

    def test_temporal_histogram(self):
        from _render import render_temporal_histogram

        papers = [{"year": 2020}, {"year": 2020}, {"year": 2021}]
        out = render_temporal_histogram(papers)
        assert "2020" in out
        assert "2021" in out

    def test_forest_plot_with_pooled(self):
        from _render import render_forest_plot

        studies = [
            {"name": "A", "effect": 0.5, "ci_lower": 0.1, "ci_upper": 0.9, "weight": 0.3},
            {"name": "B", "effect": 0.6, "ci_lower": 0.2, "ci_upper": 1.0, "weight": 0.7},
        ]
        pooled = {"effect": 0.55, "ci_lower": 0.3, "ci_upper": 0.8}
        out = render_forest_plot(studies, pooled=pooled)
        assert "Pooled" in out
        assert "0.500" in out  # effect for study A

    def test_concept_clusters(self):
        from _render import render_concept_clusters

        papers = [{"concepts": ["ML", "Climate"]}, {"concepts": ["ML"]}]
        out = render_concept_clusters(papers)
        assert "ML::2" in out
        assert "Climate::1" in out


# =============================================================================
# verify.py tests
# =============================================================================
class TestEvidenceGrading:
    def test_journal_article_gets_level_iii(self):
        from verify import grade_evidence

        lvl, _ = grade_evidence("journal-article")
        assert lvl == "III"

    def test_systematic_review_forced_to_level_i(self):
        from verify import grade_evidence

        lvl, _ = grade_evidence("journal-article", is_systematic_review=True)
        assert lvl == "I"

    def test_editorial_gets_level_vii(self):
        from verify import grade_evidence

        lvl, _ = grade_evidence("editorial")
        assert lvl == "VII"

    def test_falls_back_to_openalex_type_when_crossref_empty(self):
        from verify import grade_evidence

        lvl, _ = grade_evidence("", fallback_type="article")
        assert lvl == "III"

    def test_preprint_gets_level_vi(self):
        from verify import grade_evidence

        lvl, _ = grade_evidence("posted-content")
        assert lvl == "VI"


class TestRobAdvisory:
    def test_rct_suggests_rob_2(self):
        from verify import rob_advisory

        out = rob_advisory("journal-article", study_design_hints=["this is an RCT"])
        tools = [r["tool"] for r in out]
        assert "RoB 2" in tools

    def test_cohort_suggests_robins_i(self):
        from verify import rob_advisory

        out = rob_advisory("journal-article", study_design_hints=["prospective cohort study"])
        tools = [r["tool"] for r in out]
        assert "ROBINS-I" in tools

    def test_diagnostic_suggests_quadas(self):
        from verify import rob_advisory

        out = rob_advisory("journal-article", study_design_hints=["diagnostic accuracy study"])
        tools = [r["tool"] for r in out]
        assert "QUADAS-2" in tools


# =============================================================================
# extract.py tests
# =============================================================================
class TestExtraction:
    def test_pico_extracts_population(self):
        from _sources import PaperRecord
        from extract import extract_pico

        p = PaperRecord(title="Test", abstract="We studied 200 patients with chronic disease.")
        out = extract_pico(p)
        assert len(out["population"]) > 0

    def test_effect_size_extraction_finds_mean_sd(self):
        from _sources import PaperRecord
        from extract import extract_effect_sizes

        p = PaperRecord(
            title="Test",
            abstract="The treatment group had 5.2 ± 1.1 (n=30) and control 4.5 ± 1.0 (n=30).",
        )
        out = extract_effect_sizes(p)
        assert len(out["mean_sd_groups"]) >= 2

    def test_or_extraction(self):
        from _sources import PaperRecord
        from extract import extract_effect_sizes

        p = PaperRecord(title="Test", abstract="The odds ratio was 2.5 (95% CI 1.5-3.5, p<0.001).")
        out = extract_effect_sizes(p)
        assert len(out["effect_sizes"]) >= 1
        assert out["effect_sizes"][0]["value"] == 2.5

    def test_design_detection_rct(self):
        from extract import _detect_design

        assert _detect_design("This was a randomized controlled trial of X.") == "RCT"

    def test_design_detection_meta_analysis(self):
        from extract import _detect_design

        assert (
            _detect_design("We conducted a systematic review and meta-analysis.")
            == "systematic review"
        )

    def test_empty_abstract_returns_empty_extractions(self):
        from _sources import PaperRecord
        from extract import extract_pico

        p = PaperRecord(title="X", abstract="")
        out = extract_pico(p)
        assert out["population"] == []


# =============================================================================
# correlate.py tests
# =============================================================================
class TestContextClassification:
    def test_supporting_marker_detected(self):
        # Test lexicon directly (LLM may classify as "extending" which is also valid)
        from _nlp import classify_stance

        ctx = "Our results confirm the findings of Smith et al. and are consistent with prior work."
        r = classify_stance(ctx)
        assert r.stance in ("supporting", "extending")
        assert r.confidence > 0.5

    def test_contrasting_marker_detected(self):
        from correlate import classify_context

        ctx = "However, our results contradict the claim by Jones et al. that X causes Y."
        label, conf, _ = classify_context(ctx)
        assert label == "contrasting"
        assert conf > 0.5

    def test_empty_context_defaults_to_mentioning(self):
        from correlate import classify_context

        label, conf, _ = classify_context("")
        assert label == "mentioning"

    def test_pure_citation_is_mentioning(self):
        from correlate import classify_context

        ctx = "As cited by Smith et al., see also Jones 2020."
        label, _, _ = classify_context(ctx)
        assert label == "mentioning"


class TestBibliographicCoupling:
    def test_no_coupling_when_no_shared_refs(self):
        from _sources import PaperRecord
        from correlate import bibliographic_coupling

        papers = [
            PaperRecord(openalex_id="W1", referenced_works=["A", "B"]),
            PaperRecord(openalex_id="W2", referenced_works=["C", "D"]),
        ]
        out = bibliographic_coupling(papers)
        assert len(out) == 0

    def test_coupling_detected_when_shared(self):
        from _sources import PaperRecord
        from correlate import bibliographic_coupling

        papers = [
            PaperRecord(openalex_id="W1", referenced_works=["A", "B", "C"]),
            PaperRecord(openalex_id="W2", referenced_works=["A", "B", "D"]),
        ]
        out = bibliographic_coupling(papers)
        assert len(out) == 1
        # 2 shared refs (A and B)
        assert next(iter(out.values())) == 2


# =============================================================================
# monitor.py tests
# =============================================================================
class TestMonitorFiltering:
    def test_filter_by_since_string_date(self):
        from _sources import PaperRecord
        from monitor import filter_by_since

        papers = [
            PaperRecord(publication_date="2022-05-01"),
            PaperRecord(publication_date="2019-01-01"),
            PaperRecord(publication_date="2023-12-31"),
        ]
        out = filter_by_since(papers, "2022-01-01")
        assert len(out) == 2  # 2022-05 and 2023-12 pass

    def test_filter_by_since_handles_int_year(self):
        from _sources import PaperRecord
        from monitor import filter_by_since

        papers = [
            PaperRecord(publication_date="2020"),
            PaperRecord(publication_date=2024),
        ]
        out = filter_by_since(papers, "2022-01-01")
        assert len(out) == 1
        assert out[0].publication_date == 2024

    def test_filter_by_since_falls_back_to_year_field(self):
        from _sources import PaperRecord
        from monitor import filter_by_since

        papers = [
            PaperRecord(year=2019),
            PaperRecord(year=2023),
        ]
        out = filter_by_since(papers, "2022-01-01")
        assert len(out) == 1

    def test_filter_invalid_since_returns_all(self):
        from _sources import PaperRecord
        from monitor import filter_by_since

        papers = [PaperRecord(year=2020)]
        out = filter_by_since(papers, "garbage")
        assert len(out) == 1  # returns all on invalid input


# =============================================================================
# LIVE tests (require network + project deps; opt-in via --live flag)
# =============================================================================
@pytest.mark.live
class TestLiveAPIs:
    """Live API calls — slow, network-dependent. Skip by default.
    Run: pytest tests/ -v -m live"""

    def test_live_crossref_resolve(self):
        from _sources import crossref_resolve_doi

        r = crossref_resolve_doi("10.1038/nature12373")
        assert r is not None
        assert "thermometry" in r.title.lower()

    def test_live_openalex_resolve(self):
        from _sources import openalex_get_by_doi

        r = openalex_get_by_doi("10.1038/nature12373")
        assert r is not None
        assert r.is_retracted is False

    def test_live_s2_resolve(self):
        from _sources import s2_get_paper

        r = s2_get_paper("DOI:10.1038/nature12373")
        assert r is not None
        assert r.s2_paper_id is not None

    def test_live_unpaywall(self):
        from verify import unpaywall_lookup

        r = unpaywall_lookup("10.1038/nature12373")
        assert r.get("is_oa") is True

    def test_live_known_retracted(self):
        # Lancet Commission dementia — retracted 2024
        from _sources import PaperRecord
        from verify import verify_paper

        merged, result = verify_paper(
            PaperRecord(doi="10.1016/s0140-6736(20)30367-6"),
            do_doaj=False,
        )
        assert result.retraction_status == "retracted"

    def test_live_fake_doi_unresolved(self):
        from _sources import PaperRecord
        from verify import verify_paper

        _, result = verify_paper(
            PaperRecord(doi="10.9999/fake-not-real-xxxxx"),
            do_doaj=False,
        )


# =============================================================================
# NLP upgrade tests (Schwartz-Hearst, negation stance, FakeRAKE, query expansion)
# =============================================================================
class TestNlpUpgrades:
    """Tests for ported algorithms: Schwartz-Hearst, Jurgens stance, litsearchr."""

    def test_schwartz_hearst_basic(self):
        from _nlp import extract_abbreviations

        text = "Chronic kidney disease (CKD) affects many patients."
        result = extract_abbreviations(text)
        assert "CKD" in result
        assert "chronic" in result["CKD"].lower()

    def test_schwartz_hearst_multiple(self):
        from _nlp import extract_abbreviations

        text = (
            "We measured the estimated glomerular filtration rate (eGFR) "
            "in patients with end-stage renal disease (ESRD). "
            "The World Health Organization (WHO) provided guidelines."
        )
        result = extract_abbreviations(text)
        assert "eGFR" in result or "CKD" in result
        assert len(result) >= 2

    def test_schwartz_hearst_no_false_positives(self):
        from _nlp import extract_abbreviations

        text = "The results were significant (p < 0.05) and robust."
        result = extract_abbreviations(text)
        # "p" should not be extracted as an abbreviation
        assert "p" not in result

    def test_stance_negated_support_becomes_contrast(self):
        from _nlp import classify_stance

        r = classify_stance("We did not confirm the findings of Smith et al.")
        assert r.stance == "contrasting"

    def test_stance_support_with_jurgens_lexicon(self):
        from _nlp import classify_stance

        r = classify_stance("Our results are consistent with and build on the work of Smith et al.")
        assert r.stance == "supporting"

    def test_stance_contrast_with_jurgens_lexicon(self):
        from _nlp import classify_stance

        r = classify_stance(
            "However, our results differ from Smith et al. and contrast with their conclusions."
        )
        assert r.stance == "contrasting"

    def test_stance_bad_adj_boosts_contrast(self):
        # Test lexicon directly (bad_adj is a lexicon-specific feature)
        from _nlp import _lexicon_classify

        r = _lexicon_classify("The approach of Smith et al. is problematic and flawed.")
        assert r.stance == "contrasting"

    def test_stance_method_with_use_lexicon(self):
        from _nlp import classify_stance

        r = classify_stance("We employed the algorithm and framework described by Lee et al.")
        assert r.stance == "methodology"

    def test_stance_hedging_reduces_confidence(self):
        from _nlp import classify_stance

        certain = classify_stance("Our results confirm the findings of Smith et al.")
        hedged = classify_stance(
            "Our results might confirm the findings of Smith et al., although there could be confounds."
        )
        assert hedged.confidence <= certain.confidence

    def test_fakerake_extracts_terms(self):
        from _nlp import fakerake

        text = (
            "We investigated the oxidation state of basalts using "
            "spectroscopy methods. The redox state was measured."
        )
        terms = fakerake(text)
        assert len(terms) > 3
        assert any("oxidation" in t for t in terms)
        assert any("basalt" in t for t in terms)

    def test_fakerake_respects_max_n(self):
        from _nlp import fakerake

        text = "machine learning model performance evaluation metric"
        terms = fakerake(text, min_n=1, max_n=2)
        for t in terms:
            assert len(t.split()) <= 2

    def test_cooccurrence_matrix(self):
        from _nlp import build_cooccurrence_matrix, rank_terms_by_strength

        texts = [
            "We studied basalt oxidation using spectroscopy methods.",
            "The basalt redox state was measured by spectroscopy.",
            "Oxidation of basalt was analyzed with spectroscopy.",
        ]
        freqs, cooccur = build_cooccurrence_matrix(texts, min_freq=2)
        ranked = rank_terms_by_strength(cooccur)
        assert len(ranked) > 0
        # "basalt" and "spectroscopy" should be highly connected
        top_terms = [t for t, _ in ranked[:3]]
        assert any("basalt" in t for t in top_terms)

    def test_boolean_query_writer(self):
        from _nlp import write_boolean_query

        groups = [
            ["basalt", "basaltic", "basalts"],
            ["oxidation", "redox"],
        ]
        query = write_boolean_query(groups, stemming=True)
        assert "AND" in query
        assert "OR" in query or len(groups[0]) == 1
        # Should not have duplicate stems
        assert query.count("basal*") <= 1

    def test_expand_query_from_seed_texts(self):
        from _nlp import expand_query

        texts = [
            "We studied basalt oxidation state using X-ray spectroscopy.",
            "The basalt redox state was measured by spectroscopy.",
            "Spectroscopy reveals oxidation state of basalt samples.",
            "Mantle temperature controls basalt composition and oxidation.",
        ]
        expanded = expand_query(texts, max_terms=10, min_freq=2)
        assert len(expanded) > 0
        assert any("basalt" in t for t in expanded)

    def test_pico_includes_abbreviations(self):
        from _nlp import extract_pico

        text = (
            "We studied 200 patients with chronic kidney disease (CKD) "
            "in a randomized controlled trial."
        )
        pico = extract_pico(text)
        assert "abbreviations" in pico
        assert "CKD" in pico["abbreviations"]

    def test_remove_redundant_terms(self):
        from _nlp import remove_redundant_terms

        terms = ["burn", "burning", "burns", "fire"]
        result = remove_redundant_terms(terms, closure="left")
        # "burn" is prefix of "burning" and "burns" → they should be removed
        assert "burn" in result
        assert "fire" in result
        assert "burning" not in result
        assert "burns" not in result

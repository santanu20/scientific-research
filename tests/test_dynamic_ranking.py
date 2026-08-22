"""Pins for the dynamic, context-derived ranking contract (B1-F1/B2-F3).

Contract under test — NO static discipline tables anywhere:
- Popularity (citations/network/venue) can never pull an off-topic paper
  above an on-topic one, in ANY field.
- Semantic similarity is absolute: a weak match stays weak in a poor pool.
- Query-facet coverage is computed from the query x pool at runtime.
- The pool-relative relevance floor demotes weak-context papers without
  shrinking the corpus.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _ranking import WEIGHTS, query_facet_scores, rank_papers
from _sources import PaperRecord


def _paper(title: str, abstract: str, cites: int, year: int = 2020) -> PaperRecord:
    return PaperRecord(
        title=title,
        abstract=abstract,
        year=year,
        citation_count=cites,
        doi=None,
        authors=[{"name": "A Author"}],
        venue="Journal of Tests",
        referenced_works=[],
    )


def test__weights__context_dominant_and_normalized():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
    context = WEIGHTS["semantic"] + WEIGHTS["facet"]
    popularity = WEIGHTS["citation"] + WEIGHTS["network"] + WEIGHTS["venue"]
    assert context > popularity


def test__any_field__popularity_never_beats_relevance():
    # CRISPR query: a low-cite on-topic paper must outrank a high-cite
    # off-topic geology paper. Under the old unconditional geo multiplier
    # the garnet paper won; under dynamic scoring it must not.
    q = "CRISPR off-target effects detection methods"
    on_topic = _paper(
        "Detection of CRISPR-Cas9 off-target mutations by sequencing",
        "We benchmark off-target detection assays for genome editing.",
        cites=12,
    )
    off_topic_famous = _paper(
        "Garnet thermobarometry in metamorphic terranes",
        "Petrographic analysis of garnet amphibolite assemblages.",
        cites=5000,
    )
    ranked = rank_papers(q, [on_topic, off_topic_famous], max_n=2)
    assert ranked[0][0].title.startswith("Detection of CRISPR")


def test__semantic_is_absolute_not_pool_relative():
    from _ranking import semantic_relevance_scores

    q = "CRISPR off-target effects"
    strong = _paper("Off-target effects of CRISPR-Cas9 editing", "cas9 off-target", 5)
    weak_pool = [
        _paper("Quantum phase transitions", "hamiltonian lattice", 900),
        strong,
    ]
    scores = semantic_relevance_scores(q, weak_pool)
    # The best paper's score reflects its own quality, not the pool:
    # it must NOT be stretched to exactly 1.0 by pool-max normalization.
    assert scores.max() < 1.0


def test__facet_scores__idf_weighted_coverage():
    q = "groundwater lineament morphometric drainage basin analysis"
    both_facets = _paper(
        "Lineament-controlled groundwater potential of a drainage basin",
        "Morphometric and lineament analysis for groundwater exploration.",
        3,
    )
    one_facet = _paper("Groundwater recharge estimation", "Recharge via wells.", 3)
    none = _paper("Deep mantle plume dynamics", "Seismic tomography evidence.", 3)
    s = query_facet_scores(q, [both_facets, one_facet, none])
    assert s[0] > s[1] > s[2]
    assert 0.0 <= s.min() and s.max() <= 1.0


def test__relevance_floor__weak_paper_demoted_but_corpus_kept():
    q = "hydrogeochemical characterization of basaltic aquifers"
    on_topic = _paper(
        "Hydrogeochemistry of basaltic hard-rock aquifers",
        "Major-ion hydrogeochemical signatures in basalt terrain groundwater.",
        8,
    )
    filler = _paper("Social media sentiment analysis", "Twitter dataset study.", 4000)
    ranked = rank_papers(q, [filler, on_topic], max_n=2)
    assert len(ranked) == 2  # corpus preserved
    assert ranked[0][0].title.startswith("Hydrogeochemistry")


def test__rank_works_identically_for_non_geology_fields():
    # Medicine field: ordering must be purely relevance-driven.
    q = "randomized controlled trial of statin therapy outcomes"
    on_topic = _paper(
        "Statin therapy cardiovascular outcomes: randomized trial",
        "Double-blind placebo-controlled statin treatment cohort.",
        40,
    )
    off = _paper("Basalt geochemistry of Deccan traps", "Trace element geochem.", 3000)
    ranked = rank_papers(q, [off, on_topic], max_n=2)
    assert ranked[0][0].title.startswith("Statin therapy")


def test__polysemy_stray_excluded_when_enough_strong():
    # Fish-morphometry intruder under a drainage-morphometry query:
    # shares tokens but sits far below the on-topic semantic median.
    q = "Morphometric Analysis of a Selected Drainage Basin in Gadchiroli District"
    strong_papers = [
        _paper(
            "Morphometric analysis of drainage basin using GIS",
            "Watershed morphometric parameters: drainage density, stream order, bifurcation ratio of the river basin.",
            6,
        ),
        _paper(
            "Quantitative geomorphology of a watershed",
            "Drainage basin morphometry reveals watershed characteristics and flash flood potential.",
            9,
        ),
        _paper(
            "GIS-based morphometric characterization of sub-basin",
            "Linear and aerial morphometric parameters derived from digital elevation model of the drainage system.",
            4,
        ),
    ]
    stray = _paper(
        "Diversity and conservation of catfish from a sub basin",
        "Morphometric characters and meristic counts of catfish species recorded in the river.",
        5,
    )
    ranked = rank_papers(q, [*strong_papers, stray], max_n=25)
    titles = [p.title for p, _, _ in ranked]
    assert all("catfish" not in ti.lower() for ti in titles)
    assert len(ranked) == 3


def test__sparse_pool_keeps_fill_but_demotes():
    # With fewer than MIN_USEFUL_CORPUS strong papers, weak context is
    # kept as fill (corpus preservation for ultra-narrow topics).
    q = "Morphometric analysis of drainage basins"
    one_good = _paper(
        "Drainage morphometry of a mountain basin",
        "Stream network morphometric analysis of the alpine watershed.",
        8,
    )
    weak = _paper(
        "Catfish morphometric characters survey",
        "Fish body measurements population study.",
        3,
    )
    ranked = rank_papers(q, [one_good, weak], max_n=10)
    assert len(ranked) == 2
    assert ranked[0][0].title.startswith("Drainage")

"""Benchmark harness for research pipeline retrieval quality.

Measures recall@K, MRR (Mean Reciprocal Rank), and precision against
gold-standard query→expected-DOI fixture sets.

Usage:
    uv run python -m pytest tests/unit/test_research_benchmarks.py -v -m live
    # Or standalone:
    uv run python src/geokit/research/scripts/_benchmarks.py

The benchmark fixtures are CURATED — each contains landmark DOIs that
any competent earth-science search MUST return for that query.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger("scientific_research.benchmarks")


# =============================================================================
# Gold-standard fixtures: query → set of expected landmark DOIs
# =============================================================================


BENCHMARK_FIXTURES: list[dict] = [
    {
        "query": "pyroxene thermometry",
        "expected_dois": {
            "10.2138/am-1997-3-413",     # Brey & Kohler two-pyroxene
            "10.1016/0009-2541(87)90235-x",  # Lindsley 1983
            "10.1093/petrology/egm054",  # Putirka 2008
        },
        "description": "Classic pyroxene thermometry calibrations",
    },
    {
        "query": "garnet thermometry calibration",
        "expected_dois": {
            "10.1007/bf00387203",       # Ellis & Green garnet-cpx
            "10.1306/002212b96847",      # Hodges & Spear garnet-biotite
            "10.2138/am-1997-3-413",     # Brey & Kohler
        },
        "description": "Garnet-based thermometers with experimental calibration",
    },
    {
        "query": "zircon U-Pb geochronology",
        "expected_dois": {
            "10.1016/s0016-7037(02)00961-5",  # zircon saturation
            "10.1093/petrology/egm042",       # zircon growth
        },
        "description": "Zircon U-Pb dating methodology",
    },
    {
        "query": "amphibole thermobarometry",
        "expected_dois": {
            "10.2138/am-2012-4276",      # Ridolfi amphibole
            "10.1093/petrology/egm024",  # Holland & Blundy amphibole-plagioclase
        },
        "description": "Amphibole thermometers and barometers",
    },
    {
        "query": "Ti-in-zircon thermometry",
        "expected_dois": {
            "10.1016/j.chemgeo.2005.03.007",  # Ferry & Watson Ti-in-zircon
            "10.1093/petrology/egm042",
        },
        "description": "TitaniQ thermometer calibration",
    },
]


# =============================================================================
# Metrics
# =============================================================================


def recall_at_k(retrieved_dois: list[str], expected_dois: set[str], k: int = 20) -> float:
    """Recall@K: fraction of expected DOIs found in top-K retrieved."""
    top_k = set(d.lower().strip() for d in retrieved_dois[:k])
    expected = set(d.lower().strip() for d in expected_dois)
    if not expected:
        return 0.0
    hits = len(top_k & expected)
    return hits / len(expected)


def mrr(retrieved_dois: list[str], expected_dois: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    expected = set(d.lower().strip() for d in expected_dois)
    for i, doi in enumerate(retrieved_dois):
        if doi.lower().strip() in expected:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_dois: list[str], expected_dois: set[str], k: int = 10) -> float:
    """Precision@K: fraction of top-K retrieved that are relevant."""
    top_k = set(d.lower().strip() for d in retrieved_dois[:k])
    expected = set(d.lower().strip() for d in expected_dois)
    if not top_k:
        return 0.0
    hits = len(top_k & expected)
    return hits / len(top_k)


def evaluate_fixture(fixture: dict, retrieved_dois: list[str]) -> dict:
    """Evaluate one benchmark fixture."""
    expected = fixture["expected_dois"]
    return {
        "query": fixture["query"],
        "n_expected": len(expected),
        "n_retrieved": len(retrieved_dois),
        "recall_at_10": recall_at_k(retrieved_dois, expected, k=10),
        "recall_at_20": recall_at_k(retrieved_dois, expected, k=20),
        "mrr": mrr(retrieved_dois, expected),
        "precision_at_10": precision_at_k(retrieved_dois, expected, k=10),
        "expected_dois": expected,
        "retrieved_dois": set(retrieved_dois[:20]),
        "hits": set(d.lower().strip() for d in retrieved_dois[:20]) & set(d.lower().strip() for d in expected),
        "misses": set(d.lower().strip() for d in expected) - set(d.lower().strip() for d in retrieved_dois[:20]),
    }


def evaluate_all(retrieved_map: dict[str, list[str]]) -> dict:
    """Evaluate all fixtures. retrieved_map: {query: [doi1, doi2, ...]}.

    Returns aggregate metrics + per-fixture breakdown.
    """
    results = []
    for fixture in BENCHMARK_FIXTURES:
        q = fixture["query"]
        retrieved = retrieved_map.get(q, [])
        result = evaluate_fixture(fixture, retrieved)
        results.append(result)

    n = len(results)
    if n == 0:
        return {"error": "no results"}

    aggregate = {
        "n_fixtures": n,
        "mean_recall_at_10": sum(r["recall_at_10"] for r in results) / n,
        "mean_recall_at_20": sum(r["recall_at_20"] for r in results) / n,
        "mean_mrr": sum(r["mrr"] for r in results) / n,
        "mean_precision_at_10": sum(r["precision_at_10"] for r in results) / n,
        "per_fixture": results,
    }
    return aggregate


def render_benchmark_report(metrics: dict) -> str:
    """Render a human-readable benchmark report."""
    lines = ["# Retrieval Benchmark Report", ""]

    lines.append("## Aggregate Metrics")
    lines.append(f"- **Fixtures**: {metrics['n_fixtures']}")
    lines.append(f"- **Mean Recall@10**: {metrics['mean_recall_at_10']:.2%}")
    lines.append(f"- **Mean Recall@20**: {metrics['mean_recall_at_20']:.2%}")
    lines.append(f"- **Mean MRR**: {metrics['mean_mrr']:.3f}")
    lines.append(f"- **Mean Precision@10**: {metrics['mean_precision_at_10']:.2%}")
    lines.append("")

    lines.append("## Per-Fixture Breakdown")
    lines.append("| Query | Recall@10 | Recall@20 | MRR | Precision@10 | Hits | Misses |")
    lines.append("|-------|-----------|-----------|-----|--------------|------|--------|")
    for r in metrics["per_fixture"]:
        lines.append(
            f"| {r['query'][:30]} | {r['recall_at_10']:.0%} | {r['recall_at_20']:.0%} | "
            f"{r['mrr']:.3f} | {r['precision_at_10']:.0%} | "
            f"{len(r['hits'])}/{r['n_expected']} | {len(r['misses'])} |"
        )
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run live retrieval for each fixture
    sys.path.insert(0, str(Path(__file__).parent))
    from _sources import crossref_search, dedup_papers, openalex_search

    retrieved_map: dict[str, list[str]] = {}
    for fixture in BENCHMARK_FIXTURES:
        q = fixture["query"]
        log.info("Benchmarking: %s", q)
        cr = crossref_search(q, max_results=15)
        oa = openalex_search(q, max_results=15)
        all_papers = cr + oa
        deduped = dedup_papers(all_papers)
        dois = [p.doi for p in deduped if p.doi]
        retrieved_map[q] = dois
        log.info("  Retrieved %d DOIs", len(dois))

    metrics = evaluate_all(retrieved_map)
    print(render_benchmark_report(metrics))

#!/usr/bin/env python3
"""Cross-paper correlation engine for the scientific-research skill.

Builds citation graph (Mermaid), concept clusters, temporal distribution,
correlation matrix scaffold (claims × papers), and pre-classifies citation
contexts as supporting/contrasting/mentioning (Scite-style heuristic).

SOTA approach:
    - Bibliographic coupling: papers sharing ≥N references are related
    - Co-citation: papers cited together by ≥N others are related
    - Citation context classification: BERT classifiers exist; here we use
      contrast-marker + sentiment heuristic. LLM refines.
    - Concept clustering: OpenAlex `concepts` / `topics` / S2 `fieldsOfStudy`
      are pre-computed by source APIs (no LDA needed).

References:
    - Connected Papers algorithm: co-citation + bibliographic coupling
      (https://www.connectedpapers.com/about — verified no public API)
    - Scite.ai smart-citation contexts (https://scite.ai — verified no public API)
    - Small H. Co-citation in scientific literature. JASIS 1973;24:265-269.
    - Kessler MM. Bibliographic coupling. Science 1963;132:1409-1410.
"""

from __future__ import annotations

# --- Skill venv bootstrap (shared: see _bootstrap.py) ---
if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env(extra_imports=("networkx",), extra_installs=("networkx",))
# --- End bootstrap ---


import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _render import (
    render_author_network,
    render_citation_graph,
    render_concept_clusters,
    render_funding_sources,
    render_institutional_distribution,
    render_temporal_histogram,
)
from _sources import (
    PaperRecord,
    openalex_get_cited_by,
)

log = logging.getLogger("scientific_research.correlate")


# =============================================================================
# Citation context classification — delegated to _nlp.classify_stance
# (curated lexicon + count-based hierarchy, more comprehensive than regex)
# =============================================================================


def classify_context(context: str, claim: str = "") -> tuple[str, float, str]:
    """Classify citation stance using LLM (primary) or lexicon (fallback).

    Returns (stance_label, confidence, reason).
    """
    # Try LLM first
    try:
        from _llm_extract import classify_stance as llm_stance
        from _llm_extract import is_available

        if is_available():
            # Get lexicon fallback ready
            from _nlp import classify_stance as lexicon_stance

            lex_result = lexicon_stance(context)
            fallback = (lex_result.stance, lex_result.confidence)

            stance, conf, reason = llm_stance(
                sentence=context,
                claim=claim,
                fallback=fallback,
            )
            return (stance, conf, reason)
    except Exception as e:
        log.debug("LLM stance failed, using lexicon: %s", e)

    # Fallback: lexicon
    from _nlp import classify_stance

    r = classify_stance(context)
    return (r.stance, r.confidence, r.reason)


# =============================================================================
# Bibliographic coupling + co-citation
# =============================================================================
def bibliographic_coupling(papers: list[PaperRecord]) -> dict[tuple[str, str], int]:
    """Pairs sharing referenced_works (OpenAlex IDs).
    Returns {(id_a, id_b): shared_count}."""
    refs_by_paper: dict[str, set[str]] = {}
    for p in papers:
        if p.openalex_id and p.referenced_works:
            refs_by_paper[p.openalex_id] = set(p.referenced_works)
    out: dict[tuple[str, str], int] = {}
    paper_ids = list(refs_by_paper.keys())
    for i, a in enumerate(paper_ids):
        for b in paper_ids[i + 1 :]:
            shared = refs_by_paper[a] & refs_by_paper[b]
            if len(shared) >= 2:  # threshold: at least 2 shared refs
                key = tuple(sorted([a, b]))
                out[key] = len(shared)
    return out


def fetch_forward_citations(
    paper: PaperRecord, max_n: int = 20
) -> list[tuple[PaperRecord, str]]:
    """Fetch papers citing this one + their context strings.

    OpenAlex PRIMARY (generous rate limit). S2 only if s2_paper_id already
    known (no on-demand lookup — those individual calls trigger 429 cascades).
    """
    # OpenAlex PRIMARY
    if paper.openalex_id:
        try:
            citing = openalex_get_cited_by(paper.openalex_id, max_results=max_n)
            if citing:
                return [(c, "") for c in citing]
        except Exception as e:
            log.debug("OA cited_by failed for %s: %s", paper.openalex_id, e)
    # S2 citations REMOVED — S2 rate limit (100 req/5min unauthenticated) is
    # too tight and causes 429 cascades. OpenAlex cited_by is the primary
    # source (generous rate limit, 1000 credits/24h). S2 citation context
    # strings are nice-to-have but not worth the rate-limit risk.
    return []


# =============================================================================
# Correlation matrix — FILLED by code (TF-IDF relevance + extractive finding)
# =============================================================================
def build_matrix_scaffold(
    papers: list[PaperRecord],
    focus_outcomes: list[str] | None = None,
    no_llm_stance: bool = False,
) -> dict:
    """Build correlation matrix with CELLS FILLED by code (not empty for LLM).
    Uses TF-IDF cosine relevance to decide if paper addresses claim; if yes,
    extracts most-relevant sentence via TextRank-style extractive summarization
    and classifies direction via stance lexicon on the extracted sentence.
    """
    from _nlp import extract_finding, relevance_score

    # Row labels: top concepts (proxy for claims/sub-topics)
    concept_counter: Counter = Counter()
    for p in papers:
        for c in (p.concepts or [])[:5]:
            if c:
                concept_counter[c] += 1
    focus = focus_outcomes or [c for c, n in concept_counter.most_common(8)]
    col_labels = [
        (p.doi or p.arxiv_id or p.openalex_id or p.title[:30]) for p in papers
    ]
    matrix = []
    # Relevance threshold: 0.10 = topically related; 0.20 = directly addresses
    REL_THRESHOLD = 0.10
    for claim in focus:
        row = {
            "claim": claim,
            "cells": [],
        }
        for p in papers:
            abstract = p.abstract or ""
            score = relevance_score(claim, abstract)
            if score < REL_THRESHOLD:
                # Paper doesn't address this claim
                row["cells"].append(
                    {
                        "paper_id": p.primary_id,
                        "doi": p.doi,
                        "addresses": False,
                        "relevance": round(score, 3),
                        "finding": "",
                        "direction": "n/a",
                        "effect_size": "",
                        "evidence_level": "",
                        "notes": "below relevance threshold",
                    }
                )
                continue
            # Extract most relevant sentence
            finding = extract_finding(abstract, claim, max_chars=500)
            # Classify stance: LLM for high-relevance cells only (threshold=0.5), lexicon for rest
            if score >= 0.5 and not no_llm_stance:
                stance_label, stance_conf, stance_reason = classify_context(
                    finding, claim=claim
                )
            else:
                # Low relevance: use lexicon only (no LLM call — 100x faster)
                from _nlp import classify_stance as _lex_stance

                _r = _lex_stance(finding)
                stance_label, stance_conf, stance_reason = (
                    _r.stance,
                    _r.confidence,
                    _r.reason,
                )
            row["cells"].append(
                {
                    "paper_id": p.primary_id,
                    "doi": p.doi,
                    "addresses": True,
                    "relevance": round(score, 3),
                    "finding": finding,
                    "direction": stance_label,
                    "direction_confidence": round(stance_conf, 2),
                    "direction_reason": stance_reason,
                    "effect_size": "",  # filled by extract.py downstream
                    "evidence_level": "",
                    "notes": "",
                }
            )
        matrix.append(row)
    return {
        "matrix": matrix,
        "row_labels": focus,
        "col_labels": col_labels,
        "filling_method": (
            "Cells filled by code: (1) TF-IDF cosine relevance decides if paper "
            "addresses claim (threshold 0.10), (2) TextRank-style extractive "
            "summarization picks the most-relevant sentence, (3) stance lexicon "
            "classifies direction. Effect sizes + evidence levels merged from "
            "extract.py + verify.py by the orchestrator LLM at synthesis."
        ),
    }


# =============================================================================
# Citation context summary per paper
# =============================================================================
@dataclass
class CitationContextSummary:
    paper_id: str
    n_citing_papers: int
    n_supporting: int
    n_contrasting: int
    n_mentioning: int
    top_supporting_quotes: list[str]
    top_contrasting_quotes: list[str]
    consensus_signal: str  # 'consensus' | 'contested' | 'emerging' | 'isolated'


def summarize_contexts(
    paper: PaperRecord, contexts: list[tuple[PaperRecord, str]]
) -> CitationContextSummary:
    """Classify citation contexts for one paper. Returns Scite-style summary."""
    supporting, contrasting, mentioning = [], [], []
    for citing_paper, ctx in contexts:
        if not ctx.strip():
            mentioning.append(ctx)
            continue
        label, conf, _reason = classify_context(ctx)
        if label == "supporting" and conf >= 0.6:
            supporting.append(ctx[:200])
        elif label == "contrasting" and conf >= 0.6:
            contrasting.append(ctx[:200])
        else:
            mentioning.append(ctx[:200])
    n_citing = len(contexts)
    if n_citing == 0:
        signal = "isolated"
    elif len(supporting) > 3 and not contrasting:
        signal = "consensus"
    elif contrasting:
        signal = "contested"
    elif len(supporting) + len(contrasting) < 3:
        signal = "emerging"
    else:
        signal = "consensus"
    return CitationContextSummary(
        paper_id=paper.primary_id,
        n_citing_papers=n_citing,
        n_supporting=len(supporting),
        n_contrasting=len(contrasting),
        n_mentioning=len(mentioning),
        top_supporting_quotes=supporting[:3],
        top_contrasting_quotes=contrasting[:3],
        consensus_signal=signal,
    )


# =============================================================================
# Main correlate
# =============================================================================
def correlate_corpus(
    papers: list[PaperRecord],
    max_contexts_per_paper: int = 20,
    fetch_citations: bool = True,
    no_llm_stance: bool = False,
) -> dict:
    """Run full correlation analysis on verified corpus."""
    # 1. Citation graph + bibliographic coupling
    graph = render_citation_graph([p.to_dict() for p in papers])
    coupling = bibliographic_coupling(papers)
    # 2. Concept clusters
    clusters = render_concept_clusters([p.to_dict() for p in papers])
    # 3. Temporal distribution
    timeline = render_temporal_histogram([p.to_dict() for p in papers])
    # 4. Author network
    authors_net = render_author_network([p.to_dict() for p in papers])
    # 5. Funding sources
    funding = render_funding_sources([p.to_dict() for p in papers])
    # 6. Institutional / geographic
    institutions = render_institutional_distribution([p.to_dict() for p in papers])
    # 7. Correlation matrix scaffold
    matrix = build_matrix_scaffold(papers, no_llm_stance=no_llm_stance)
    # 8. Citation context summary per paper (Scite-style)
    contexts_summary: list[CitationContextSummary] = []
    if fetch_citations:
        for i, p in enumerate(papers, 1):
            log.info("[%d/%d] fetching citations for %s", i, len(papers), p.primary_id)
            try:
                ctxs = fetch_forward_citations(p, max_n=max_contexts_per_paper)
                contexts_summary.append(summarize_contexts(p, ctxs))
            except Exception as e:
                log.warning("Citation context fetch failed for %s: %s", p.primary_id, e)
                contexts_summary.append(
                    CitationContextSummary(
                        paper_id=p.primary_id,
                        n_citing_papers=0,
                        n_supporting=0,
                        n_contrasting=0,
                        n_mentioning=0,
                        top_supporting_quotes=[],
                        top_contrasting_quotes=[],
                        consensus_signal="unknown",
                    )
                )

    # 9. Per-paper key findings (extractive — non-LLM by default)
    from _classifiers import classify_novelty, detect_discipline, extract_key_finding

    key_findings: list[dict] = []
    for p in papers:
        abstract = p.abstract or ""
        if not abstract.strip():
            continue
        finding = extract_key_finding(abstract, max_chars=300)
        if not finding:
            continue
        key_findings.append(
            {
                "paper_id": p.primary_id,
                "doi": p.doi or "",
                "title": (p.title or "")[:120],
                "year": p.year or "",
                "discipline": detect_discipline(abstract) or "",
                "novelty": classify_novelty(abstract) or "",
                "key_finding": finding,
            }
        )
    log.info("Extracted key findings for %d/%d papers", len(key_findings), len(papers))

    return {
        "n_papers": len(papers),
        "citation_graph_mermaid": graph,
        "bibliographic_coupling_edges": [
            {"a": a, "b": b, "shared_refs": n}
            for (a, b), n in sorted(coupling.items(), key=lambda x: -x[1])
        ],
        "concept_clusters_mermaid": clusters,
        "temporal_histogram": timeline,
        "author_network_mermaid": authors_net,
        "funding_sources": funding,
        "institutions": institutions,
        "correlation_matrix": matrix,
        "citation_contexts": [asdict(c) for c in contexts_summary],
        "chronological_timeline": _build_chronological_timeline(papers),
        "key_findings": key_findings,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _build_chronological_timeline(papers: list[PaperRecord]) -> list[dict]:
    """Build chronological timeline of papers sorted by year.

    Identifies milestone papers (high citation count) and tracks
    methodological/conceptual evolution over time.
    """
    # Sort by year (None years go last)
    sorted_papers = sorted(
        papers, key=lambda p: (p.year or 9999, -(p.citation_count or 0))
    )

    # Determine citation threshold for "milestone" (top 25% by citations)
    citations = [p.citation_count or 0 for p in sorted_papers if p.citation_count]
    milestone_threshold = sorted(citations)[len(citations) * 3 // 4] if citations else 0

    timeline = []
    for p in sorted_papers:
        year = p.year or 0
        cites = p.citation_count or 0
        is_milestone = cites >= milestone_threshold and cites > 0

        timeline.append(
            {
                "year": year,
                "title": (p.title or "?")[:120],
                "doi": p.doi,
                "authors": [a.get("name", "") for a in (p.authors or [])[:3]],
                "citation_count": cites,
                "is_milestone": is_milestone,
                "venue": (p.venue or "")[:80],
                "openalex_id": p.openalex_id,
            }
        )

    return timeline


# =============================================================================
# Rendered report (compact, for LLM consumption)
# =============================================================================
def render_correlation_report(result: dict) -> str:
    out = ["# Cross-Paper Correlation Report", ""]
    out.append(f"Corpus: {result['n_papers']} verified papers")
    out.append("")
    out.append("## Consensus / disagreement signals (per paper)")
    out.append("| Paper | Citing | Support | Contrast | Signal |")
    out.append("|---|---|---|---|---|")
    for c in result["citation_contexts"]:
        ident = c["paper_id"]
        out.append(
            f"| `{ident[:40]}` | {c['n_citing_papers']} | "
            f"{c['n_supporting']} | {c['n_contrasting']} | "
            f"**{c['consensus_signal']}** |"
        )
    out.append("")
    out.append("## Concept clusters")
    out.append("```mermaid")
    out.append(result["concept_clusters_mermaid"])
    out.append("```")
    out.append("")
    out.append("## Temporal distribution")
    out.append(result["temporal_histogram"])
    out.append("")
    out.append("## Chronological timeline (milestone papers marked with ★)")
    out.append("| Year | Title | Cites | Venue |")
    out.append("|---|---|---|---|")
    for entry in result.get("chronological_timeline", []):
        marker = " ★" if entry.get("is_milestone") else ""
        title = entry["title"][:80] + marker
        out.append(
            f"| {entry['year']} | {title} | {entry['citation_count']} | {entry.get('venue', '')} |"
        )
    out.append("")
    out.append("## Funding sources")
    out.append(result["funding_sources"])
    out.append("")
    out.append("## Institutions / countries")
    out.append(result["institutions"])
    out.append("")
    out.append("## Correlation matrix (stance direction)")
    out.append(
        result["correlation_matrix"].get("filling_method")
        or result["correlation_matrix"].get("filling_instructions", "")
    )
    out.append("")
    out.append(
        "| Claim \\ Paper | "
        + " | ".join(
            f"`{c[:15]}`"
            for c in result.get("correlation_matrix", {}).get("col_labels", [])[:8]
        )
        + " |"
    )
    out.append(
        "|"
        + "---|"
        * (min(8, len(result.get("correlation_matrix", {}).get("col_labels", []))) + 1)
    )
    for row in result.get("correlation_matrix", {}).get("matrix", [])[:10]:
        cells = row["cells"][:8]
        out.append(
            "| **"
            + row["claim"][:40]
            + "** | "
            + " | ".join(c["direction"] or "—" for c in cells)
            + " |"
        )
    out.append("")

    # ------------------------------------------------------------------
    # Finding excerpts per claim-paper pair (the actual evidence sentence)
    # ------------------------------------------------------------------
    out.append("## Claim-Paper finding excerpts (evidence sentences)")
    out.append(
        "> Each row shows the sentence extracted from the paper's abstract "
        "most relevant to the claim, plus the classified stance.\n"
    )
    for row in result.get("correlation_matrix", {}).get("matrix", [])[:15]:
        claim = row["claim"]
        out.append(f"### Claim: _{claim[:100]}_\n")
        out.append("| Paper | Stance | Relevance | Finding excerpt |")
        out.append("|---|---|---|---|")
        for cell in row["cells"]:
            if not cell.get("addresses"):
                continue
            finding = (cell.get("finding") or "").strip()
            if not finding:
                continue
            # Truncate finding for table readability
            if len(finding) > 200:
                finding = finding[:197] + "..."
            # Escape pipe characters in finding
            finding_safe = finding.replace("|", "\\|").replace("\n", " ")
            direction = cell.get("direction", "—") or "—"
            relevance = cell.get("relevance", 0.0)
            paper_id = cell.get("paper_id", "?")[:30]
            out.append(
                f"| `{paper_id}` | **{direction}** | {relevance:.2f} | {finding_safe} |"
            )
        out.append("")

    if result["bibliographic_coupling_edges"]:
        out.append("## Bibliographic coupling (papers sharing ≥2 references)")
        for edge in result["bibliographic_coupling_edges"][:10]:
            out.append(
                f"- `{edge['a'][-20:]}` ⟷ `{edge['b'][-20:]}` — {edge['shared_refs']} shared refs"
            )

    # ------------------------------------------------------------------
    # Per-paper key findings (standalone — not tied to claims)
    # ------------------------------------------------------------------
    key_findings = result.get("key_findings", [])
    if key_findings:
        out.append("")
        out.append("## Key findings by paper")
        out.append(
            "> Extractive summary — the single most result-oriented sentence "
            "from each paper's abstract (non-LLM by default).\n"
        )
        out.append("| Year | Discipline | Novelty | Title | Key finding | DOI |")
        out.append("|---|---|---|---|---|---|")
        for kf in key_findings:
            title = (kf.get("title") or "")[:60].replace("|", "\\|")
            finding = (kf.get("key_finding") or "").strip()
            if len(finding) > 250:
                finding = finding[:247] + "..."
            finding_safe = finding.replace("|", "\\|").replace("\n", " ")
            doi = kf.get("doi") or ""
            doi_cell = f"[{doi}](https://doi.org/{doi})" if doi else "—"
            out.append(
                f"| {kf.get('year', '')} | {kf.get('discipline', '')} | "
                f"{kf.get('novelty', '')} | {title} | {finding_safe} | {doi_cell} |"
            )

    return "\n".join(out)


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="correlate",
        description="Cross-paper correlation: graph + clusters + matrix + contexts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("verified", type=Path, help="verified.json from verify.py")
    p.add_argument(
        "--no-contexts",
        action="store_true",
        help="skip citation context mining (faster, no Scite-style analysis)",
    )
    p.add_argument(
        "--use-llm-stance",
        action="store_true",
        help="use LLM for stance classification (higher quality, ~3s/cell). Default: SVM+lexicon.",
    )
    p.add_argument(
        "--max-contexts",
        type=int,
        default=20,
        help="max citing papers to fetch per seed (default 20)",
    )
    p.add_argument(
        "--focus-outcome",
        action="append",
        default=[],
        help="specific outcome claim to use as matrix row (repeatable)",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("research_outputs/correlation.json")
    )
    p.add_argument(
        "--report", type=Path, default=Path("research_outputs/correlation.md")
    )
    p.add_argument("--graph", type=Path, default=Path("research_outputs/graph.mmd"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    from _artifact import ArtifactShapeError, load_verified

    try:
        data = load_verified(args.verified)
    except ArtifactShapeError as e:
        raise SystemExit(f"FATAL: {e}") from e
    papers = [PaperRecord.from_dict(d) for d in data.get("papers", [])]
    log.info("Loaded %d verified papers (typed artifact contract)", len(papers))

    # Auto-detect S2 availability — if down, skip citation contexts
    fetch_citations = not args.no_contexts
    if fetch_citations:
        try:
            import urllib.error
            import urllib.request

            s2_req = urllib.request.Request(
                "https://api.semanticscholar.org/graph/v1/paper/DOI:10.1038/nature12373?fields=title",
                method="GET",
            )
            urllib.request.urlopen(s2_req, timeout=5)
        except (urllib.error.URLError, OSError):
            log.info("S2 API unavailable — skipping citation contexts")
            fetch_citations = False

    result = correlate_corpus(
        papers,
        max_contexts_per_paper=args.max_contexts,
        fetch_citations=fetch_citations,
        no_llm_stance=not args.use_llm_stance,
    )
    from _artifact import save_artifact

    save_artifact(result, args.output)
    print(f"Wrote correlation.json → {args.output}")

    args.report.write_text(render_correlation_report(result))
    print(f"Wrote correlation.md → {args.report}")

    args.graph.write_text(result["citation_graph_mermaid"])
    print(f"Wrote Mermaid graph → {args.graph}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

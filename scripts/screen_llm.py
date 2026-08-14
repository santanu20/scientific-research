"""Paper screening — LLM-only, no regex.

Replaces the old 7-stage regex pipeline (800+ LOC of _GEO_QUERY_SIGNALS,
_GEO_PAPER_SIGNALS, _OFF_TOPIC_PATTERNS, etc.) with a single semantic
authority: the LLM.

Architecture:
  1. Content-type filter — structural check for non-research items
     (dictionary entries, front matter, peer reviews). ~15 LOC, no regex.
  2. LLM judge — sole relevance authority. Understands ANY query including
     misspellings, compound words, and jargon. Cached per paper+query.
  3. Fallback — when LLM unavailable, accept all (better to include than
     to exclude with broken regex).

Per O'Mara-Eves et al. 2015 (DOI:10.1186/2046-4053-4-5):
  keyword methods plateau ~70% precision; semantic methods reach 90%+.
"""

from __future__ import annotations

import logging
from typing import Protocol


class ScreenablePaper(Protocol):
    """Structural type screen_paper accepts (PaperRecord, dict adapters, test doubles)."""

    title: str
    abstract: str
    raw_metadata: dict


_log = logging.getLogger(__name__)

# =============================================================================
# Content-type filter — structural patterns for non-research items
# =============================================================================

# Titles that are NEVER research papers. Simple substring match — no regex bag.
# These are structural document parts, not topical filters.
_NON_RESEARCH_TITLES = frozenset(
    {
        "front matter",
        "back matter",
        "index",
        "copyright",
        "preface",
        "foreword",
        "acknowledgments",
        "concluding remarks",
        "table of contents",
        "cover",
        "errata",
        "erratum",
        "corrigendum",
    }
)

# Titles starting with these patterns are peer-review comments, not papers.
_NON_RESEARCH_PREFIXES = (
    "review for ",
    "review of ms",
    "review of manuscript",
    "referee report",
    "reviewer comment",
    "disclosure",
)

# Publication types that are NEVER research papers (from raw_metadata.type_crossref).
_NON_RESEARCH_TYPES = frozenset(
    {
        "reference-entry",
        "dictionary-entry",
        "book-entry",
        "encyclopedia-entry",
    }
)


def _is_non_research(
    title: str, abstract: str, raw_metadata: dict | None = None
) -> tuple[bool, str]:
    """Detect non-research items (dictionary, front matter, peer review, etc.).

    Returns (True, reason) if the paper is not a research article.
    These are STRUCTURAL document types — not topical relevance judgments.
    """
    title_lower = title.strip().lower()

    # 1. Exact title match (dictionary entries, front matter, etc.)
    if title_lower in _NON_RESEARCH_TITLES:
        return True, f"non-research title: {title_lower!r}"

    # 2. Peer-review / referee comments
    for prefix in _NON_RESEARCH_PREFIXES:
        if title_lower.startswith(prefix):
            return True, f"peer review / referee comment: {title_lower[:50]!r}"

    # 3. Supplemental review material (review text published as supplement)
    if "review for " in title_lower and "supplemental" in title_lower:
        return True, "supplemental review material"

    # 4. Publication type from metadata
    if raw_metadata:
        ptype = (
            raw_metadata.get("type") or raw_metadata.get("type_crossref") or ""
        ).lower()
        if ptype in _NON_RESEARCH_TYPES:
            return True, f"non-research type: {ptype!r}"

    # 5. Single-word title + no abstract — likely dictionary/encyclopedia entry.
    # Real research papers have descriptive multi-word titles.
    title_words = title.strip().split()
    if (
        len(title_words) <= 2
        and len(title.strip()) < 30
        and len(abstract.strip()) < 100
        and not any(c.isdigit() for c in title)
    ):
        return True, "single-word title + no abstract (likely encyclopedia)"

    return False, ""


# =============================================================================
# Screen — the ENTIRE screening logic
# =============================================================================


def screen_paper(
    paper: ScreenablePaper,
    query: str,
    *,
    use_llm: bool = False,
    llm_model: str | None = None,
    research_type: str | None = None,
    ensemble: bool = False,
) -> tuple[bool, str, str]:
    """Screen a paper for topical relevance. Returns (decision, stage, reason).

    Stages:

    1. Content-type filter (always runs, ~0ms):
       Excludes non-research items (dictionary, front matter, peer reviews).
       These are structural types that no LLM should waste time judging.

    2. LLM judge (when use_llm=True, ~2s/call):
       Asks the LLM whether the paper is relevant to the query.
       Handles misspellings, compound words, jargon — ANY query.
       ensemble=True: majority vote across distinct available models
       (Sanghera 2025 JAMIA 32(5):893-904) instead of a single judge.

    When use_llm=False: only content-type filter runs.
    Permissive by design — better to include a marginal paper than
    to exclude with a broken filter.
    """
    title = (getattr(paper, "title", "") or "").strip()
    abstract = (getattr(paper, "abstract", "") or "").strip()
    raw_metadata = getattr(paper, "raw_metadata", None) or {}

    # Stage 1: Content-type filter — always runs
    non_research, reason = _is_non_research(title, abstract, raw_metadata)
    if non_research:
        return False, "content_type", reason

    # Empty paper — nothing to judge
    if not title and not abstract:
        return False, "empty", "no title and no abstract"

    # Stage 2: LLM judge — sole relevance authority
    if use_llm:
        try:
            if ensemble:
                from _llm_extract import llm_screen_paper_ensemble

                verdict = llm_screen_paper_ensemble(
                    paper, query, research_type=research_type
                )
                if verdict is not None:
                    relevant, llm_reason, _votes = verdict
                    stage = "llm_ensemble" if relevant else "llm_ensemble_reject"
                    if not relevant:
                        return False, stage, f"LLM ensemble: {llm_reason}"
                    return True, stage, f"LLM ensemble: {llm_reason}"
            else:
                from _llm_extract import llm_screen_paper

                verdict = llm_screen_paper(
                    paper, query, model=llm_model, research_type=research_type
                )
                if verdict is not None:
                    relevant, llm_reason = verdict
                    if not relevant:
                        return False, "llm_judge", f"LLM: {llm_reason}"
                    return True, "llm_pass", f"LLM: {llm_reason}"
            # LLM unavailable — fall through to permissive accept
            return True, "llm_unavailable", "LLM unavailable — accepted permissively"
        except Exception as e:
            _log.debug("LLM judge failed, accepting permissively: %s", e)
            return True, "llm_error", f"LLM error: {e}"

    # Stage 3: Fallback — no LLM, accept all geo-ish papers
    return True, "pass", "accepted (no LLM filtering)"


__all__ = [
    "_is_non_research",
    "screen_paper",
]

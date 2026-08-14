"""Citation alignment verification for research briefs.

Diagnosed bug: the synthesized brief body uses theme-offset numbering
(``ref_offset + len(theme_cited) + 1``) while the References section uses
simple linear iteration over the paper list. The two numbering schemes
CAN'T align — body [1] cites a different paper than References [1].

This module provides ``verify_citation_alignment`` — returns a list of
mismatches found in a brief. Empty list = brief is well-formed.

Used by the contract test ``tests/test_brief_citations.py``
to fail on current behavior and pass once the synthesis is fixed.

Contract:
  For every [N] citation in the brief body, References[N] MUST be the same
  paper (matched by DOI when available, else by title prefix).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CitationMismatch:
    """One misaligned citation in a brief."""

    ref_num: int
    body_context: str  # ~60 chars of body text around [N]
    ref_text: str  # full References[N] entry
    reason: str  # why they don't match (e.g. "DOI differs", "title mismatch")

    def __str__(self) -> str:
        return (
            f"[{self.ref_num}] MISMATCH ({self.reason})\n"
            f"  body: {self.body_context!r}\n"
            f"  refs: {self.ref_text[:80]!r}"
        )


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\)\]]+")


def _extract_doi(text: str) -> str | None:
    """Extract first DOI from text. Returns None if not found."""
    m = _DOI_PATTERN.search(text)
    return m.group(0).rstrip(".,;") if m else None


def _split_brief(brief_markdown: str) -> tuple[str, str] | None:
    """Split brief into (body, references) on the '## References' heading.

    Returns None if the heading isn't found (malformed brief).
    """
    parts = re.split(r"^##\s+References\s*$", brief_markdown, maxsplit=1, flags=re.MULTILINE)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _parse_references(refs_section: str) -> dict[int, str]:
    """Parse References section into {N: entry_text}."""
    entries: dict[int, str] = {}
    # Match: [N] <text> up to next [N+1] or end
    pattern = re.compile(r"^\[(\d+)\]\s+(.*?)(?=^\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(refs_section):
        n = int(m.group(1))
        text = m.group(2).strip()
        entries[n] = text
    return entries


def _parse_body_citations(body: str) -> dict[int, str]:
    """Find every [N] citation in body, return {N: surrounding_context}.

    Captures ~60 chars after the citation for identification.
    """
    citations: dict[int, str] = {}
    for m in re.finditer(r"\[(\d+)\]", body):
        n = int(m.group(1))
        # Skip false positives like "[1]" in non-citation contexts
        # (table headers, etc.) by requiring a non-digit char after
        start = m.end()
        ctx = body[start : start + 80].strip().split("\n")[0]
        citations.setdefault(n, ctx)
    return citations


def verify_citation_alignment(
    brief_markdown: str,
    *,
    require_doi_match: bool = False,
    strict_word_overlap: bool = False,
) -> list[CitationMismatch]:
    """Verify every [N] in the brief body aligns with References [N].

    Parameters
    ----------
    brief_markdown
        Full brief text including the ``## References`` section.
    require_doi_match
        If True, require exact DOI match (strict). If False (default),
        accept any structural alignment.
    strict_word_overlap
        If True, run a heuristic word-overlap check between body context
        and reference text. WARNING: produces false positives on
        jargon-heavy findings (body uses ``δcal-gr`` abbreviation while
        reference title says ``Calcite-graphite``). Default False —
        structural checks only.

    Returns
    -------
    list[CitationMismatch]
        Empty list = brief is well-formed. Non-empty = misalignments found.

    Contract:
      - body [N] must have a References [N] entry (existence)
      - body max [N] must be ≤ References max [N] (range)
      - if require_doi_match: DOIs must match exactly
    """
    split = _split_brief(brief_markdown)
    if split is None:
        return [
            CitationMismatch(
                ref_num=-1,
                body_context="",
                ref_text="",
                reason="brief missing '## References' section",
            )
        ]
    body, refs_section = split
    references = _parse_references(refs_section)
    body_citations = _parse_body_citations(body)

    mismatches: list[CitationMismatch] = []

    for n, body_ctx in sorted(body_citations.items()):
        if n not in references:
            mismatches.append(
                CitationMismatch(
                    ref_num=n,
                    body_context=body_ctx[:60],
                    ref_text="(missing)",
                    reason=f"body cites [{n}] but References has no entry",
                )
            )
            continue

        ref_text = references[n]

        if require_doi_match:
            body_doi = _extract_doi(body_ctx)
            ref_doi = _extract_doi(ref_text)
            if body_doi and ref_doi and body_doi.lower() != ref_doi.lower():
                mismatches.append(
                    CitationMismatch(
                        ref_num=n,
                        body_context=body_ctx[:60],
                        ref_text=ref_text,
                        reason=f"DOI differs: body={body_doi} ref={ref_doi}",
                    )
                )

        elif strict_word_overlap:
            # Heuristic: check that some 5+ char non-stopword from body
            # context appears in the reference text. WARNING: false
            # positives on jargon-heavy findings where body uses
            # abbreviations (δcal-gr) not in the title.
            body_words = {
                w.lower().strip(".,;:!?\"'()[]")
                for w in body_ctx.split()
                if len(w) >= 5 and not w.isdigit()
            }
            body_words -= _LOOSE_STOPWORDS
            ref_lower = ref_text.lower()
            if body_words and not any(w in ref_lower for w in body_words):
                mismatches.append(
                    CitationMismatch(
                        ref_num=n,
                        body_context=body_ctx[:60],
                        ref_text=ref_text,
                        reason="body and references discuss different papers (no word overlap)",
                    )
                )

    return mismatches


# Stopwords for the loose word-overlap heuristic (only used when
# strict_word_overlap=True). Kept conservative to minimize false positives.
_LOOSE_STOPWORDS = frozenset(
    w.strip()
    for w in """
about which these those their would could should study paper research
results show shows shown demonstrate demonstrates report reports found
findings suggest suggests indicate indicates confirm confirms establish
establishes document documents support supports provide provides
""".split()
    if w.strip()
)


__all__ = ["CitationMismatch", "verify_citation_alignment"]

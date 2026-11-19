"""Verbatim evidence locator construction from parsed source text."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .models import EvidenceSpan


class EvidenceLocatorError(ValueError):
    """Raised when a quotation cannot be grounded unambiguously."""


def source_sha256(source_content: bytes) -> str:
    """Hash the exact bytes stored as a source version."""
    if not isinstance(source_content, bytes) or not source_content:
        raise EvidenceLocatorError("source_content must be non-empty bytes")
    return hashlib.sha256(source_content).hexdigest()


def build_evidence_span(
    *,
    evidence_id: str,
    source_id: str,
    source_content: bytes,
    page_texts: Sequence[str],
    quote: str,
    page_number: int | None = None,
    section: str | None = None,
) -> EvidenceSpan:
    """Locate one exact quote and return a page-local evidence span.

    Page numbers are one-based. Character offsets are zero-based and local to
    the selected page text. The caller must ensure page_texts were extracted
    from the exact source_content bytes whose hash is recorded.

    No fuzzy matching or whitespace normalization is performed. If no page
    is supplied, the quote must occur exactly once across all pages.
    """
    if not isinstance(source_content, bytes) or not source_content:
        raise EvidenceLocatorError("source_content must be non-empty bytes")
    if not isinstance(page_texts, Sequence) or isinstance(page_texts, (str, bytes)) or not page_texts:
        raise EvidenceLocatorError("page_texts must be a non-empty sequence of page strings")
    if not isinstance(quote, str) or not quote.strip():
        raise EvidenceLocatorError("quote must be a non-empty verbatim string")
    if section is not None and (not isinstance(section, str) or not section.strip()):
        raise EvidenceLocatorError("section must be a non-empty string when supplied")
    for index, page_text in enumerate(page_texts, start=1):
        if not isinstance(page_text, str):
            raise EvidenceLocatorError(f"page_texts[{index}] must be a string")

    if page_number is not None:
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 1 <= page_number <= len(page_texts)
        ):
            raise EvidenceLocatorError("page_number must be a valid one-based page index")
        candidates = [page_number]
    else:
        candidates = list(range(1, len(page_texts) + 1))

    matches: list[tuple[int, int]] = []
    for candidate in candidates:
        page = page_texts[candidate - 1]
        cursor = 0
        while True:
            position = page.find(quote, cursor)
            if position < 0:
                break
            matches.append((candidate, position))
            cursor = position + 1

    if not matches:
        raise EvidenceLocatorError("quote was not found verbatim in the requested page or pages")
    if len(matches) > 1:
        raise EvidenceLocatorError(
            "quote is ambiguous; provide a page_number with exactly one occurrence or a more specific quote"
        )

    matched_page, start = matches[0]
    end = start + len(quote)
    digest = source_sha256(source_content)
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        quote=quote,
        source_sha256=digest,
        locator=f"page:{matched_page};char:{start}-{end}",
        page_number=matched_page,
        section=section,
        char_start=start,
        char_end=end,
    )


__all__ = ["EvidenceLocatorError", "build_evidence_span", "source_sha256"]

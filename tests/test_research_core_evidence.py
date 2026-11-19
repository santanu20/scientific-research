from __future__ import annotations

import hashlib

import pytest

from research_core.evidence import EvidenceLocatorError, build_evidence_span, source_sha256
from research_core.models import ResearchBundle, ResearchProtocol, SourceRecord


SOURCE_BYTES = b"%PDF-1.7 exact immutable test fixture"


def test_builds_verbatim_page_and_character_locator() -> None:
    pages = ("Introduction", "Methods\nA precise quotation appears here.", "Discussion")
    quote = "A precise quotation appears here."
    span = build_evidence_span(
        evidence_id="e1",
        source_id="s1",
        source_content=SOURCE_BYTES,
        page_texts=pages,
        quote=quote,
        section="Methods",
    )
    assert span.page_number == 2
    assert span.section == "Methods"
    assert span.source_sha256 == hashlib.sha256(SOURCE_BYTES).hexdigest()
    assert span.char_start == pages[1].index(quote)
    assert span.char_end - span.char_start == len(quote)
    assert span.locator == f"page:2;char:{span.char_start}-{span.char_end}"


def test_page_number_disambiguates_repetition_across_pages() -> None:
    pages = ("Repeated exact statement.", "Repeated exact statement.")
    with pytest.raises(EvidenceLocatorError, match="ambiguous"):
        build_evidence_span(
            evidence_id="e1", source_id="s1", source_content=SOURCE_BYTES,
            page_texts=pages, quote="Repeated exact statement.",
        )
    span = build_evidence_span(
        evidence_id="e2", source_id="s1", source_content=SOURCE_BYTES,
        page_texts=pages, quote="Repeated exact statement.", page_number=2,
    )
    assert span.page_number == 2


def test_repeated_quote_on_same_page_remains_ambiguous() -> None:
    with pytest.raises(EvidenceLocatorError, match="ambiguous"):
        build_evidence_span(
            evidence_id="e1", source_id="s1", source_content=SOURCE_BYTES,
            page_texts=("The same phrase. Then the same phrase.",),
            quote="the same phrase.", page_number=1,
        )


def test_rejects_paraphrases_and_non_exact_text() -> None:
    with pytest.raises(EvidenceLocatorError, match="not found verbatim"):
        build_evidence_span(
            evidence_id="e1", source_id="s1", source_content=SOURCE_BYTES,
            page_texts=("The intervention significantly improved the outcome.",),
            quote="The intervention improved outcomes.",
        )


@pytest.mark.parametrize("page_number", [0, -1, 3, True, 1.5])
def test_rejects_invalid_one_based_page_number(page_number) -> None:
    with pytest.raises(EvidenceLocatorError, match="one-based"):
        build_evidence_span(
            evidence_id="e1", source_id="s1", source_content=SOURCE_BYTES,
            page_texts=("One page only.",), quote="One page only.",
            page_number=page_number,
        )


def test_hashes_exact_source_bytes() -> None:
    assert source_sha256(SOURCE_BYTES) == hashlib.sha256(SOURCE_BYTES).hexdigest()
    with pytest.raises(EvidenceLocatorError, match="non-empty bytes"):
        source_sha256(b"")


def test_generated_span_matches_research_bundle_source_version() -> None:
    span = build_evidence_span(
        evidence_id="e1", source_id="s1", source_content=SOURCE_BYTES,
        page_texts=("Results show a clear change.",),
        quote="Results show a clear change.", page_number=1, section="Results",
    )
    source = SourceRecord(
        source_id="s1", provider="test", provider_record_id="record-1",
        title="A reproducibility fixture",
        retrieved_at_utc="2026-11-19T11:00:00Z",
        content_sha256=span.source_sha256,
    )
    protocol = ResearchProtocol(
        protocol_id="p1", research_question="Does X change Y?",
        created_at_utc="2026-11-19T10:59:00Z",
        inclusion_criteria=("Primary studies",), exclusion_criteria=(),
        planned_sources=("test",),
    )
    bundle = ResearchBundle(protocol, (), (source,), (span,), (), ())
    assert bundle.evidence_spans[0] == span

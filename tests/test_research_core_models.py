from __future__ import annotations

import json

import pytest

from research_core import (
    ClaimEvidenceLink,
    ClaimKind,
    ContractError,
    EvidenceRelation,
    EvidenceSpan,
    ResearchBundle,
    ResearchClaim,
    ResearchProtocol,
    ReviewStatus,
    SearchExecution,
    SearchExecutionStatus,
    SourceRecord,
)

DIGEST = "a" * 64


def sample_bundle() -> ResearchBundle:
    protocol = ResearchProtocol(
        protocol_id="protocol-001",
        research_question="Does intervention X improve outcome Y?",
        created_at_utc="2026-11-19T08:30:00Z",
        inclusion_criteria=("Human studies", "Reports outcome Y"),
        exclusion_criteria=("Editorials",),
        planned_sources=("crossref", "openalex"),
    )
    search = SearchExecution(
        execution_id="search-001",
        provider="crossref",
        exact_query='title-abs:"intervention X" AND "outcome Y"',
        executed_at_utc="2026-11-19T08:35:00+00:00",
        result_count=17,
        parameters=(("rows", "100"), ("sort", "relevance")),
        export_sha256=DIGEST,
    )
    source = SourceRecord(
        source_id="source-001",
        provider="crossref",
        provider_record_id="record-123",
        title="A study of intervention X",
        doi="10.1234/example",
        retrieved_at_utc="2026-11-19T08:40:00Z",
        publication_year=2024,
        content_sha256=DIGEST,
    )
    span = EvidenceSpan(
        evidence_id="evidence-001",
        source_id="source-001",
        quote="The primary outcome improved.",
        source_sha256=DIGEST,
        page_number=4,
        section="Results",
    )
    claim = ResearchClaim(
        claim_id="claim-001",
        statement="Intervention X improved the primary outcome in this study.",
        kind=ClaimKind.FINDING,
        created_at_utc="2026-11-19T08:45:00Z",
    )
    link = ClaimEvidenceLink(
        claim_id="claim-001",
        evidence_id="evidence-001",
        relation=EvidenceRelation.SUPPORTS,
        rationale="The cited results passage reports improvement in the named outcome.",
    )
    return ResearchBundle(protocol, (search,), (source,), (span,), (claim,), (link,))


def test_bundle_round_trip_preserves_all_records() -> None:
    bundle = sample_bundle()
    loaded = ResearchBundle.from_json(bundle.to_json())
    assert loaded == bundle
    assert loaded.to_dict()["schema_version"] == "1.0"


def test_canonical_json_and_digest_are_stable() -> None:
    bundle = sample_bundle()
    loaded = ResearchBundle.from_json(bundle.to_json(indent=None))
    assert loaded.canonical_sha256 == bundle.canonical_sha256
    assert bundle.to_json(indent=None) == loaded.to_json(indent=None)


def test_bundle_reports_unlinked_records_without_dropping_them() -> None:
    bundle = sample_bundle()
    assert bundle.unlinked_claim_ids == ()
    assert bundle.unlinked_evidence_ids == ()


def test_research_protocol_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ContractError, match="timezone"):
        ResearchProtocol(
            protocol_id="p1",
            research_question="A question",
            created_at_utc="2026-11-19T08:30:00",
            inclusion_criteria=("A criterion",),
            exclusion_criteria=(),
            planned_sources=("crossref",),
        )


def test_research_protocol_requires_explicit_scope_and_sources() -> None:
    with pytest.raises(ContractError, match="inclusion_criteria"):
        ResearchProtocol(
            protocol_id="p1",
            research_question="A question",
            created_at_utc="2026-11-19T08:30:00Z",
            inclusion_criteria=(),
            exclusion_criteria=(),
            planned_sources=("crossref",),
        )


def test_search_execution_preserves_query_and_parameters() -> None:
    search = sample_bundle().search_executions[0]
    assert search.exact_query == 'title-abs:"intervention X" AND "outcome Y"'
    assert search.parameters == (("rows", "100"), ("sort", "relevance"))


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_search_execution_rejects_invalid_result_count(count: object) -> None:
    with pytest.raises(ContractError, match="result_count"):
        SearchExecution(
            execution_id="s",
            provider="provider",
            exact_query="query",
            executed_at_utc="2026-11-19T08:30:00Z",
            result_count=count,  # type: ignore[arg-type]
        )


def test_source_requires_persistent_identifier() -> None:
    with pytest.raises(ContractError, match="persistent locator"):
        SourceRecord(
            source_id="source",
            provider="provider",
            title="A title",
            retrieved_at_utc="2026-11-19T08:30:00Z",
        )


def test_source_hash_must_be_valid_sha256() -> None:
    with pytest.raises(ContractError, match="SHA-256"):
        SourceRecord(
            source_id="source",
            provider="provider",
            provider_record_id="id",
            title="A title",
            retrieved_at_utc="2026-11-19T08:30:00Z",
            content_sha256="not-a-hash",
        )


def test_evidence_span_requires_a_locator() -> None:
    with pytest.raises(ContractError, match="requires locator"):
        EvidenceSpan(
            evidence_id="e1",
            source_id="s1",
            quote="An exact quotation.",
            source_sha256=DIGEST,
        )


def test_evidence_offsets_must_match_verbatim_quote_length() -> None:
    with pytest.raises(ContractError, match="quote length"):
        EvidenceSpan(
            evidence_id="e1",
            source_id="s1",
            quote="exact quote",
            source_sha256=DIGEST,
            page_number=1,
            char_start=10,
            char_end=20,
        )


def test_evidence_must_reference_source_content_version() -> None:
    bundle = sample_bundle()
    source = bundle.sources[0]
    span = bundle.evidence_spans[0]
    changed_source = SourceRecord(
        source_id=source.source_id,
        provider=source.provider,
        provider_record_id=source.provider_record_id,
        title=source.title,
        doi=source.doi,
        retrieved_at_utc=source.retrieved_at_utc,
        content_sha256="b" * 64,
    )
    with pytest.raises(ContractError, match="content version"):
        ResearchBundle(
            bundle.protocol,
            bundle.search_executions,
            (changed_source,),
            (span,),
            bundle.claims,
            bundle.relationships,
        )


def test_evidence_cannot_reference_metadata_only_source() -> None:
    bundle = sample_bundle()
    source = bundle.sources[0]
    metadata_only = SourceRecord(
        source_id=source.source_id,
        provider=source.provider,
        provider_record_id=source.provider_record_id,
        title=source.title,
        doi=source.doi,
        retrieved_at_utc=source.retrieved_at_utc,
    )
    with pytest.raises(ContractError, match="needs content_sha256"):
        ResearchBundle(
            bundle.protocol,
            bundle.search_executions,
            (metadata_only,),
            bundle.evidence_spans,
            bundle.claims,
            bundle.relationships,
        )


def test_rejects_duplicate_identifiers() -> None:
    bundle = sample_bundle()
    with pytest.raises(ContractError, match="duplicate source identifiers"):
        ResearchBundle(
            bundle.protocol,
            bundle.search_executions,
            (bundle.sources[0], bundle.sources[0]),
            bundle.evidence_spans,
            bundle.claims,
            bundle.relationships,
        )


def test_rejects_dangling_evidence_source_reference() -> None:
    bundle = sample_bundle()
    missing_span = EvidenceSpan(
        evidence_id="orphan",
        source_id="missing-source",
        quote="Some text",
        source_sha256=DIGEST,
        page_number=1,
    )
    with pytest.raises(ContractError, match="missing source"):
        ResearchBundle(
            bundle.protocol,
            bundle.search_executions,
            bundle.sources,
            (missing_span,),
            bundle.claims,
            (),
        )


def test_rejects_dangling_claim_or_evidence_relationship() -> None:
    bundle = sample_bundle()
    broken_link = ClaimEvidenceLink(
        claim_id="missing-claim",
        evidence_id="evidence-001",
        relation=EvidenceRelation.SUPPORTS,
        rationale="Not valid.",
    )
    with pytest.raises(ContractError, match="missing claim"):
        ResearchBundle(
            bundle.protocol,
            bundle.search_executions,
            bundle.sources,
            bundle.evidence_spans,
            bundle.claims,
            (broken_link,),
        )


def test_human_review_requires_reviewer_identity() -> None:
    with pytest.raises(ContractError, match="reviewer_id"):
        ClaimEvidenceLink(
            claim_id="c1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Reviewer confirmed the link.",
            review_status=ReviewStatus.HUMAN_REVIEWED,
        )


def test_unreviewed_link_cannot_claim_reviewer_identity() -> None:
    with pytest.raises(ContractError, match="must not carry reviewer_id"):
        ClaimEvidenceLink(
            claim_id="c1",
            evidence_id="e1",
            relation=EvidenceRelation.SUPPORTS,
            rationale="Awaiting review.",
            reviewer_id="reviewer-1",
        )


def test_from_dict_rejects_unknown_fields_and_schema_versions() -> None:
    payload = sample_bundle().to_dict()
    payload["unexpected"] = "must not be ignored"
    with pytest.raises(ContractError, match="keys mismatch"):
        ResearchBundle.from_dict(payload)

    payload = sample_bundle().to_dict()
    payload["schema_version"] = "2.0"
    with pytest.raises(ContractError, match="unsupported"):
        ResearchBundle.from_dict(payload)


def test_from_dict_rejects_unknown_nested_fields() -> None:
    payload = sample_bundle().to_dict()
    payload["protocol"]["silent_default"] = True
    with pytest.raises(ContractError, match="unknown fields"):
        ResearchBundle.from_dict(payload)


def test_json_parse_error_is_a_contract_error() -> None:
    with pytest.raises(ContractError, match="valid JSON"):
        ResearchBundle.from_json("{not-json")



def test_search_execution_distinguishes_empty_results_from_failure() -> None:
    empty = SearchExecution(
        execution_id="empty-1",
        provider="crossref",
        exact_query="a query with no matches",
        executed_at_utc="2026-11-19T10:00:00Z",
        result_count=0,
    )
    failed = SearchExecution(
        execution_id="failed-1",
        provider="openalex",
        exact_query="a query",
        executed_at_utc="2026-11-19T10:00:00Z",
        result_count=0,
        status=SearchExecutionStatus.FAILED,
        error_type="ConnectionError",
    )
    assert empty.status is SearchExecutionStatus.EMPTY
    assert failed.status is SearchExecutionStatus.FAILED
    assert empty.error_type is None


def test_search_execution_records_timing_cache_and_page_metadata() -> None:
    execution = SearchExecution(
        execution_id="search-timed",
        provider="openalex",
        exact_query="exact adapter query",
        executed_at_utc="2026-11-19T10:01:00Z",
        result_count=22,
        status=SearchExecutionStatus.SUCCEEDED,
        started_at_utc="2026-11-19T10:00:59Z",
        finished_at_utc="2026-11-19T10:01:00Z",
        elapsed_ms=1000,
        page_count=2,
        cache_hit=False,
    )
    assert execution.elapsed_ms == 1000
    assert execution.page_count == 2
    assert execution.cache_hit is False


@pytest.mark.parametrize(
    ("status", "count", "error_type"),
    [
        (SearchExecutionStatus.SUCCEEDED, 0, None),
        (SearchExecutionStatus.EMPTY, 2, None),
        (SearchExecutionStatus.FAILED, 1, "TimeoutError"),
        (SearchExecutionStatus.TIMED_OUT, 0, None),
    ],
)
def test_search_execution_rejects_inconsistent_outcome(status, count, error_type) -> None:
    with pytest.raises(ContractError):
        SearchExecution(
            execution_id="bad",
            provider="provider",
            exact_query="query",
            executed_at_utc="2026-11-19T10:00:00Z",
            result_count=count,
            status=status,
            error_type=error_type,
        )


def test_search_execution_rejects_negative_elapsed_time_or_pages() -> None:
    with pytest.raises(ContractError, match="elapsed_ms"):
        SearchExecution(
            execution_id="bad-time",
            provider="provider",
            exact_query="query",
            executed_at_utc="2026-11-19T10:00:00Z",
            result_count=1,
            elapsed_ms=-1,
        )
    with pytest.raises(ContractError, match="page_count"):
        SearchExecution(
            execution_id="bad-pages",
            provider="provider",
            exact_query="query",
            executed_at_utc="2026-11-19T10:00:00Z",
            result_count=1,
            page_count=-1,
        )


def test_search_execution_rejects_finish_before_start() -> None:
    with pytest.raises(ContractError, match="precede"):
        SearchExecution(
            execution_id="bad-order",
            provider="provider",
            exact_query="query",
            executed_at_utc="2026-11-19T10:00:00Z",
            result_count=1,
            started_at_utc="2026-11-19T10:01:00Z",
            finished_at_utc="2026-11-19T10:00:00Z",
        )


def test_search_execution_status_survives_bundle_serialization() -> None:
    bundle = sample_bundle()
    failed = SearchExecution(
        execution_id="search-failed",
        provider="s2",
        exact_query="a query",
        executed_at_utc="2026-11-19T08:36:00Z",
        result_count=0,
        status=SearchExecutionStatus.FAILED,
        error_type="HTTPError",
    )
    mixed = ResearchBundle(
        bundle.protocol,
        (*bundle.search_executions, failed),
        bundle.sources,
        bundle.evidence_spans,
        bundle.claims,
        bundle.relationships,
    )
    loaded_bundle = ResearchBundle.from_json(mixed.to_json())
    assert loaded_bundle.search_executions[1].status is SearchExecutionStatus.FAILED

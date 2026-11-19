"""Versioned, provider-neutral research domain contracts.

These models form the stable boundary between acquisition, document processing,
methodological assessment, quantitative analysis, and synthesis. They use only
the Python standard library so the integrity layer is available offline and
does not depend on an LLM or a provider SDK.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar

SCHEMA_VERSION = "1.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

T = TypeVar("T")


class ContractError(ValueError):
    """Raised when a research artifact violates a domain contract."""


class ProtocolStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REGISTERED = "registered"
    AMENDED = "amended"
    COMPLETED = "completed"


class ClaimKind(StrEnum):
    FINDING = "finding"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"
    LIMITATION = "limitation"
    RECOMMENDATION = "recommendation"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    QUALIFIES = "qualifies"
    CONTEXTUALIZES = "contextualizes"


class SearchExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    HUMAN_REVIEWED = "human_reviewed"


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _text(value, field_name)


def _timestamp(value: Any, field_name: str) -> str:
    value = _text(value, field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{field_name} must include a timezone offset")
    return value


def _sha256(value: Any, field_name: str) -> str:
    value = _text(value, field_name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return value


def _string_tuple(value: Any, field_name: str, *, minimum: int = 0) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{field_name} must be an array of strings")
    result = tuple(_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))
    if len(result) < minimum:
        raise ContractError(f"{field_name} must contain at least {minimum} item(s)")
    if len(set(result)) != len(result):
        raise ContractError(f"{field_name} must not contain duplicates")
    return result


def _enum(value: Any, enum_type: type[StrEnum], field_name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractError(f"{field_name} must be one of: {allowed}") from exc


def _parameter_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        value = sorted(value.items())
    if not isinstance(value, (list, tuple)):
        raise ContractError("parameters must be an object or an array of key/value pairs")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ContractError(f"parameters[{index}] must contain exactly two strings")
        key = _text(item[0], f"parameters[{index}].key")
        val = _text(str(item[1]), f"parameters[{index}].value")
        pairs.append((key, val))
    if len({key for key, _ in pairs}) != len(pairs):
        raise ContractError("parameters must not contain duplicate keys")
    return tuple(sorted(pairs))


@dataclass(frozen=True, slots=True)
class ResearchProtocol:
    """The planned research scope, recorded independently of its results."""

    protocol_id: str
    research_question: str
    created_at_utc: str
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    planned_sources: tuple[str, ...]
    status: ProtocolStatus = ProtocolStatus.DRAFT
    preregistration_uri: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _text(self.protocol_id, "protocol_id"))
        object.__setattr__(self, "research_question", _text(self.research_question, "research_question"))
        object.__setattr__(self, "created_at_utc", _timestamp(self.created_at_utc, "created_at_utc"))
        object.__setattr__(self, "inclusion_criteria", _string_tuple(self.inclusion_criteria, "inclusion_criteria", minimum=1))
        object.__setattr__(self, "exclusion_criteria", _string_tuple(self.exclusion_criteria, "exclusion_criteria"))
        object.__setattr__(self, "planned_sources", _string_tuple(self.planned_sources, "planned_sources", minimum=1))
        object.__setattr__(self, "status", _enum(self.status, ProtocolStatus, "status"))
        object.__setattr__(self, "preregistration_uri", _optional_text(self.preregistration_uri, "preregistration_uri"))


@dataclass(frozen=True, slots=True)
class SearchExecution:
    """An executed provider query, including empty and failed attempts.

    exact_query is the exact string passed to the provider adapter. Adapter
    implementations that transform it must record the effective expression
    in provider parameters or a future adapter-specific request field.
    error_type stores only the exception class, never arbitrary error text.
    """

    execution_id: str
    provider: str
    exact_query: str
    executed_at_utc: str
    result_count: int
    parameters: tuple[tuple[str, str], ...] = ()
    export_sha256: str | None = None
    status: SearchExecutionStatus | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    elapsed_ms: int | None = None
    error_type: str | None = None
    page_count: int | None = None
    cache_hit: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution_id", _text(self.execution_id, "execution_id"))
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "exact_query", _text(self.exact_query, "exact_query"))
        object.__setattr__(self, "executed_at_utc", _timestamp(self.executed_at_utc, "executed_at_utc"))
        if isinstance(self.result_count, bool) or not isinstance(self.result_count, int) or self.result_count < 0:
            raise ContractError("result_count must be a non-negative integer")
        object.__setattr__(self, "parameters", _parameter_pairs(self.parameters))
        if self.export_sha256 is not None:
            object.__setattr__(self, "export_sha256", _sha256(self.export_sha256, "export_sha256"))

        status = self.status
        if status is None:
            status = SearchExecutionStatus.SUCCEEDED if self.result_count else SearchExecutionStatus.EMPTY
        object.__setattr__(self, "status", _enum(status, SearchExecutionStatus, "status"))
        if self.status is SearchExecutionStatus.SUCCEEDED and self.result_count == 0:
            raise ContractError("succeeded search executions must return at least one result")
        if self.status is SearchExecutionStatus.EMPTY and self.result_count != 0:
            raise ContractError("empty search executions must have result_count=0")
        if self.status in (SearchExecutionStatus.FAILED, SearchExecutionStatus.TIMED_OUT):
            if self.result_count != 0:
                raise ContractError("failed or timed-out executions cannot claim a complete result count")
            object.__setattr__(self, "error_type", _text(self.error_type, "error_type"))
        elif self.error_type is not None:
            raise ContractError("successful or empty executions cannot carry error_type")

        for name in ("started_at_utc", "finished_at_utc"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value, name))
        if self.started_at_utc and self.finished_at_utc:
            started = datetime.fromisoformat(self.started_at_utc.replace("Z", "+00:00"))
            finished = datetime.fromisoformat(self.finished_at_utc.replace("Z", "+00:00"))
            if finished < started:
                raise ContractError("finished_at_utc cannot precede started_at_utc")

        if self.elapsed_ms is not None and (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise ContractError("elapsed_ms must be a non-negative integer")
        if self.page_count is not None and (
            isinstance(self.page_count, bool)
            or not isinstance(self.page_count, int)
            or self.page_count < 0
        ):
            raise ContractError("page_count must be a non-negative integer")
        if self.cache_hit is not None and not isinstance(self.cache_hit, bool):
            raise ContractError("cache_hit must be a boolean or null")


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Normalized bibliographic identity plus an optional exact content version."""

    source_id: str
    provider: str
    title: str
    retrieved_at_utc: str
    provider_record_id: str | None = None
    doi: str | None = None
    url: str | None = None
    publication_year: int | None = None
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "title", _text(self.title, "title"))
        object.__setattr__(self, "retrieved_at_utc", _timestamp(self.retrieved_at_utc, "retrieved_at_utc"))
        for name in ("provider_record_id", "doi", "url"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        if not any((self.provider_record_id, self.doi, self.url)):
            raise ContractError("a source requires at least one persistent locator: provider_record_id, doi, or url")
        if self.publication_year is not None and (
            isinstance(self.publication_year, bool)
            or not isinstance(self.publication_year, int)
            or not 1450 <= self.publication_year <= 9999
        ):
            raise ContractError("publication_year must be an integer between 1450 and 9999")
        if self.content_sha256 is not None:
            object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """Verbatim, locatable evidence tied to the exact source content version."""

    evidence_id: str
    source_id: str
    quote: str
    source_sha256: str
    locator: str | None = None
    page_number: int | None = None
    section: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "quote", _text(self.quote, "quote"))
        object.__setattr__(self, "source_sha256", _sha256(self.source_sha256, "source_sha256"))
        for name in ("locator", "section"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        if not any((self.locator, self.section, self.page_number is not None)):
            raise ContractError("an evidence span requires locator, section, or page_number")
        if self.page_number is not None and (
            isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or self.page_number < 1
        ):
            raise ContractError("page_number must be a positive, one-based integer")
        if (self.char_start is None) != (self.char_end is None):
            raise ContractError("char_start and char_end must be supplied together")
        if self.char_start is not None and (
            isinstance(self.char_start, bool)
            or isinstance(self.char_end, bool)
            or not isinstance(self.char_start, int)
            or not isinstance(self.char_end, int)
            or self.char_start < 0
            or self.char_end <= self.char_start
        ):
            raise ContractError("character offsets must be non-negative and end after start")
        if self.char_start is not None and self.char_end - self.char_start != len(self.quote):
            raise ContractError("character offset length must equal the verbatim quote length")


@dataclass(frozen=True, slots=True)
class ResearchClaim:
    """A proposition intended for synthesis, with epistemic type kept explicit."""

    claim_id: str
    statement: str
    kind: ClaimKind
    created_at_utc: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "statement", _text(self.statement, "statement"))
        object.__setattr__(self, "kind", _enum(self.kind, ClaimKind, "kind"))
        object.__setattr__(self, "created_at_utc", _timestamp(self.created_at_utc, "created_at_utc"))


@dataclass(frozen=True, slots=True)
class ClaimEvidenceLink:
    """An explicit relation between a claim and a specific evidence span."""

    claim_id: str
    evidence_id: str
    relation: EvidenceRelation
    rationale: str
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "relation", _enum(self.relation, EvidenceRelation, "relation"))
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))
        object.__setattr__(self, "review_status", _enum(self.review_status, ReviewStatus, "review_status"))
        object.__setattr__(self, "reviewer_id", _optional_text(self.reviewer_id, "reviewer_id"))
        if self.review_status is ReviewStatus.HUMAN_REVIEWED and not self.reviewer_id:
            raise ContractError("human_reviewed links require reviewer_id")
        if self.review_status is ReviewStatus.UNREVIEWED and self.reviewer_id:
            raise ContractError("unreviewed links must not carry reviewer_id")


@dataclass(frozen=True, slots=True)
class ResearchBundle:
    """A validated, portable evidence graph with referential integrity."""

    protocol: ResearchProtocol
    search_executions: tuple[SearchExecution, ...]
    sources: tuple[SourceRecord, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    claims: tuple[ResearchClaim, ...]
    relationships: tuple[ClaimEvidenceLink, ...]

    def __post_init__(self) -> None:
        for name, expected in (
            ("search_executions", SearchExecution),
            ("sources", SourceRecord),
            ("evidence_spans", EvidenceSpan),
            ("claims", ResearchClaim),
            ("relationships", ClaimEvidenceLink),
        ):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ContractError(f"{name} must be an array")
            values = tuple(value)
            if any(not isinstance(item, expected) for item in values):
                raise ContractError(f"{name} contains an invalid record type")
            object.__setattr__(self, name, values)
        if not isinstance(self.protocol, ResearchProtocol):
            raise ContractError("protocol must be a ResearchProtocol")

        _unique_ids(self.search_executions, "execution_id", "search execution")
        _unique_ids(self.sources, "source_id", "source")
        _unique_ids(self.evidence_spans, "evidence_id", "evidence span")
        _unique_ids(self.claims, "claim_id", "claim")

        source_by_id = {item.source_id: item for item in self.sources}
        claim_ids = {item.claim_id for item in self.claims}
        evidence_by_id = {item.evidence_id: item for item in self.evidence_spans}

        for span in self.evidence_spans:
            source = source_by_id.get(span.source_id)
            if source is None:
                raise ContractError(f"evidence span {span.evidence_id!r} references missing source {span.source_id!r}")
            if source.content_sha256 is None:
                raise ContractError(f"source {source.source_id!r} needs content_sha256 before it can ground evidence")
            if source.content_sha256 != span.source_sha256:
                raise ContractError(f"evidence span {span.evidence_id!r} does not match the source content version")

        seen_links: set[tuple[str, str, EvidenceRelation]] = set()
        for link in self.relationships:
            if link.claim_id not in claim_ids:
                raise ContractError(f"relationship references missing claim {link.claim_id!r}")
            if link.evidence_id not in evidence_by_id:
                raise ContractError(f"relationship references missing evidence span {link.evidence_id!r}")
            key = (link.claim_id, link.evidence_id, link.relation)
            if key in seen_links:
                raise ContractError("duplicate claim/evidence relationship")
            seen_links.add(key)

    @property
    def unlinked_claim_ids(self) -> tuple[str, ...]:
        """Claims with no explicit evidence relationship; useful for report QA."""
        linked = {link.claim_id for link in self.relationships}
        return tuple(claim.claim_id for claim in self.claims if claim.claim_id not in linked)

    @property
    def unlinked_evidence_ids(self) -> tuple[str, ...]:
        """Evidence spans not yet used by a claim; they are not discarded."""
        linked = {link.evidence_id for link in self.relationships}
        return tuple(span.evidence_id for span in self.evidence_spans if span.evidence_id not in linked)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation with an explicit version."""
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": _to_primitive(self.protocol),
            "search_executions": _to_primitive(self.search_executions),
            "sources": _to_primitive(self.sources),
            "evidence_spans": _to_primitive(self.evidence_spans),
            "claims": _to_primitive(self.claims),
            "relationships": _to_primitive(self.relationships),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically; indent=None returns canonical JSON."""
        options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
        if indent is None:
            options["separators"] = (",", ":")
        else:
            options["indent"] = indent
        return json.dumps(self.to_dict(), **options)

    @property
    def canonical_sha256(self) -> str:
        """SHA-256 of the canonical serialized bundle."""
        return hashlib.sha256(self.to_json(indent=None).encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, content: str) -> ResearchBundle:
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractError("research bundle must be valid JSON") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Any) -> ResearchBundle:
        """Load and validate a bundle; unknown fields and schema versions fail."""
        if not isinstance(payload, Mapping):
            raise ContractError("research bundle root must be an object")
        expected = {
            "schema_version",
            "protocol",
            "search_executions",
            "sources",
            "evidence_spans",
            "claims",
            "relationships",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown or missing:
            raise ContractError(f"research bundle keys mismatch (missing={sorted(missing)}, unknown={sorted(unknown)})")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"unsupported research bundle schema_version: {payload['schema_version']!r}")

        protocol = _construct(ResearchProtocol, payload["protocol"], enum_fields={"status": ProtocolStatus}, tuple_fields={"inclusion_criteria", "exclusion_criteria", "planned_sources"})
        searches = _construct_many(SearchExecution, payload["search_executions"], enum_fields={"status": SearchExecutionStatus}, tuple_fields={"parameters"})
        sources = _construct_many(SourceRecord, payload["sources"])
        spans = _construct_many(EvidenceSpan, payload["evidence_spans"])
        claims = _construct_many(ResearchClaim, payload["claims"], enum_fields={"kind": ClaimKind})
        links = _construct_many(ClaimEvidenceLink, payload["relationships"], enum_fields={"relation": EvidenceRelation, "review_status": ReviewStatus})
        return cls(protocol, searches, sources, spans, claims, links)


def _unique_ids(items: tuple[Any, ...], attribute: str, label: str) -> None:
    values = [getattr(item, attribute) for item in items]
    if len(values) != len(set(values)):
        raise ContractError(f"duplicate {label} identifiers")


def _construct_many(
    model_type: type[T],
    values: Any,
    *,
    enum_fields: dict[str, type[StrEnum]] | None = None,
    tuple_fields: set[str] | None = None,
) -> tuple[T, ...]:
    if not isinstance(values, (list, tuple)):
        raise ContractError(f"{model_type.__name__} collection must be an array")
    return tuple(
        _construct(model_type, value, enum_fields=enum_fields, tuple_fields=tuple_fields)
        for value in values
    )


def _construct(
    model_type: type[T],
    value: Any,
    *,
    enum_fields: dict[str, type[StrEnum]] | None = None,
    tuple_fields: set[str] | None = None,
) -> T:
    if not isinstance(value, Mapping):
        raise ContractError(f"{model_type.__name__} must be an object")
    model_fields = {item.name: item for item in fields(model_type)}
    unknown = set(value) - set(model_fields)
    if unknown:
        raise ContractError(f"{model_type.__name__} has unknown fields: {sorted(unknown)}")
    enum_fields = enum_fields or {}
    tuple_fields = tuple_fields or set()
    kwargs: dict[str, Any] = {}
    for name, field in model_fields.items():
        if name not in value:
            if field.default is MISSING and field.default_factory is MISSING:
                raise ContractError(f"{model_type.__name__} is missing required field {name!r}")
            continue
        item = value[name]
        if name in enum_fields:
            item = _enum(item, enum_fields[name], name)
        elif name in tuple_fields:
            if name == "parameters":
                item = _parameter_pairs(item)
            else:
                item = _string_tuple(item, name)
        kwargs[name] = item
    try:
        return model_type(**kwargs)
    except TypeError as exc:
        raise ContractError(f"invalid {model_type.__name__}: {exc}") from exc


def _to_primitive(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_to_primitive(item) for item in value]
    if isinstance(value, list):
        return [_to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_primitive(item) for key, item in value.items()}
    return value

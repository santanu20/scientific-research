# ADR 0004: Persist provider-adapter search execution records

- Status: Accepted incrementally; Phase 1 only
- Date: 2026-11-19

## Context

The discovery pipeline returned a merged list of paper records. This did not retain a machine-readable record for each source/query attempt, including empty returns, exceptions, timeouts, or elapsed time. A paper corpus alone cannot reconstruct the searches that produced it.

## Decision

Add an optional execution-record collector to the multi-source discovery function. The established list return type remains unchanged. When a collector is supplied, each source adapter attempt records its adapter-level query string, provider name, returned record count, outcome, UTC start/finish times, elapsed milliseconds, maximum-results/filter parameters, and exception class on raised errors.

The pipeline's Phase 1 primary, typo-repair rediscovery, and zero-term recall searches pass a shared collector and atomically persist it to search_executions.json. The artifact is also surfaced in result paths and the data-availability section.

## Semantics and limitations

- result_count is the number of records returned by the adapter, not a provider-reported total hit count.
- exact_query is the query string passed to the adapter. A provider adapter may further transform it; e.g. arXiv category augmentation must be surfaced by a later adapter-level enhancement before this can be called a wire-exact query.
- EMPTY means the adapter returned no records. Some pre-existing wrappers intentionally swallow internal errors and return an empty list; those attempts cannot yet be distinguished from a true zero-result response.
- FAILED captures exceptions that escape the adapter; TIMED_OUT captures futures still running at the orchestrator timeout.
- Error messages are intentionally excluded to avoid persisting arbitrary provider text or secrets; only exception class is retained.
- This change covers Phase 1 search strategies, typo-repair and zero-term recall. Later reflective-search and direct CLI discovery paths require separate wiring.

## Consequences

The pipeline begins retaining search-attempt provenance without changing existing callers. Full PRISMA-S fidelity still requires provider-specific request/response metadata, provider totals, pagination, exact wire queries, export snapshots, limits, and structured soft-failure reporting.

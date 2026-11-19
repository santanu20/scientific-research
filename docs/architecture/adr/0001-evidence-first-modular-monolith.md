# ADR 0001: Evidence-first modular monolith

- **Status:** Accepted as the target direction; implementation is incremental
- **Date:** 2026-11-19
- **Decision owners:** Repository maintainers

## Context

The repository already provides multi-provider discovery, identifier verification, screening, document processing, extraction, quantitative tooling, claim synthesis, provenance, and a CLI. These capabilities are useful, but their interfaces are predominantly script-oriented JSON artifacts. A sequence of scripts does not by itself guarantee that a final claim can be traced to the exact source version and passage that supports or contradicts it.

The system must improve research rigor without requiring a wholesale rewrite, cloud deployment, or paid infrastructure. Existing users and agent workflows need a compatibility path.

## Decision

Adopt an evidence-first modular monolith with explicit domain boundaries and versioned, provider-neutral contracts.

The core dependency direction is:

```text
CLI / Agent interface
        |
Application orchestration
        |
Domain contracts and policies
        |
Ports (discovery, storage, models, document processing)
        |
Adapters (Crossref, OpenAlex, S2, arXiv, web, local files, Ollama)
```

Domain code must not import provider SDKs. Adapters translate provider responses to domain records. Orchestration coordinates stages but does not redefine scientific rules. Model-generated content is untrusted until it passes structural and evidence-grounding validation.

The initial storage implementation remains filesystem-based and local-first. Domain IDs and contracts must remain storage-agnostic.

## Non-negotiable invariants

1. A research protocol is recorded separately from observed results.
2. Source identity and source content version are distinct.
3. Evidence spans carry source-content hashes and human-auditable locators.
4. Claims and evidence are joined by explicit typed relationships.
5. Search and transformation steps produce provenance records.
6. Human review state cannot be inferred from model confidence.
7. Scientific appraisal and quantitative synthesis reject unsupported or incompatible inputs rather than silently inventing values.
8. Legacy artifacts are migrated through explicit adapters; they are not silently reinterpreted as fully traceable evidence.

## Alternatives considered

### Rewrite as microservices
Rejected. Current workload and deployment model do not justify service orchestration, network failure modes, distributed transactions, or operational cost. Stable module boundaries provide a future extraction path.

### Keep all scripts independent
Rejected as the long-term architecture. It makes cross-stage invariants, schema evolution, and consistent provenance difficult to enforce. Scripts remain as compatibility entry points during migration.

### Adopt a database immediately
Deferred. The current local-first filesystem workflow is adequate as a starting point. A database decision requires measured needs for concurrent reviewers, corpus size, query latency, and transactional semantics.

### Make the LLM the central orchestrator and evidence judge
Rejected. The system must remain useful without a model and must distinguish extraction proposals from validated evidence and human judgments.

## Consequences

**Positive:** stable interfaces, testable boundaries, reproducible evidence lineage, replaceable providers/models, and incremental migration without breaking the current CLI.

**Costs:** temporary duplication between legacy and canonical records; adapters and schema migration tests; stricter validation may expose previously tolerated incomplete data.

**Risks:** the new model can become another disconnected artifact if not integrated. Subsequent PRs must connect adapters to the core contracts and must not claim system-wide adoption until that integration is complete.

## Implementation sequence

See [the architecture migration plan](../README.md#migration-sequence). The first implementation slice introduces versioned protocol, search-execution, source, evidence-span, claim, and relationship contracts; later slices integrate them into the current pipeline.

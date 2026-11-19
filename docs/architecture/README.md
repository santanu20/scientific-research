# Research platform architecture

## Purpose

Move the project from a sequence of research scripts to an evidence-first research system while preserving the existing CLI, local-first operating model, and agent-skill interface. The architecture must make a research result reproducible, inspectable, and appropriately cautious—not merely fluent.

## Target architecture

```text
Research protocol
  question · objectives · eligibility · planned sources · analysis plan
        |
        v
Acquisition adapters
  scholarly APIs · web search · repositories · local documents
        |
        v
Canonical source registry + search execution log
  stable IDs · identifiers · timestamps · exact queries · checksums
        |
        v
Document processing
  retrieval · parsing · normalization · page/section/character locators
        |
        v
Evidence layer
  immutable source versions · verbatim evidence spans · extraction provenance
        |
        +---------------------+
        |                     |
        v                     v
Methodological appraisal   Quantitative analysis
  design-aware tools         compatible estimands · effects · uncertainty
  reviewer judgments         sensitivity · heterogeneity
        |                     |
        +----------+----------+
                   |
                   v
Claim–evidence graph
  claim nodes · supports/refutes/qualifies links · review state
                   |
                   v
Synthesis and export
  cited narrative · evidence tables · reproducible supplements · limitations
```

Cross-cutting controls apply to every stage: versioned contracts, provenance, deterministic validation, human review, security boundaries, evaluation, and observable execution.

## Architectural decisions

1. **Modular monolith, not premature microservices.** The CLI and local workflow remain the initial deployment. Domain boundaries and ports/adapters make later service extraction possible without imposing network infrastructure now.
2. **Evidence is a first-class domain object.** A claim is not considered traceable merely because it names a paper. It must be linked to a versioned source and a verbatim span with a stable locator.
3. **Protocols precede execution.** The research question, inclusion/exclusion criteria, planned sources, and intended synthesis are recorded independently of the results. Amendments should be explicit and timestamped.
4. **Adapters isolate external providers.** Crossref, OpenAlex, Semantic Scholar, arXiv, web search, and local files must normalize into provider-neutral contracts. Provider-specific fields remain available as raw metadata.
5. **LLMs are fallible processors, not authorities.** They may propose search queries, extractions, or claim groupings. Schema validation, source grounding, methodological rules, and human adjudication remain separate controls.
6. **No unsupported certainty.** A missing locator, absent full text, unreviewed judgment, incompatible quantitative outcome, or inconclusive evidence must remain explicit in outputs.
7. **Local-first persistence.** Continue using files and atomic writes initially. Introduce a database only when measured concurrency, query, or corpus-size requirements justify it. Contracts must not depend on a particular storage backend.
8. **Versioned artifacts are public interfaces.** JSON outputs require schema versions, migration rules, and compatibility tests. Existing artifacts remain readable through adapters during migration.
9. **Deterministic core, optional intelligence.** Parsing, identifiers, integrity checks, calculations, and export validation must work without an LLM. Model providers are replaceable adapters.
10. **Security is part of ingestion.** Web pages, PDFs, metadata, and model output are untrusted inputs. Enforce size/time limits, safe file handling, bounded network access, and strict output validation at the relevant boundaries.

## Domain boundaries

| Boundary | Owns | Must not own |
|---|---|---|
| Protocol | Research question, scope, eligibility, planned methods | Search results or conclusions |
| Discovery | Provider queries and retrieval execution records | Eligibility decisions |
| Source registry | Normalized bibliographic identity and source versions | Scientific interpretation |
| Document processing | Extracted text, layout, locators, content hashes | Claim truth |
| Evidence | Verbatim spans and extraction provenance | Overall certainty judgments |
| Screening | Per-record decisions, reasons, reviewer history | Source identity mutation |
| Appraisal | Study-design-aware methodological judgments | Automatic GRADE claims from one paper |
| Quantitative analysis | Typed estimands, input compatibility, calculations | Silent mixing of incompatible effects |
| Claim graph | Claims and explicit evidence relationships | Fabricated support or implicit confidence |
| Synthesis | Evidence-grounded narrative and limitations | Unlinked factual claims |
| Evaluation | Held-out datasets, metrics, calibration and regression thresholds | Self-reported quality claims |

## Trust and quality invariants

- Every source record has a stable internal ID and at least one persistent external locator.
- Every evidence span identifies the exact source version by SHA-256 and has a page, section, or equivalent locator.
- A claim-to-evidence edge states whether the span supports, refutes, qualifies, or contextualizes the claim.
- Human review is represented as data; a model-generated judgment is never labelled human-reviewed.
- Search executions preserve the exact query, provider, timestamp, parameters, result count, and export hash when an export exists.
- Meta-analysis rejects incompatible populations, interventions/exposures, outcomes, timepoints, units, or effect measures unless a declared conversion/stratification rule applies.
- Reports disclose missing full text, unreviewed decisions, exclusions, unresolved contradictions, and unavailable calculations.
- Quality claims are backed by versioned, held-out benchmarks with metrics and confidence intervals.

## Migration sequence

### Stage 0 — Establish contracts and decision records
Introduce the architecture decision record and versioned protocol/evidence bundle contracts. Keep existing scripts unchanged.

### Stage 1 — Add compatibility adapters
Map the current corpus, verification, extraction, screening, and synthesis JSON artifacts to the canonical records. Preserve original payloads and record lossy mappings explicitly.

### Stage 2 — Capture reproducible acquisition and source versions
Instrument provider adapters with exact search execution records; persist raw response/export hashes and normalized source identities.

### Stage 3 — Ground extraction and claims
Add page/section/span locators to extraction. Build claim-evidence links from extracted evidence. Require abstention when the source cannot be located.

### Stage 4 — Independent review and methodology
Persist reviewer decisions and adjudication history. Integrate design-appropriate appraisal instruments as separate, versioned assessments; do not collapse them into a heuristic score.

### Stage 5 — Validated quantitative and synthesis outputs
Enforce analysis compatibility, add numerical parity/golden tests, and render structured reports from validated claims and evidence.

### Stage 6 — Evaluation and operational maturity
Publish held-out screening/extraction benchmarks, supply-chain/security controls, reproducibility receipts, and release criteria.

## Pull request policy

Each migration stage should be split into small PRs that:
- state the contract or invariant being introduced;
- include positive, negative, and regression tests;
- preserve compatibility or provide a migration adapter;
- do not silently weaken lint, typing, or CI gates;
- report what has not yet been implemented.

The first implementation PRs intentionally add a new core boundary without rewriting the existing pipeline. Integration follows only after contracts and compatibility mappings are proven.

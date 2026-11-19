# ADR 0006: Verbatim source-grounded evidence spans

- Status: Accepted as a parser-independent evidence boundary
- Date: 2026-11-19

## Context

A paper-level citation does not establish that a quotation supports a research claim. Evidence must be located in the exact document version from which it was extracted. Paraphrases and ambiguous quotations must not be stored as verbatim evidence.

## Decision

Add a parser-independent helper that receives exact source bytes, their page-text extraction, and a proposed quotation. It hashes the original bytes, requires the quote to occur verbatim and unambiguously, records a one-based page and zero-based page-local offsets, and fails closed for missing or ambiguous matches. It does not normalize whitespace or use an LLM to repair text.

The caller must ensure supplied page text was extracted from the exact bytes being hashed. The helper creates an EvidenceSpan compatible with ResearchBundle source-version validation.

## Non-goals and limitations

This PR does not implement PDF parsing, OCR, section detection, candidate-quote extraction, or claim-entailment assessment. Character offsets are page-local, not document-global. Parser version/layout provenance must be captured by a later document-processing adapter.

## Consequences

Extraction adapters gain a deterministic way to reject unsupported quotes and tie accepted evidence to an immutable source version. Evidence grounding can be tested independently of a particular parser or LLM provider.

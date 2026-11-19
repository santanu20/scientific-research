# ADR 0003: Independent screening and adjudication

- Status: Accepted as a domain contract; legacy CLI integration is separate
- Date: 2026-11-19

## Context

The current screening path can emit include/exclude suggestions from deterministic content-type filters or a local model. Those outputs are useful triage signals, but they do not constitute independent dual-review screening or adjudication.

A research-grade systematic review must distinguish a machine suggestion from a human decision, preserve reviewer-level decisions, and retain unresolved conflicts rather than silently converting them to consensus.

## Decision

Introduce provider-neutral screening review records with screening stage, outcome, actor and origin, timestamp, rationale, exclusion reason, and eligibility criterion references. Full-text decisions identify the exact reviewed source content by SHA-256. Model-originated suggestions preserve model name, version, and prompt hash.

Adjudication can use only decisions from at least two distinct human reviewers on the same source and stage. Model/rule suggestions cannot count as independent reviewer votes. A disagreeing pair cannot be labelled consensus; it must be adjudicated by an independent third reviewer or remain unresolved.

## Consequences

The contract enables dual-screening metrics, conflict tracking, reviewer agreement analysis, and audit-ready screening exports. Existing screening decisions remain unchanged until a compatibility adapter and user-facing workflow are implemented. This PR does not claim that the CLI already performs dual independent screening.

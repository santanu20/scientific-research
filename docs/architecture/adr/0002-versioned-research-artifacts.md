# ADR 0002: Versioned research artifacts

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Existing stage outputs are JSON documents whose shapes are validated in places, but their semantics and cross-document references are not one common, versioned interface. Shape validation alone cannot prove that a citation points to a source version or that a claim is connected to a specific passage.

## Decision

New cross-stage research artifacts use explicit schema versions and provider-neutral records. The first canonical envelope is a research bundle containing:

- one research protocol;
- search execution records, preserving exact query strings and execution metadata;
- normalized source records;
- evidence spans tied to a source content SHA-256;
- typed claims;
- explicit claim/evidence relationships with review state.

The runtime model rejects duplicate identifiers, dangling references, invalid checksums, invalid timestamps, and human-review states without a reviewer identity.

## Compatibility

Current `corpus.json`, `verified.json`, `extracted.json`, `correlation.json`, and prose outputs remain unchanged in the first slice. Compatibility adapters will be added before any existing stage is switched to the canonical envelope. A version change requires a schema migration and round-trip fixtures.

## Consequences

Consumers can validate research bundles independently of provider libraries or LLM availability. The first schema does not claim that legacy pipeline outputs already satisfy it. Locators and source hashes are mandatory for evidence spans; metadata-only sources may exist without an evidence span until full-text content is available.

# Aspect Matrix — Stage 1 (2026-08-21, scoped run)

Universe = DISCOVERED aspects for scope (selection path + output structure + orchestration). Full-project D8 unit census deferred to unscoped sweep (declared, not silent).

| Aspect | Where | Risk hypothesis | Pass | Status |
|---|---|---|---|---|
| A1 Ranking pipeline | _ranking.py:633 rank_papers | domain multiplier ungated; weight imbalance | P9/P13 | FINDINGS F1,F3 |
| A2 Intent/topic layer | _intent.py, _topic_templates.py | geo-only registry = dead layer for most fields | P1/D2 | FINDING F2 |
| A3 Discovery filter chain | discover.py main 1601-1964 | order wastes calls; coarse gates | P1/P4 | FINDING R1 |
| A4 Source adapters | _sources.py (+epmc/usgs/eartharxiv) | field maps, rate limits — prior rounds audited | D4 | clean this scope (prior audits) |
| A5 Web-search supplement | discover.py:544-660 | DOI regex noise; email regression | P10/P7 | FINDING F4 |
| A6 Orchestrator | pipeline.py:64 run_pipeline | screen stage absent | P1 | FINDING F5 |
| A7 Synthesis assembly | synthesize.py:688-985 | fixed sections, geo enrichers, silent skips | P5/P12 | FINDINGS F6,F9 |
| A8 Research-type detection | _classifiers.detect_research_type | keyword brittleness, catch-all default | P9 | FINDING F6 |
| A9 Verification gate | verify.py | §H20 multi-resolver + RW — prior rounds green | P9 | clean (out of scope today) |
| A10 Meta-analysis math | _stats.py | golden-parity vs metafor pinned | P9 | clean (pinned goldens) |
| A11 Docs | SKILL.md | corruption + table breakage + geo-vs-field-agnostic claim | P15/D12 | FINDINGS F7,F8 |
| A12 Env/gates | .venv, ruff.toml, pytest.ini | pyright absent; work-tree dirty | S2/S4/G1 | FINDING F10 |

DISCOVERED-LATE: none beyond matrix (open-world scan of scoped surfaces complete; usgs_search/eartharxiv_search noted under A4, geo-gated adapters).

## Generated passes (D10)
- GP1 "non-geo query as first-class input" — charter: every selection stage must behave correctly for a query outside earth science. → surfaced F1,F2,F3,F6. 
- GP2 "hostile snippet ingress" — charter: web supplement must not ingest citation-noise DOIs. → surfaced F4.

## Pass ledger (scoped run, all tiers T1-T4 reported)
- P1 flow trace: 6 surfaces traced (discover CLI, pipeline orchestrator, web-supplement, verify, synthesize, export) — findings F4,F5,R1
- P2 state matrix: N/A this scope (no UI state); manifest TTL paths prior-audited
- P4 edge sequences: sparse-corpus auto-supplement traced (pipeline.py:180-202) — works, feeds F4
- P5 error paths: synthesize ImportError-pass sites — F9; search_multi_source pool.shutdown audited clean
- P7 config symmetry: mailto regression — F4; env table otherwise matches code (A-round verified)
- P9 domain logic: ranking math + domain_score semantics — F1,F3
- P10 security/trust boundary: DOI ingestion validation — F4
- P12 output structure: brief assembly — F6
- P13 architecture: discipline coupling — F1,F2,F5
- P15 docs/repo: F7,F8,F10
- P14 tests: suite green (283) but no non-geo e2e golden exists — gap recorded as F2 test evidence; mutation spot-check deferred to fix round (will pin F1 with non-geo corpus golden)
- G1 lint: ruff 0 errors (receipt). pyright NOT RUN (missing from venv — F10)
- G2 full offline suite: 283 passed / 1 skipped / 9 deselected (receipt)
- G4 smoke-run: deferred to fix round (needs live APIs; offline repro of F1 constructed via unit-level reasoning + existing geo fixtures)
- G5 secrets: no new secrets observed in scoped diff surfaces; full gitleaks sweep deferred to unscoped round

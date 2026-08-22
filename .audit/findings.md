# Findings — Stage 1 (2026-08-21, scope: selection quality + output structure + architecture)

Severity order: CRIT > HIGH > MED > LOW. All file:line verified this session.

| ID | Sev | Scale | Where | Flaw | User symptom |
|----|-----|-------|-------|------|--------------|
| F1 | CRIT | MACRO | _ranking.py:675-700 + _ontology.py:408,465 | `rank_papers` multiplies geology-only `domain_score()` into combined score UNCONDITIONALLY (comment admits "boost geo papers, kill non-geo"); no query-domain gate unlike `_is_domain_relevant` (discover.py:1058 self-gates) | Non-earth-science queries: legit papers penalized ("polymer" −0.5, physics terms −0.6..−0.8), geo-flavored strays boosted → garbage selection / corpus distortion |
| F2 | CRIT | MACRO | _topic_templates.py:32-1605 (31/31 geo templates); _intent.py:1391,1440 | Entire intent layer (landmarks/must_have/exclude/report_sections) registered for earth-science only; zero templates for bio/med/CS/physics/chem/social | Non-geo queries: intent_filter/intent_boost no-op → peripheral papers unfiltered; brief loses concept-based section structure |
| F3 | HIGH | MESO | _ranking.py:40-48,664-671,128-134,161 | semantic weight 0.30 vs popularity 0.60 (citation+network+venue); sims relative max-normalization flattens semantic spread | Famous off-topic paper outranks on-topic obscure one on niche queries |
| F4 | HIGH | MESO | discover.py:628-645 (also 464,767) | web_search supplement harvests ANY `10.xxxx/...` substring from snippets/URLs (incl. cited refs of result pages) → citation-noise DOIs resolved into corpus; `mailto="geokit@dev"` hardcoded ×3 bypasses A9 unified `_EMAIL` identity | Web-supplement injects tangential papers; polite-pool regression |
| F5 | MED | MACRO | pipeline.py (zero `screen` refs) | screen.py/screen_llm.py not wired into default pipeline — no PRISMA triage stage; only coarse filters (abstract>50ch, geo-gated domain, IDF co-occurrence) run before ranking | Off-topic survivors reach extraction/synthesis unvetted |
| F6 | MED | T3/MESO | synthesize.py:830-985 (_geo_enrich imports), _classifiers.detect_research_type | Brief enrichers (convergence/gaps) live in `_geo_enrich`; research-type detection = English keyword cues with default=subtopic catch-all; fixed section order regardless of topic/corpus shape | Output not dynamic per topic; non-geo briefs get generic/degraded sections |
| F7 | MED | MICRO | SKILL.md:147-148 | Dangling corrupted sentence: "…across all 5 LLM-using scripts (`_intent.py`, changes needed — pure env-var configuration." | Doc lies/confuses |
| F8 | LOW | MICRO | SKILL.md:553-559 | Key-flags table malformed: rows switch to 3-cell Module/Purpose/Usage schema under Flag/Script/Purpose header mid-table | Doc rendering broken |
| F9 | LOW | MICRO | synthesize.py gap/convergence blocks | `except ImportError: pass` silent section drops (H1/H2 smell) | Sections vanish without trace when module missing |
| F10 | INFO | META | .venv vs _bootstrap._REQ_INSTALLS; git work-tree | pyright missing from venv (pinned gate unrunnable); 2026-08-21 port round uncommitted (M scripts/_artifact.py et al.) | Gate drift; S4 noted, untouched |

## Reorder candidate (user item 1)
R1 | LOW-MED | discover.main(): abstract filter runs AFTER snowball+expansion merge → title-only seeds get snowballed/expansion-searched wastefully. Moving require_abstract before snowball saves calls and reduces reference-graph noise. Behavior-preserving reorder, needs e2e re-run to confirm corpus parity.

## Witness passes
- F1 witnessed by P9 (domain logic), P13 (architecture), D11-SABOTEUR (non-geo query as hostile input).
- F2 witnessed by P1 (flow trace), D2 (domain model), D12 (SKILL.md claims field-agnostic vs code geo-only).
- F4 witnessed by P10 (input validation at trust boundary), D11-INCOMPETENT (garbage snippet input).

## SKILL.md contradiction (D12)
SKILL.md line 7 claims "field-agnostic PICO extraction"; selection layer is earth-science-hardcoded (F1/F2). Contradiction = doc vs behavior.

# STAGE 2 FIX LEDGER (2026-08-22 round)

| ID | Status | Proof |
|---|---|---|
| F1 | FIXED (dynamic deletion) | rank_papers has no domain multiplier; _disciplines.py deleted; pins test_dynamic_ranking.py ×6 |
| F2 | FIXED (dynamic gate) | _is_domain_relevant = query-overlap; geo tables removed from discover; workflow_audit rewritten |
| F3 | FIXED | WEIGHTS sem .40+facet .20; absolute sims; relevance floor — pins green |
| F4 | FIXED | _clean_harvested_doi + _query_overlap_ok; mailto=_EMAIL ×3; pins test_web_ingest.py |
| F5 | FIXED | screen stage wired pipeline.py (--screen config, sparse guard <3) |
| F6 | PARTIAL→CORE DONE | dynamic theme labels + honest gaps (static mechanism deleted); residual: _METHOD_PATTERNS domain names (polish, documented) |
| F7/F8 | FIXED | SKILL.md sentence + Module-reference table split + screen flag row |
| F9 | FIXED (scope) | convergence empty-heading suppressed; ImportError-pass sites in gaps path deleted with mechanism |
| NEW H20 hole | FIXED | verification_status stamp + _extract_papers gate + 2 pins |
| NEW logging bug | FIXED | basicConfig force=True (-v was dead) |
| R1 | FIXED | early abstract filter before snowball |
| B3 context module | DEFERRED (evidence-driven) | selection fixed dynamically; revisit on multi-facet recall evidence |

Fixpoint status: sweep-2 of iteration loop complete (briefs v5→v7 clean of junk labels/gaps/collisions). Suite 294 green. pyright 115/0 baseline.

# SWEEP ROUND (2026-08-22 continued)
| Item | Status |
|---|---|
| 25-topic full-chain validation | DONE — 25/25 briefs, 0 misalignment, junk-grep clean |
| Polysemy leak (catfish/morphometric) | FIXED — median sem floor + exclusion ≥3 strong |
| Small-corpus ranking bypass | FIXED — rank always-on |
| Co-occurrence phantom-mass v2 | FIXED — coverage 45% + ≥2 terms + head-term rule |
| Basaltic/basalt morphology miss | FIXED |
| Cross-journal repost dup (Kharif pair) | FIXED — exact+prefix-window merges, specificity-gated |
| Process-exit hang (OpenAlex Retry-After) | FIXED — pyalex session retry cap |
| Berra dedup pins (3) | PRESERVED via specificity gate; 296 green |

# DE-HARDODING SWEEP + LIVE QC (2026-08-22 final)
| Item | Status |
|---|---|
| A _SYNONYM_GROUPS removal | FIXED — identity expansion; 296 green |
| B topic-template layer deletion | FIXED — 2 modules deleted, 3 consumers stripped, zero test pins broken |
| C geo-gated discovery expansion | FIXED — identity fn; hidden source-injection removed (H4) |
| D _METHOD_PATTERNS genericization | FIXED — 25→6 study-design entries; content-cluster fallback keeps narrative pin green |
| E dead static blocks | DELETED — _ontology.py, _geodict.py, geo theme vocab (~150 entries); facies tables restored from HEAD as live deps |
| Small-pool cooccurrence loophole | FIXED live — df==1 dropped when ≥2 multi-confirmed exist (Kharif regression killed) |
| Live validation | DONE — 4 topics force-refresh full chain; QC table 25/25 clean |
| pyright | 100/0 (improved from 115 baseline) |

# GEOKIT SYNC ROUND (2026-08-22)
| Item | Status |
|---|---|
| Skill commits | 7 total, worktree clean |
| Geokit capture commit | 25757222 (their r2 preserved) |
| Geokit sync commit | G2 overlay + bootstrap-fix follow-up |
| Deleted-module importers in geokit | zero (verified pre-rm) |
| Live validation from geokit venv | p2 15/15, g15 11/11 — briefs QC clean |
| Reverse-port | _context.py + _honesty.py vendored unwired |

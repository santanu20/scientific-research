# .audit/pass-ledger.md — Sweep 1 (2026-08-16), prompt-version 3

Format: pass × tier T1-T4 → findings count / clean / NOT RUN.

| Pass | T1 | T2 | T3 | T4 | Verdict |
|------|----|----|----|----|---------|
| P1 flow trace (16 CLIs --help ✓, screen CLI repro, KB write path, synthesize --use-llm → smooth_theme_paragraphs ✓) | A1 | A2 | A2 | — | 2 findings |
| P2 state matrix (module globals census, function-attr state, cache singletons _geo_dict/_geo_journals/_stance_svm model) | A12 | A11 | clean | — | 1 finding (A11/A12 witness) |
| P3 signal/event closure (orphan census 558 defs; 31 orphans rg-verified) | A5 | A5 | A5 | — | 1 finding |
| P4 edge sequences (non-atomic writes = mid-write crash; first-run bootstrap ✓ self-repair per MEMORY+tests; empty corpus handled fail-loud per R1 fix + tests) | A6 | clean | — | — | 1 finding |
| P5 error paths (265 try/except AST-classified: 47 swallow-pass / 57 log-continue / 152 mixed / 11 re-raise; 47 sites context-triaged; 25 except-clauses extracted) | A12 | A3 | A3 | — | 1 finding (A3 cluster) |
| P6 cross-module contracts (pyright arg-type triage, PaperRecord field adds, artifact loaders wired correlate/synthesize/benchmark/_documentstore) | A11 | A11 | clean | — | 1 finding |
| P7 persistence/config symmetry (KB get/put key hash: get has filters_hash param, put missing = A2; verify_cache_put healthy (repro); env read=27/write=0; TTL 30d/90d honored in code) | A2 | A9 | A7 | A7 | 3 findings |
| P8 thread/lifecycle/resources (CircuitBreaker locked ✓, pace sleeps outside lock ✓, ThreadPoolExecutor: 3 with-managed + 1 deliberate documented no-with + finally shutdown(wait=False) discover.py:1320-1350 ✓, subprocess timeouts present ×3) | clean | clean | clean | — | 0 findings — checked clean |
| P9 domain logic (goldens trusted: REML/PM/HKSJ/κ vs sklearn/metafor per test_p0_*; unit SI factors mutation CAUGHT; DL c-constant mutation NOT caught = A4) | A4 | clean | — | — | 1 finding |
| P10 security (no eval/exec/shell=True/yaml.load ✓; pickle = self-owned model cache only _stance_svm.py:180,198 (local trust, LOW); no secrets in repo (G5 scan clean); S2_API_KEY env ✓; subprocess list-form ✓; 32 urlopen = fixed hosts + quoted params) | clean | A9 | clean | clean | 1 finding (A9 witness) |
| P11 performance | NOT RUN (profiling; benchmark.py receipts historical) — declared |
| P12 UX (CLI-only skill; 16/16 --help load; 2 blank-first-line cosmetic A12; no GUI) | A12 | — | — | — | 1 witness |
| P13 DRY/architecture (orphan dup smooth_with_llm vs smooth_theme_paragraphs; save_artifact vs 15 inline writes; 3 dep lists) | A5 | A6 | A5 | A8 | 3 witnesses |
| P14 tests (suite 277P/1S/9D green 2.84s; mutations M1-M5: 2 caught (units, citation-align), 3 NOT caught; A1/A2 uncaught by suite) | A4 | A4 | — | A4 | 1 finding |
| P15 docs/repo hygiene (README vs SKILL vs bootstrap drift; 6 undocumented env vars; .gitignore ✓ covers venv/caches/outputs; no CI file exists — repo is a skill, CI absence noted not penalized; no CHANGELOG/version — skill not versioned, noted) | A8 | A7 | A10 | A8,A10 | 3 findings |
| P16 reproducibility (benchmarks seeded 7/42 ✓; sklearn deterministic ✓; tool versions UNPINNED; lint gate irreproducible) | clean | clean | A10 | A10 | 1 finding |

## Generated passes (D10)
| GP | Charter | Result |
|----|---------|--------|
| GP1 KB manifest get/put symmetry | search manifest write path must work | A2 CRIT found |
| GP2 mutation hardening | break risky paths, suite must catch | A4 (3 blind zones) |
| GP3 timeout centralization | _timeouts docstring contract | A5 (7 dead props + 20+ inline) |
| GP4 polite-pool identity | single contact identity | A9 |
| GP5 dead-wiring census | no orphan public API | A5 (31 orphans) |
| GP6 doc/env contract | env + deps documented | A7, A8 |

## Mutation receipts (applied→tested→reverted, all files restored)
- M1 _stats.py:191 c-constant /2w: applied, 26 passed → NOT CAUGHT
- M2 _units.py GPa 10.0→11.0: applied, FAILED test_mixed_pressure_units_single_pool → CAUGHT
- M3 _citation_check always-aligned: applied, FAILED 2× → CAUGHT
- M4 _narrative.py:1217 refs[0]+1: applied, 15 passed 1 skipped → NOT CAUGHT
- M5 screen.py:75 score=0.0: applied, 86 passed → NOT CAUGHT

## Gates
- G1 lint/typecheck: ruff 249 (14 files; BLE001×112, EXE001×27, C401×20, F841×16, S110×13, ISC004×11, F821×2, F402×1...) / pyright 125 (scripts 106, tests 19) — baselines recorded
- G2 suite: 277 passed, 1 skipped, 9 deselected (live), 2.84s — GREEN
- G3 build: N/A (no build step; imports compile via suite)
- G4 smoke: 16/16 CLIs --help OK; screen.py CLI e2e → NameError (A1 live-bug outranks static); tiny-corpus repro ×2
- G5 deps+secrets: secrets scan clean; httpx2 = habanero transitive (legit); fastembed/pyzotero/sentence-transformers optional fail-loud ✓; requests undeclared-but-transitive (lazy import, would ImportError loudly if parent removed — note only)
- G6 hostile self-review: weakest = P11 (NOT RUN, declared), P9 (golden-delegated), P2 (census-level). P1/P5/P14 re-verified with live repros — strong. ACCEPTED with declared gaps.
- G7 unit account: see unit-inventory.md — census 100%, verdicts assigned (depth-tiered per L13)
- G8 scale-spotlift: P8 reported zero findings at T1-T3 → re-ran deep read of _ratelimits.py + discover pool shutdown + subprocess timeouts → still clean (documented no-with is deliberate)
- G9 diagnostics ledger: ruff 249 → resolved 0 / pre-existing 249 (all at HEAD, none introduced by dirty diff — dirty files contribute 0 ruff errors) / false-positive 0 (BLE001 majority = deliberate optional-dep pattern, still counted). pyright 125 → pre-existing (worktree-HEAD comparison contaminated by version drift = A10; dirty-diff introduces 0 new: the 29 "delta" sites are in files the diff doesn't touch → version noise) / noise-classified 100 (typing), real-bug subset already extracted (A1 x2 confirmed, A11 None-contracts). Observed 374 = resolved 2 (A1,A2 root) + pre-existing 372 - overlaps... final: observed ruff249+pyright125 = 374; dispositioned 374 (every diagnostic classified by rule; real-bug subset enumerated; rest typing/hygiene). No "env noise" dismissals.

## Personas (D11)
- RED TEAM: injection surface clean (list-form subprocess, quoted params, no eval); API keys env-only; pickle self-owned. → A9 identity abuse only.
- SABOTEUR: corrupted cache JSON → silent skip in _search_cache migration (minor); cache `null` → _geodict None-crash (A11); truncated artifact → fail-loud load (A6 mitigation exists but write still non-atomic); hostile corpus fields → parser suite (20 adversarial sentences) green.
- INCOMPETENT USER: wrong CLI order → argparse handles; double-run → manifests idempotent-ish (save overwrites); cancel mid-write → A6.

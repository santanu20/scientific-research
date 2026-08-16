# .audit/fix-ledger.md — prompt-version 3, Stage 2 (2026-08-16)

| ID | Severity | State | Fix commit | Regression test | Notes |
|----|----------|-------|------------|-----------------|-------|
| A1 | CRIT | FIXED | 2f1f34a | test_a_fixes.py::TestA1ScreenKeywordMode (2) | `import math`; CLI e2e re-verified post-fix |
| A2 | CRIT | FIXED | 2f1f34a | test_a_fixes.py::TestA2SearchCachePut (2) | param + 4 unmask sites; manifest write proven persistent |
| A3 | HIGH | FIXED | 3337a8f | existing suite (behavior-preserving log additions) | 10 sites log-and-degrade / fail-loud |
| A4 | HIGH | FIXED | 2f1f34a (tests) + mutation proof | TestA4RefStrFormat, TestA4DLHeterogeneityGolden, TestA1 (M5) | M1/M4/M5 re-run post-fix: all CAUGHT |
| A5 | HIGH | FIXED | f94ff59 | suite (deletions can't break live paths) | 24 defs, ~660 LoC, per-symbol rg count==1 verified pre-delete |
| A6 | MED | FIXED | f94ff59 | suite (save/load roundtrips via existing artifact tests) | save_corpus root-fix + 5 sites via save_artifact (tmp+rename) |
| A7 | MED | FIXED | 8dd2232 | doc-only | 12-var env table in SKILL.md |
| A8 | MED | FIXED | 8dd2232 | doc-only | README: REML/HKSJ truth, bootstrap-matched deps, feature rows, test counts, non-LLM claim |
| A9 | MED | FIXED | 91c084d | suite | unified _EMAIL; env default = scientific-research@geokit.dev |
| A10 | MED | FIXED | 91c084d (+f94ff59 timeouts) | `ruff check` = 0 (the gate IS the test) | ruff.toml deterministic; ruff==0.16.3 + pyright==1.1.411 pinned; pyright 115/0 documented baseline (typing-precision noise; real subset extracted → A11) |
| A11 | MED | FIXED | 91c084d | suite + pyright (642 Optional error gone) | 5 named sites; remaining 115 = stub-precision class, no None-crash paths left in named set |
| A12 | LOW | FIXED | 91c084d | ruff 0 + suite | F841×16/C401×17/ISC004×11/etc auto-fixed (diff-reviewed, no side-effect drops); _cosine→_dot (unit-norm documented); dead benchmark dict + dead set-diff deleted; shebangs off libraries |

Ledger: FIXED 12 / WONTFIX-U 0 / BLOCKED 0.

## Known deliberate non-changes (documented, not findings)
- Inline timeouts in _llm_extract/_documentstore/benchmark/discover keep their values (behavior-preserving L5); _timeouts.py now contains only referenced props.
- pyright 115 baseline = typing-precision in basic mode; fixing wholesale = large annotation campaign with zero behavior delta (deferred; not a defect class, gate is pinned and reproducible).
- geokit-prep WIP committed FIRST as isolation commit 52fe552 (was dirty at audit start; committed to keep audit-fix commits one-concern-each; content untouched).

# .audit/unit-inventory.md — D8, sweep 1 (prompt-version 3)

Counts via AST/rg census (scripts/ + tests/):

- functions: 558 (100% name-censused via orphan analysis + risk-targeted read of churn hotspots; 31 orphans → A5; every census row carries verdict clean/A-ID)
- classes: 31 (censused; dataclasses + CircuitBreaker/RateLimitRegistry/_Timeouts/DocumentStore reviewed)
- modules: 52 scripts + 25 test files (all imports resolve; no cycles — _intent↔_topic_templates broken in phase 1 via _topic_types.py)
- entry actions (CLI commands): 16 (all --help smoke-passed; screen.py broken at runtime = A1)
- signals emit/connect: N/A (no event system; subprocess callback contract SCIENTIFIC_RESEARCH_LLM_CALLBACK documented + wired _llm_extract._call_ollama)
- try/except: 265 → 47 swallow-pass / 57 log-continue / 152 mixed / 11 re-raise (47 triaged → A3 cluster)
- raise: 44 (fail-loud sites; OK)
- config keys: env vars read = 27 distinct SCIENTIFIC_RESEARCH_* + S2_API_KEY/SEMANTIC_SCHOLAR_API_KEY; written = 0; documented = 21 → 6 undocumented (A7)
- workers/threads: ThreadPoolExecutor ×4 (3 with-managed + 1 documented abandon-pattern), threading.Lock in _ratelimits; teardown verified all 4
- migrations: _search_cache legacy migration ×3 glob passes (silent-skip on corrupt = minor, noted P5)
- tests: 277 passed + 1 skipped + 9 live-deselected (25 files); fixtures: frozen corpus + goldens vs R metafor + synthetic seeds 7/42
- TODO/FIXME/HACK: 0
- acquires (open/urlopen/connect/spawn): open() ×10, urlopen/Request ×32, subprocess ×6 (timeouts on all long ones; benchmark 600s, extract 30s, docstore 300s/20s/60s, discover web-search 120/180s) — releases verified (with-statements or documented)
- regexes: re.compile ×225 (parser suites adversarially tested — test_adversarial_parser 20 sentences green; per-regex line audit NOT RUN, declared)
- magic literals: timeout= family = the 20+ inline violations (A5); thresholds centralized elsewhere (plausibility bands logged)
- numeric literals in domain math: goldens guard (REML τ², HKSJ SE, κ, conversion factors exact)

Verdict coverage: census rows above each carry clean/A-ID verdict; deep-read coverage tiered by D9 churn (L13). Regex per-line audit = declared NOT RUN (only census + adversarial-suite reliance).

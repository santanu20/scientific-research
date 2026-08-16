# .audit/aspect-matrix.md — prompt-version 3, sweep 1

| Aspect | WHERE | Risk hypothesis | PASS | Status |
|---|---|---|---|---|
| D1 CLI surfaces (16 entries) | shebang census + __main__ blocks | dead handlers, broken flows | P1 | A1 (screen dead), rest ✓ |
| D1 web-search subprocess integration | discover.py:547-701 | injection/timeout | P10,P8 | clean (timeouts 120/180s) |
| D2 domain model: PaperRecord, StudyEffect, PooledEffect, PublicationBias, TopicTemplate, artifacts (corpus/verified/extracted/correlation/meta) | _sources.py:99, _stats.py, _topic_types.py, _artifact.py | contract drift | P6,P9 | A11 (None contracts) |
| D2 invariants: §5 H20 verify-gate, pooling purity (no cross-unit), AL reorder-only, non-LLM default, fail-loud imports, KB permanence | SKILL.md hard rules, MEMORY key invariants | invariant violations | P9,P14 | held by tests except DL c-constant (A4) |
| D3 data flow: discover→corpus.json→verify→verified.json→extract→extracted.json→correlate→correlation.json→meta→meta.json→synthesize→brief.md | README stage table | dead-ends, shape drift | P1,P6 | A6 (non-atomic writes); shapes enforced on load (_artifact) |
| D4 deps: habanero, pyalex, semanticscholar, arxiv, numpy, scipy, sklearn, httpx, pypdf, matplotlib, networkx (correlate extra), requests (transitive, undeclared), fastembed/pyzotero/sentence-transformers (optional) | _bootstrap._REQ_INSTALLS, lazy imports | undeclared/missing dep crash | P10 | note: requests undeclared; optionals fail loud ✓ |
| D4 external services: Crossref, OpenAlex, S2, arXiv, EPMC, DOAJ, Unpaywall, Wikidata/Wikipedia, Ollama, RW git clone, web-search skill | _sources.py, _geodict.py | rate limits, breakers, timeouts | P8 | clean (breakers+timeouts wired) |
| D5 execution modes: CLI per stage, pipeline.py orchestration, --al, --embeddings, --use-llm family, --self-check ×13, KB cache mode, force-refresh | entry scripts | mode divergence | P1 | A1 (keyword mode dead); --use-llm routes correctly |
| D6 failures: circuit breakers ×4 APIs, retry_with_backoff, FuturesTimeout abandon, graceful optional-dep degrade | _ratelimits, _sources | unhandled failure | P5 | A3 (masking pattern) |
| D7 temporal: KB TTL 30d/90d, geo-dict cache 30d, LLM extract cache 90d, smooth cache content-hash, RW shallow clone | _search_cache, _geodict, _llm_extract | stale caches, TTL bugs | P7 | clean |
| D8 units | unit-inventory.md | coverage | G7 | census 100% |
| D9 churn hotspots | synthesize/_narrative/meta_analyze ×7 commits | bug prediction | risk-order | _narrative+synthesize = top pyright too — audited first |
| D10 GP1-GP6 | pass-ledger | — | executed | 6/6 |
| D11 personas ×3 | pass-ledger | — | executed | 3/3 reported |
| D12 cross-artifact: SKILL.md vs README vs MEMORY vs code | doc diff | contradictions | P15 | A8, A10, A7 |
| DISCOVERED-LATE: none — no aspect surfaced outside D1-D12 naming (open-world check done: geokit-integration dirty diff classified under D5 config surface, no new pass needed — it is 14 lines, inert, audited as part of D1/D5) | — | — | — | closed |

Universe: DISCOVERED 18 ∪ DISCOVERED-LATE 0 = 18 rows, all with PASS + verdict.

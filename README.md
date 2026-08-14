# scientific-research skill

Heavy-work toolkit for deep scientific academic **research** (web-source only).
**Scripts do the heavy lifting (search / verify / extract / correlate / analyze
/ export). The LLM orchestrates and synthesizes.**

5-phase pipeline: **SCOPING → DISCOVERY → VERIFICATION → CORRELATION → SYNTHESIS**.

~46K LLM tokens per 50-paper research run.

## Scope

**In scope:**
- Web research APIs: Crossref, OpenAlex, Semantic Scholar, arXiv, DOAJ
- Citation graph traversal + correlation matrix
- PICO extraction, effect-size extraction, meta-analysis pooling
- Risk-of-bias scaffolds (RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa)
- §5 H20 verification gate (DOI/arXiv resolver + retraction + DOAJ + evidence grading)
- PRISMA 2020 flow generation
- Multi-format citation export (BibTeX / RIS / APA / CSL JSON / Zotero)

**Out of scope (delegate elsewhere):**
- Local PDF text extraction → use the **pdf-ocr** skill
- RAG / QA over local PDF corpora → use PaperQA2 directly in a GPU/OpenAI env
  (CPU Ollama + PaperQA2 is too slow: 5-15 min/PDF, 4-12 hours for 50-PDF corpus)

## Quickstart

```bash
# Required deps (in project venv)
uv add habanero pyalex semanticscholar arxiv numpy scipy

# Optional (specific features)
uv add sentence-transformers   # embedding-based screening in screen.py --embeddings
uv add pyzotero                 # Zotero push in export_citations.py --zotero-key

# Run a research workflow (LLM invokes these via the skill)
SK=~/.config/opencode/skills/scientific-research/scripts
uv run python $SK/discover.py "topic" --max 50 -o corpus.json
uv run python $SK/verify.py corpus.json -o verified.json
uv run python $SK/extract.py verified.json
uv run python $SK/correlate.py verified.json
uv run python $SK/export_citations.py verified.json --format all
```

## What it does

| Stage | Tool | What | Output |
|---|---|---|---|
| 1. SCOPING | LLM interactive | clarify question, PICO, criteria | `research_plan.json` |
| 2. DISCOVERY | `discover.py` | multi-source web search (Crossref + OpenAlex + S2 + arXiv) + Cohen 2018 snowballing + dedup | `corpus.json` + `corpus_summary.md` |
| 3. VERIFICATION | `verify.py` | §5 H20 DOI/arXiv resolver gate + OpenAlex `is_retracted` + DOAJ + evidence grading (L I-VII) + RoB tool selection | `verified.json` + `verification_results.json` + `verification_report.md` |
| 3b. SCREENING | `screen.py` | PRISMA Phase 2 (title/abstract triage, keyword + optional embedding mode) | `screened.json` + `screening_report.md` |
| 4a. EXTRACTION | `extract.py` | PICO/SPIDER + effect sizes (mean±SD, events, OR/RR/HR, Cohen's d) + methodology keywords | `extracted.json` + `extraction_report.md` |
| 4b. ASSESSMENT | `assess.py` | RoB scaffolds (RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa) + GRADE profiles | `assessment.json` + `assessment_report.md` |
| 4c. CORRELATION | `correlate.py` | citation graph (Mermaid) + bibliographic coupling + concept clusters + temporal histogram + author network + funding analysis + correlation matrix scaffold + Scite-style citation context pre-classification | `correlation.json` + `correlation.md` + `graph.mmd` |
| 4d. META-ANALYSIS | `meta_analyze.py` | DerSimonian-Laird fixed/random pooling + heterogeneity (Cochrane Q, I², τ²) + forest plot + subgroup analysis | `meta.json` + `forest.mmd` |
| 5. SYNTHESIS | `synthesize.py` | 5 research types (verification/survey/subtopic/comparative/data_compilation), chronological narrative with [N] citations, interpretation enrichment, significance inference, theme clustering. `--use-llm` for prose smoothing. | `research_brief.md` |
| 5b. EXPORT | `export_citations.py` | BibTeX + RIS + APA + CSL JSON + optional Zotero push | `references.{bib,ris,md,csl.json}` |
| opt. MONITOR | `monitor.py` | living-document alerts on new papers since date | `alerts.json` |

## Sources (verified live)

- **Crossref** via `habanero` (DOI resolution, citation export, metadata)
- **OpenAlex** via `pyalex` + raw API (319M works, semantic search, `is_retracted`, citation graph, concepts, funders)
- **Semantic Scholar** via `semanticscholar` (TLDR, `influentialCitationCount`, recommendations, citation contexts/intents)
- **arXiv** via `arxiv` (preprints, abstracts)
- **DOAJ** via direct API v2 (`/api/v2/search/journals`) — whitelist check (positive OA signal only)

## References (verified, in source code)

- PRISMA 2020 — Page MJ et al. BMJ 2021;372:n71
- Cohen 2018 snowballing — J Clin Epidemiol
- RoB 2 — Sterne JAC et al. BMJ 2019;366:l4898
- ROBINS-I — Sterne JAC et al. Ann Intern Med 2016
- QUADAS-2 — Whiting PF et al. Ann Intern Med 2011;155:529-536
- Newcastle-Ottawa Scale — Wells GA et al.
- GRADE — Guyatt GH et al. BMJ 2008;336:924-926
- DerSimonian-Laird — Control Clin Trials 1986;7:177-188
- Cochrane Handbook — https://training.cochrane.org/handbook/current

## §5 H20 verification gate (anti-fabrication)

Every paper cited MUST pass `verify.py`:

1. **Resolve via ≥1 canonical resolver** — OpenAlex (preferred, more reliable than
   Crossref per Master Bruce's note in MEMORY.md), Crossref fallback, S2 fallback.
2. **Retraction check** — OpenAlex `is_retracted` (115k+ retracted papers indexed
   live) + Crossref `type:retraction`.
3. **DOAJ whitelist** — positive OA signal only. NOT a predatory blacklist;
   many legit subscription journals not in DOAJ.
4. **Evidence grading** — Oxford CEBM Level I (systematic review) → VII (editorial).
5. **RoB tool selection** — auto-picks RoB 2 (RCT) / ROBINS-I (non-randomized
   interventions) / QUADAS-2 (diagnostic) / Newcastle-Ottawa (observational).

**Unresolved papers are FORBIDDEN to cite.** Citing unresolved = fabrication.

## Knowledge base + cache

**Knowledge base** at `~/.local/share/scientific-research/` (XDG data,
permanent): per-paper JSON at `<sha256>/paper.json` keyed by primary identifier
(DOI > arXiv > PMID > OpenAlex ID > title hash). Survives across runs.

**LLM caches** at `~/.cache/scientific_research/`:
- `llm_extract/` — extraction cache (TTL 90d, prompt-versioned)
- `llm_smooth/` — prose smoothing cache (content-hash)

Set `SCIENTIFIC_RESEARCH_KB` env var to override knowledge base location.

## Limitations

- **Crossref returns bad DOIs occasionally** — verify.py cross-checks OpenAlex
  + Crossref + S2 to defend against this.
- **Non-LLM extraction catches ~80-90% of findings** — SVM stance (100% test),
  effect parser (100% TP), discipline (100%), PICO (96-100%). LLM (`--use-llm`)
  refines edge cases.
- **Non-LLM synthesis gap vs LLM ~15%** — template stitching has interpretive
  enrichment (interpretation appending, significance inference, verb rephrasing)
  but lacks LLM-level sentence merging and contextual rephrasing.
- **S2 API rate-limits unauthenticated** — circuit breaker skips after 1 failure
  (10s max wasted), Crossref/OpenAlex/arXiv continue.
- **Pure JSON knowledge base** — no DB. For 1000s of papers across many runs,
  dir scan may get slow. (User choice — kept simple.)
- **No local PDF processing** — by design. Use pdf-ocr skill for that.

## File layout

```
~/.config/opencode/skills/scientific-research/
├── SKILL.md                          # orchestrator prompt
├── README.md                         # this file
├── pytest.ini                        # test config
├── tests/
│   └── test_skill.py                 # unit tests (72 tests)
└── scripts/
    ├── _sources.py                   # 4 source wrappers + parallel search + retry
    ├── _search_cache.py              # knowledge base (manifests + paper store)
    ├── _classifiers.py               # discipline/novelty/study_type + research type detection
    ├── _nlp.py                       # PICO + TextRank + FakeRAKE + Schwartz-Hearst
    ├── _lexicon.py                   # stance lexicon (Jurgens 2018)
    ├── _stance_svm.py                # SVM stance classifier (927 training sentences)
    ├── _effect_parser.py             # field-agnostic effect size extraction
    ├── _llm_extract.py               # Ollama LLM wrapper (auto-detect, caching, /no_think)
    ├── _narrative.py                 # narrative builders (chronological/verification/comparative/compilation)
    ├── _ranking.py                   # 6-criteria paper ranking
    ├── _render.py                    # Mermaid renderers
    ├── _stats.py                     # effect sizes + heterogeneity + pooling
    ├── discover.py                   # web search + snowball + PRISMA
    ├── screen.py                     # PRISMA Phase 2 screening
    ├── verify.py                     # §5 H20 verification gate + circuit breaker
    ├── extract.py                    # PICO + effect sizes + HTML sanitization
    ├── assess.py                     # RoB scaffolds + GRADE profiles
    ├── correlate.py                  # citation graph + clusters + matrix + contexts
    ├── meta_analyze.py               # DerSimonian-Laird pooling + forest
    ├── synthesize.py                 # Phase 5 narrative synthesis orchestrator
    ├── monitor.py                    # living-document alerts
    ├── export_citations.py           # BibTeX/RIS/APA/CSL JSON/Zotero
    └── data/
        ├── stance_training.jsonl     # 927 labeled stance sentences (19 fields)
        ├── rob_tools.json            # RoB tool selection rules
        ├── grade_factors.json        # GRADE evidence profiles
        └── evidence_levels.json      # Oxford CEBM evidence levels
```

22 scripts + SKILL.md + README.md + tests + data. ~8000 lines total.

## License

Skill code: same as opencode config (per-user).
Dependencies retain their upstream licenses (all MIT/Apache-2.0/Unlicense).

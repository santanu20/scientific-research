---
name: scientific-research
description: >
  Deep scientific academic research workflow with cross-paper correlation,
  citation-context mining, §5 H20 DOI/arXiv verification gate, PRISMA 2020
  methodology, Cohen 2018 snowballing, field-agnostic PICO extraction, RoB
  scaffolds (RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa), GRADE evidence
  profiles, fixed/random meta-analysis pooling, multi-format citation export
  (BibTeX/RIS/APA/CSL JSON/Zotero). 5-phase pipeline (SCOPING → DISCOVERY →
  VERIFICATION → CORRELATION → SYNTHESIS). SOTA non-LLM path (SVM stance +
  weighted keyword discipline + effect parser + field-agnostic PICO). Optional
  LLM extraction via Ollama (--use-llm). Web research only — for PDF text
  extraction, delegate to the pdf-ocr skill. Triggers: scientific research,
  literature review, find papers, cross-reference papers, cross-correlate
  papers, research brief, systematic review, meta-analysis, snowballing, PICO
  extraction, risk of bias, RoB, PRISMA, evidence synthesis.
model: inherit
---

# Scientific Research Skill

Heavy-work toolkit for deep scientific academic **research** (web-source only).
The LLM **orchestrates** scripts that do the heavy lifting (search / verify /
extract / correlate / analyze / export); the LLM synthesizes narrative +
advisory suggestions.

**Scope:** web-research APIs (Crossref + OpenAlex + Semantic Scholar + arXiv) +
DOAJ + Unpaywall. **Out of scope:** local PDF processing (use the **pdf-ocr**
skill for that), RAG over PDF corpora (outsource to PaperQA2/GPT-Researcher in a
dedicated environment).

## When to invoke

Trigger when the user asks to:
- Find / discover papers on a topic
- Do a literature review
- Cross-correlate / cross-reference papers
- Verify references (DOI / arXiv / PMID)
- Build a research brief
- Conduct a systematic review / meta-analysis
- Snowball from a seed paper
- Extract PICO data or effect sizes
- Assess risk of bias (RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa)
- Export citations (BibTeX / RIS / Zotero)

## Hard rules

1. **§5 H20 verification is MANDATORY.** Every paper cited in any brief, matrix,
   or export MUST pass `verify.py`. Unresolved = forbidden cite.
2. **Advisory-only suggestions.** Never auto-generate research questions. Surface
   gaps + next directions as suggestions; the user decides.
3. **NEVER clear the knowledge base.** The knowledge base
   (`~/.local/share/scientific-research/`) is a permanent, growing library.
   Paper-wise files store full records (title, abstract, authors, references).
   Two manifest indexes (`search_manifest.json`, `verify_manifest.json`) point
   to paper IDs. Over time, repeated/related searches find most papers already
   stored → 0 API calls, instant results.
   Do NOT run `rm -rf ~/.local/share/scientific-research/` or equivalent.
   Check stats: `uv run python scripts/_search_cache.py`
4. **No fabrication.** Every claim in a synthesis must trace to a verified paper
   in `verified.json` with a stable identifier (DOI preferred).
5. **Fail loud on missing deps.** Scripts raise `ImportError` with install
   instructions; do not silently fall back.
6. **Web research only.** Direct users to the `pdf-ocr` skill for any local PDF
   text extraction need. Do NOT attempt to wrap or replicate pdf-ocr here.
7. **Non-LLM is the default.** All extraction (PICO, stance, effects, discipline,
   novelty) runs via regex + SVM + keyword classifiers — no LLM needed. Use
   `--use-llm` flag on extract.py / `--use-llm-stance` on correlate.py for
   higher quality (requires Ollama). Use `--force-refresh` on discover.py /
   verify.py to skip the knowledge base and fetch fresh from APIs.

## Pre-flight check

**Skill venv auto-bootstraps** on first run of any entry script. Creates
`~/.config/opencode/skills/scientific-research/.venv` + installs habanero,
pyalex, semanticscholar, arxiv, numpy, scipy, scikit-learn, httpx, pytest,
ruff. Re-execs into skill venv on every subsequent run — works from any cwd
without polluting project venvs. Stale venv (missing dep) self-repairs via
one auto-install. Bootstrap logic lives in ONE place: `scripts/_bootstrap.py`
(entry scripts call `_bootstrap.ensure_env()`; `correlate.py` passes
`networkx` extras).

Opt-out env:
- `SCIENTIFIC_RESEARCH_NO_SKILL_VENV=1` — force cwd venv (skip re-exec)
- `SCIENTIFIC_RESEARCH_NO_BOOTSTRAP=1` — skip auto-install

To pre-create (skip auto-bootstrap on first run):
```bash
# Required deps (in skill venv, auto-used by all scripts)
SKILL=~/.config/opencode/skills/scientific-research
uv venv "$SKILL/.venv" --python 3.13
uv pip install --python "$SKILL/.venv/bin/python" \
    habanero pyalex semanticscholar arxiv numpy scipy scikit-learn \
    httpx pytest ruff

# Retraction Watch local index (optional but recommended — 109k DOIs):
git clone https://gitlab.com/crossref/retraction-watch-data \
    ~/.local/share/scientific-research/retraction-watch-data
# auto-discovered at that path; update with `git pull` (daily upstream refresh)

# Optional LLM extraction (requires Ollama with text-generation model)
# Auto-detects smallest suitable model (excludes vision/embedding)
# Check: uv run python scripts/_llm_extract.py
# Override: SCIENTIFIC_RESEARCH_LLM_MODEL=qwen3.5:2b
```

### Agent-side LLM callback (separation of concerns)

Per AGENTS.md §4.23: **skill = infrastructure, agent = cognition**. LLM calls
are cognition. The skill supports two paths:

1. **Default**: skill calls Ollama directly (via `--use-llm` flags). Fine for
   standalone use, but couples skill to a local LLM server.
2. **Agent-injected** (preferred for agent workflows): set
   `SCIENTIFIC_RESEARCH_LLM_CALLBACK=/path/to/executable` and the skill
   delegates ALL LLM calls to that executable. Skill builds prompts + parses
   responses; agent owns the LLM.

**Callback contract:**

```bash
# Agent registers a wrapper executable:
export SCIENTIFIC_RESEARCH_LLM_CALLBACK=/usr/local/bin/my-llm-wrapper.sh
```

The wrapper receives JSON on **stdin**, returns response text on **stdout**:

```json
// stdin payload (one JSON object per LLM call):
{
  "model": "qwen2.5:3b",      // suggested model (agent may override)
  "prompt": "...",             // the rendered prompt
  "task": "moderate"           // difficulty hint: light|moderate|heavy
}
```

```bash
# Example wrapper (delegates to agent's own Ollama / OpenAI / Anthropic / etc):
#!/usr/bin/env bash
input=$(cat)
prompt=$(echo "$input" | jq -r .prompt)
# Agent's choice — could be Claude, GPT, local Ollama, anything:
echo "$prompt" | my-agent-llm-call
```

The hook fires inside `_llm_extract._call_ollama()` and transparently
replaces EVERY Ollama call across all 5 LLM-using scripts (`_intent.py`,
`_llm_extract.py`, `_query.py`, `assess.py`, `synthesize.py`). No code
changes needed — pure env-var configuration.

**Run-scoped rate limits + centralized timeouts** are wired into `_sources.py`
by default. To tune at runtime, set env vars (`SCIENTIFIC_RESEARCH_TIMEOUT_*`):
```bash
SCIENTIFIC_RESEARCH_TIMEOUT_CROSSREF=60   # default 30s
SCIENTIFIC_RESEARCH_TIMEOUT_OPENALEX=20   # default 10s
SCIENTIFIC_RESEARCH_TIMEOUT_S2=30         # default 10s
```
Circuit-breaker thresholds + pacing intervals live in `_ratelimits.RateLimitRegistry`
(defaults: Crossref threshold=3/0.5s, OpenAlex 5/1.0s, S2 5/3s, arXiv 3/1.0s).

The skill scripts live at:
`~/.config/opencode/skills/scientific-research/scripts/`

Each entry script is **self-contained** (own inline bootstrap). Run them via
`python scripts/<name>.py` from anywhere — venv re-exec handles the rest.
Outputs default to `./research_outputs/`.

## Knowledge base architecture

```
~/.local/share/scientific-research/          ← PERMANENT (XDG data, never deleted)
  ├── <hash>/paper.json    ×N papers         ← Full records (title, abstract, authors, refs)
  ├── search_manifest.json ×1 file           ← Index: query → paper_ids + timestamp (TTL 30d)
  └── verify_manifest.json ×1 file           ← Index: doi → verification result (TTL 90d)

~/.cache/scientific_research/                ← TEMPORARY (disposable)
  ├── llm_extract/         ×N extractions    ← LLM extraction cache (TTL 90d)
  └── llm_smooth/          ×N smoothing      ← LLM prose smoothing cache (content-hash)
```

**Paper-wise files are the single source of truth.** Each paper gets its own
`<sha256_hash>/paper.json` with the complete PaperRecord (title, abstract,
authors, year, venue, citations, references, concepts, keywords, funding, OA
URL, raw metadata — everything). Manifests are lightweight indexes pointing to
paper IDs.

**Flow:** Search → fetch papers from API → save each to paper-wise file →
update manifest → next time same/related search → manifest hit → load from
paper-wise files (0 API calls). Use `--force-refresh` to bypass.

## 5-Phase workflow

```
SCOPING ─→ DISCOVERY ─→ VERIFICATION ─→ CORRELATION ─→ SYNTHESIS
               ↑              ↑
               └─ corpus <threshold ─┐
                             if key paper unverifiable ─┘
```

Max 2 loop-backs per phase. No infinite loops.

### Phase 1 — SCOPING (LLM interactive)

Clarify the research question with the user. Capture (use `AskUserQuestion`,
max 1 turn, batched):

| Element | Why needed |
|---|---|
| Research question (1 sentence) | Drives all downstream queries |
| Field/discipline | Helps ranking + extraction |
| Scope (systematic review / narrative / scoping) | Determines PRISMA strictness |
| Corpus size cap (default ≤100) | `discover.py --max` |
| Optional: seed paper(s) | `discover.py --snowball` anchor |

Write plan to `research_outputs/research_plan.json`.

### Phase 2 — DISCOVERY (`discover.py`)

Web research only (Crossref + OpenAlex + S2 + arXiv).

```bash
# Default: non-LLM, fast, cached
uv run python scripts/discover.py "research question" \
    --max 100 --sources crossref,openalex \
    --snowball 1 --snowball-saturation --snowball-depth-max 2 \
    --expand --expand-max-terms 12 \
    -o research_outputs/corpus.json --summary

# Force fresh fetch (skip knowledge base):
uv run python scripts/discover.py "research question" \
    --max 100 --sources crossref,openalex --force-refresh \
    -o research_outputs/corpus.json --summary

# With year range + OA filter + publication type:
uv run python scripts/discover.py "research question" \
    --max 100 --sources crossref,openalex \
    --from-year 2020 --to-year 2024 --open-access-only --type journal-article \
    -o research_outputs/corpus.json --summary
```

**6-criteria ranking** (TF-IDF semantic + citation velocity + co-citation +
PageRank + venue h-index + recency boost + diversity filter + recency balance).
Fetches up to 500 via OpenAlex cursor pagination, ranks, selects top N.

**Query expansion** (`--expand`): FakeRAKE + co-occurrence network (ported from
litsearchr). **Saturation snowballing** (`--snowball-saturation`): stops when
new-paper ratio drops below 15%. **PRISMA diagram** auto-generated as `prisma.png`.

#### Web-search skill integration (`--use-web-search` family)

The scientific-research skill can delegate to the **web-search skill** (separate
MCP skill at `~/.config/opencode/skills/web-search/`) for SOTA web-wide paper
discovery. Three modes, all opt-in:

| Mode | Flag | Speed | Best for |
|------|------|-------|----------|
| **Text mode** | `--use-web-search` | ~10s | General supplementation — 9 metasearch backends (DDG/Google/Brave/Yandex/...) + 4-layer bot-block bypass. Finds papers Crossref/OpenAlex/S2 miss (conference proceedings, niche journals). |
| **Agentic mode** | `--use-web-search-agentic` | 30-120s | Deep research — Crawl4AI AdaptiveCrawler on top search hit. Semantic relevance + confidence score. Finds papers in JS-rendered sites / behind search forms. |
| **Auto-supplement** | `--auto-web-search-threshold N` (default 5) | ~10s when triggered | Zero-config safety net — auto-enables text mode when post-dedup corpus < N papers. Catches niche topics where academic APIs miss the field. Set 0 to disable. |

**How it works:** web-search subprocess fetches results, skill extracts DOIs
from snippets/pages, resolves via Crossref for metadata. Clean separation:
web-search = web infrastructure, scientific-research = academic knowledge.

**`--force-refresh`** propagates to the web-search subprocess (sets
`WEB_SEARCH_NO_CACHE=1`), bypassing per-URL extract cache.

```bash
# Explicit text-mode supplement:
uv run python scripts/discover.py "research question" \
    --use-web-search --max 100 -o research_outputs/corpus.json

# Deep research with adaptive crawl on top hit:
uv run python scripts/discover.py "research question" \
    --use-web-search-agentic --max 50 -o research_outputs/corpus.json

# Zero-config auto-supplement (default threshold=5):
uv run python scripts/discover.py "niche topic" \
    --max 30 -o research_outputs/corpus.json
# → if post-dedup papers < 5, auto-adds web_search source
```

### Phase 3 — VERIFICATION (`verify.py`) — §5 H20 GATE

MANDATORY before any citation. Multi-resolver cross-check (OpenAlex + Crossref),
retraction check, evidence-level grading, RoB tool selection.

```bash
# Fast: skip S2 (often slow), use knowledge base
uv run python scripts/verify.py research_outputs/corpus.json \
    -o research_outputs/verified.json \
    --keep-unresolved --skip-s2 --no-doaj

# Force re-verify (skip knowledge base):
uv run python scripts/verify.py research_outputs/corpus.json \
    -o research_outputs/verified.json --force-refresh
```

### Phase 4 — CORRELATION

Four sub-phases. **Non-LLM by default** (regex + SVM + keyword classifiers).
Use `--use-llm` for higher quality (requires Ollama).

```bash
# 4a. Extraction (NON-LLM default: regex + SVM stance + keyword discipline)
#     Extracts: subject, method, discipline, novelty, study_type, key_finding,
#     interpretation, effect sizes, abbreviations, risk_of_bias
uv run python scripts/extract.py research_outputs/verified.json \
    --topic "research question"

# 4a. WITH LLM (higher quality, ~5s/paper):
uv run python scripts/extract.py research_outputs/verified.json \
    --topic "research question" --use-llm

# 4b. Assessment (reads RoB from extraction, instant)
uv run python scripts/assess.py research_outputs/verified.json \
    --extractions research_outputs/extracted.json

# 4c. Correlation (SVM stance default, instant)
uv run python scripts/correlate.py research_outputs/verified.json \
    --no-contexts

# 4c. WITH LLM stance (higher quality, ~3s/cell for high-relevance):
uv run python scripts/correlate.py research_outputs/verified.json \
    --no-contexts --use-llm-stance

# 4d. Meta-analysis (for clinical: --measure auto; for geology: --measure single)
#     Defaults since 2026-08-14 (Cochrane MECIR-aligned): tau2=REML, CI=HKSJ,
#     prediction interval, leave-one-out influence, Vevea-Hedges selection
#     model + p-curve alongside Egger/trim-and-fill. Escape hatches:
#     --tau2 dl|reml|pm and --no-hksj restore legacy behavior.
uv run python scripts/meta_analyze.py \
    research_outputs/extracted.json research_outputs/verified.json \
    --model random --correction hedges
```

**Non-LLM accuracy** (validated on test set):
- Stance (SVM+lexicon+discourse): ~80-100%
- Discipline (weighted keywords, 19 fields): 100%
- Novelty (cue phrases): 100%
- Key finding / interpretation: 100%
- PICO (sentence-position + structured abstract): 96-100%
- Effect parser (field-agnostic): 100% true positives, 0% false positives

**LLM extraction** (`_llm_extract.py`): auto-detects smallest Ollama model
(excludes vision/embedding), `think: false`, `temperature: 0.0`, `top_k: 1`,
`/no_think` prefix, `keep_alive: 30m`. NO `format: "json"` (causes empty
responses with Qwen3.5 on long prompts — JSON parsed by `_extract_json_from_text`).
Task-based selection (simple=stance, moderate=extraction). Auto-expands context
window for long abstracts. Empty Ollama response → immediate regex fallback
(no retry — saves 15s/paper).

### Phase 5 — SYNTHESIS (`synthesize.py` + `_narrative.py`) + EXPORT

`synthesize.py` produces a narrative `research_brief.md` from
extracted.json + verified.json + correlation.json. **Five research types**
drive output structure:

| Type | Detected from query | Output format |
|---|---|---|
| **verification** | "is it true...", "does X...", "?" suffix | Verdict + stance evidence summary |
| **survey** | "latest research on", "overview of", short broad topic | Chronological narrative paragraphs |
| **subtopic** | Multi-word technical focus ("O isotopes in arc magmas") | Focused narrative + measurement summary |
| **comparative** | "compare", "vs", "versus", "difference between" | Side-by-side grouping by approach |
| **data_compilation** | "compile", "reported values", "global dataset" | Data table + statistical summary |

NON-LLM DEFAULT: template stitching from `key_finding` + `interpretation` per
paper, with verb rephrasing, significance inference (support/contrast/extend),
year-gap/discipline-shift transitions, theme-based paragraph clustering, and
topical filtering. Numbered citations [1][2]. `--use-llm` opt-in: Ollama
rewrites templates into smoother prose (cached via content-hash at
`~/.cache/scientific_research/llm_smooth/`).

```bash
uv run python scripts/synthesize.py \
    --query "latest research on basalt geochemistry" \
    --extracted research_outputs/extracted.json \
    --verified  research_outputs/verified.json \
    --correlation research_outputs/correlation.json \
    -o research_outputs/research_brief.md
# Add --use-llm for prose smoothing (needs Ollama running)
```

```
research_outputs/
├── research_plan.json         (Phase 1)
├── corpus.json                (Phase 2)
├── prisma.png                 (Phase 2, PRISMA flow diagram)
├── verified.json              (Phase 3)
├── extracted.json             (Phase 4a)
├── assessed.json              (Phase 4b)
├── correlation.json           (Phase 4c)
├── correlation.md             (Phase 4c)
├── meta.json                  (Phase 4d)
├── meta_report.md             (Phase 4d)
├── forest.png / funnel.png    (Phase 4d)
└── research_brief.md          (Phase 5, synthesize.py)
```

Synthesis brief structure (`synthesize.py` auto-generates — not LLM-written):

```markdown
# Research Brief: <topic>

**Research type**: <verification|survey|subtopic|comparative|data_compilation>
**Corpus**: <n> papers
**Date range**: <earliest>–<latest>
**Generated**: <timestamp>
**Method**: template-based (non-LLM) | LLM-smoothed

---

<narrative paragraphs — chronological with [N] citations, transitions,
interpretation enrichment, theme-based paragraph breaks>

## References
[1] Author et al. (Year). Title. DOI:xxx
[2] ...
```

**Paper categorization**: use `study_type` and `novelty` fields from
extracted.json to group papers into: Foundational (novelty=milestone), Reviews
(study_type=review), Methods (study_type=method_development), Applications
(study_type=experimental/field_study), Recent (year ≥ current-5).

Then export citations:

```bash
uv run python scripts/export_citations.py research_outputs/verified.json \
    --format all -o citations
```

## PRISMA accountability & brief verification

Two audit tools + one citation verifier. Pure additive infrastructure
ported from geokit's research module — foundational modules for future
wiring of run-scoped rate limiting and centralized timeouts into the
existing pipeline scripts.

### Brief citation alignment (`_citation_check.py`)

Verifies that every `[N]` citation in a synthesized brief body aligns
with the `References[N]` entry. Catches the theme-offset-vs-linear
numbering mismatch that produces phantom citations.

```bash
# Inline check (returns nothing if aligned, lists mismatches if not)
uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from _citation_check import verify_citation_alignment
brief = open('research_outputs/research_brief.md').read()
mismatches = verify_citation_alignment(brief, require_doi_match=True)
for m in mismatches: print(m)
"
```

Options: `require_doi_match=True` (strict DOI equality), `strict_word_overlap=True`
(heuristic body-vs-ref word overlap; **warning**: false positives on jargon-heavy
findings). Default = structural checks only (existence + range).

Contract test: `tests/test_brief_citations.py` (5 cases: aligned, DOI mismatch,
missing-refs section, orphan citation, disjoint word overlap).

### Screening evaluation (`eval_screening.py`) — PRISMA precision/recall

After manually labeling a corpus, measure screening precision/recall/F1
against the LLM-only `screen_llm.py` filter. Outputs a confusion matrix
+ DOI-level false-positive/false-negative lists.

```bash
# Label corpus.json (JSONL, one line per paper):
#   {"doi": "10.xxx", "label": "include"}   # relevant
#   {"doi": "10.yyy", "label": "exclude"}   # not relevant
#   {"doi": "10.zzz", "label": "maybe"}     # ambiguous, excluded from metrics

uv run python scripts/eval_screening.py research_outputs/corpus.json \
    --query "amphibole thermometery" \
    --labels research_outputs/labels.jsonl \
    -o research_outputs/screening_metrics.json
# Exit code: 0 if F1 ≥ 0.85 (target), 2 otherwise
```

### Screening explanation (`explain_screening.py`) — per-paper audit

Prints per-paper screening decisions + stage statistics for debugging
"why was paper X excluded?" questions and PRISMA audit trails.

```bash
uv run python scripts/explain_screening.py research_outputs/corpus.json \
    --query "amphibole thermometery" --show-excluded
```

### `screen.py` vs `screen_llm.py` — two screening modes

| Module | Mode | When to use |
|---|---|---|
| `screen.py` (existing) | Regex + optional sentence-transformer embeddings | Offline, no Ollama; PRISMA Phase 2 triage |
| `screen_llm.py` (new) | LLM-only judge (content-type filter + Ollama verdict) | Higher precision (~90%+ per O'Mara-Eves 2015); needs Ollama running. Falls back to permissive-accept when Ollama unavailable. |

`eval_screening.py` and `explain_screening.py` use `screen_llm.py` by
default. For evaluating the regex/embedding mode, run `screen.py` separately
and compare counts manually.

### Foundational infrastructure (wired into pipeline)

| Module | Purpose | Status |
|---|---|---|
| `_timeouts.py` | Centralized env-driven timeout constants (`SCIENTIFIC_RESEARCH_TIMEOUT_*`) | **Wired** into `_sources.py` (replaces hardcoded timeout=5/15/30) |
| `_ratelimits.py` | Run-scoped `CircuitBreaker` + `RateLimitRegistry` (eliminates cross-run state pollution) | **Wired** into `_sources.py` (replaces module-level breaker globals for Crossref, OpenAlex, S2) |

Per PaperQA2 v5 + STORM run-scoped context pattern. The pipeline creates a
fresh `RateLimitRegistry` per run via `set_registry()` → automatic reset, no
globals to forget. Backward-compat preserved: `reset_crossref_circuit()`,
`reset_openalex_circuit()`, `reset_s2_circuit()`, `enable_s2_for_enrichment()`,
`force_skip_s2()` remain as thin wrappers delegating to the registry.

Thread-safety verified by `tests/test_ratelimits.py` (concurrent failure + pacing tests).

## Key flags

| Flag | Script | Purpose |
|---|---|---|
| `--force-refresh` | discover.py, verify.py | Skip knowledge base, fetch fresh from APIs |
| `--use-llm` | extract.py | Use Ollama LLM for extraction (higher quality, ~5s/paper) |
| `--use-llm` | synthesize.py | Use Ollama LLM for prose smoothing (cached, ~11s) |
| `--use-llm-stance` | correlate.py | Use Ollama LLM for stance (higher quality, ~3s/cell) |
| `--skip-s2` | verify.py | Skip Semantic Scholar (much faster) |
| `--no-doaj` | verify.py | Skip DOAJ lookup (faster) |
| `--keep-unresolved` | verify.py | Keep unresolved papers in output |
| `--measure single` | meta_analyze.py | Pool single measurements (non-clinical fields) |
| `--expand` | discover.py | Auto-expand query using FakeRAKE |
| `--snowball-saturation` | discover.py | Saturation-detecting snowballing |
| `--from-year` | discover.py | Earliest publication year |
| `--to-year` | discover.py | Latest publication year |
| `--open-access-only` | discover.py | Only discover OA papers |
| `--type` | discover.py | Publication type (journal-article, book-chapter, etc.) |
| `screen.py` | PRISMA-style paper screening (regex + optional `--embeddings` mode) | `screen.py corpus.json --query "topic"` |
| `screen_llm.py` | LLM-only paper screening (content-type filter + Ollama judge; permissive fallback) | used by `eval_screening.py` / `explain_screening.py` |
| `eval_screening.py` | PRISMA precision/recall evaluation against human-labeled corpus | `eval_screening.py corpus.json --query "..." --labels labels.jsonl` |
| `explain_screening.py` | Per-paper screening decision audit + stage statistics | `explain_screening.py corpus.json --query "..." --show-excluded` |
| `_citation_check.py` | Brief citation alignment verifier (body `[N]` vs References `[N]`) | `verify_citation_alignment(brief, require_doi_match=True)` |
| `_ratelimits.py` | Run-scoped `CircuitBreaker` + `RateLimitRegistry` (wired into `_sources.py`) | `from _ratelimits import get_registry; get_registry().crossref.pace()` |
| `_timeouts.py` | Centralized env-driven timeout constants (wired into `_sources.py`) | `from _timeouts import TIMEOUTS; urlopen(req, timeout=TIMEOUTS.openalex)` |
| `monitor.py` | Literature monitoring — finds new papers since last search | `monitor.py --query "topic" --since 2024-01-01` |
| `_provenance.py` | Run provenance manifest — per-stage SHA-256 I/O hashes, params, dep versions (`run_provenance.json`; FAIR/PRISMA-S audit trail, wired into pipeline.py) | `from _provenance import record_stage, load_manifest` |
| `report.py` | PRISMA-S 16-item search report + PROSPERO draft renderer (auto-fills from provenance; unknowns stay `[FILL]`, never guessed) | `report.py research_outputs/ --query "..."` |

## P0/P1 SOTA upgrades (2026-08-14)

| Capability | Where | Notes |
|---|---|---|
| Canonical Duval-Tweedie trim-and-fill (L0, metafor-port) | `_stats.trim_and_fill` | Golden-parity vs R metafor 5.0-1 (k0 + pooled match); replaced count-diff approximation |
| REML / Paule-Mandel τ² + HKSJ CI + prediction interval + leave-one-out | `_stats.pool_random_advanced`, `meta_analyze --tau2/--hksj` | REML default (MECIR); goldens vs metafor; PM ≈ REML numerically |
| Vevea-Hedges 3-PSM + p-curve | `_stats.vevea_hedges_selection_model`, `p_curve_test` | Auto-run in meta_analyze publication-bias block; low-power caution at k_sig<5 |
| Active-learning screening (ELAS-Ultra stack) | `screen.py --al --labels labels.jsonl` | TF-IDF(1-2)+LinearSVC, P(include) queue, **reorder-only — never auto-excludes**; advisory recall curve |
| LLM screening ensemble (majority vote) | `screen_llm.py screen_paper(ensemble=True)`, `eval_screening.py --ensemble` | Votes by DISTINCT Ollama models; degrades to honest single-judge note when only 1 model |
| Retraction Watch secondary check | `verify.py --rw-csv PATH` or env `SCIENTIFIC_RESEARCH_RW_CSV` | Concern-only (never auto-retracted), fail-open; CSV: `git clone https://gitlab.com/crossref/retraction-watch-data` |
| Europe PMC source | `discover.py --sources ...,epmc` | REST, no key, live-verified mapping; PubMed+PMC+preprints |
| Blocked dedup (author-surname+year) | `_sources.dedup_papers` | Kills generic-title false merges (Berra 2023 method); strict fallback threshold when block incomplete; collision disambiguation stops silent paper loss |
| Cohen's κ | `eval_screening.py` | Machine-vs-human agreement, sklearn-cross-checked |
| Stance training data extension | env `SCIENTIFIC_RESEARCH_STANCE_DATA=/path/extra.jsonl` | Same-schema `{"text","label","field"}`; SciCite intent labels rejected (not stance — anti-fabrication) |
| Pluggable embedding model | env `SCIENTIFIC_RESEARCH_EMBED_MODEL` | Default stays bge-base-en-v1.5; cache keys include model name |
| PRISMA-S + PROSPERO renderers | `report.py` | Auto-fill from provenance manifest; `[FILL]` elsewhere (no guessing) |
| RoB tool registry extensions | `data/rob_tools.json` | +PROBAST (4 domains, verified); +ROBINS-E (7 domains count-verified; names deliberately unfilled — fetch riskofbias.info before use) |
| Bootstrap self-repair | all entry scripts | Missing deps auto-install once (uv); `httpx`,`pytest`,`ruff` added to `_REQ_INSTALLS` |

## Performance (validated)

| Stage | Non-LLM (default) | LLM (--use-llm) | Notes |
|---|---|---|---|
| Discover | ~5s (parallel, S2 circuit breaker) | same | 4 sources in parallel, S2 force-skipped if not in --sources |
| Verify | ~5s (parallel 8 threads), ~1s (cached) | same | S2 circuit breaker (threshold=2), force_skip persists across phases |
| Extract | **0.1s** | ~5s/paper | Non-LLM default (regex + title fallback). Empty LLM → instant fallback |
| Assess | **0.3s** | 0.3s | Reads from extraction |
| Correlate | **2s** | ~3min | SVM stance default |
| Synthesize | **<1s** | ~11s | Template + interpretation enrichment / LLM smoothing (cached) |
| Meta | **1.5s** | 1.5s | Pure math |
| Export | **0.3s** | 0.3s | From PaperRecord |
| **Cached re-search** | **<1s** | <1s | Knowledge base HIT |

## Anti-fabrication checklist (before reporting done)

- [ ] Every citation in `research_brief.md` has a corresponding entry in `verified.json`
- [ ] Every entry in `verified.json` has `resolved: true`
- [ ] No retracted paper cited
- [ ] `verify_citation_alignment(brief, require_doi_match=True)` returns `[]` (body `[N]` ↔ References `[N]` aligned)
- [ ] Suggestions are advisory-only
- [ ] Matrix / forest plot / correlation cells trace to verified sources
- [ ] Limitations section notes unresolved papers, borderline cases

## Loop architecture (when to escalate)

| Phase | Trigger | Action |
|---|---|---|
| 2 → 1 | corpus < 10 papers | Re-scope with user |
| 3 → 2 | key seed paper unverifiable | Return to discovery |
| 4 stuck | contradictions unresolvable | Note in brief, proceed |
| any | >3 failures same task | Hand off to `/debugger` |

## Out of scope (delegate elsewhere)

| Need | Delegate to |
|---|---|
| Local PDF text extraction | `pdf-ocr` skill |
| RAG / QA over PDF corpus | PaperQA2 (GPU env) |
| Web research outside academic APIs | `duckduckgo-search` or `ctx_fetch_and_index` |

## Self-improvement loop

This skill is designed to be self-improving. During use, if the AI agent
detects any issues, it should act based on severity:

**Minor issues** (broken regex in `_citation_check.py`, stale source config
in `_sources.py`, ranking-heuristic gap in `_ranking.py`, screening
false-positive pattern, metadata-field drift from Crossref/OpenAlex/S2,
timeout/rate-limit tuning): the agent **should fix the skill itself
directly** in `~/.config/opencode/skills/scientific-research/scripts/`,
then re-run verification:

```bash
cd ~/.config/opencode/skills/scientific-research
uv run pytest -m "not live"                                        # unit/integration suite
uv run python scripts/verify.py research_outputs/corpus.json       # §5 H20 DOI gate
```

**User approval required** before applying any edit (per AGENTS.md §2
Approval Gate).

**Major failures or architectural issues** (§5 H20 DOI verification
bypass, `verify.py` output-schema regression, source API breaking change
requiring a new adapter, fabricate-citation vulnerability,
correlation/meta-analysis numerical regression): the agent **must STOP,
report the issue with full diagnosis** — what broke, why, blast radius,
evidence (logs/output), and a proposed architectural fix or improvement
with trade-offs. **Do not attempt a patchwork fix** (AGENTS.md §5 H8).
Wait for user decision before proceeding.

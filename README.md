# scientific-research

[![CI](https://github.com/santanu20/scientific-research/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/santanu20/scientific-research/actions/workflows/ci.yml)
[![License: MIT](https://github.com/santanu20/scientific-research/blob/master/LICENSE)](https://github.com/santanu20/scientific-research/blob/master/LICENSE)

A deep academic research pipeline as one CLI command. Ask a research question;
it discovers papers across six scholarly sources, verifies every identifier,
optionally reads full texts, cross-examines claims for contradictions, and
writes a cited research brief — with a web-synthesis supplement when its
sibling tool is installed. LLM stages are optional and run locally on Ollama.

Two ways to use it:

- **As a CLI** — one command, JSON out, artifacts on disk.
- **As an agent skill** — clone it into your AI agent's skills directory; the
  agent reads `SKILL.md` and drives the same pipeline.

## What it does

| Capability | Detail |
|---|---|
| Discovery | Crossref, OpenAlex, Semantic Scholar, arXiv, EarthArXiv, USGS in one sweep |
| Verification | Every DOI/arXiv ID resolved across multiple resolvers, retraction-aware; unverified papers dropped |
| Screening | PRISMA-style keep/drop with measured workload savings (WSS@95 benchmark included) |
| Full text (optional) | Bounded open-access PDF download, text extraction, OCR fallback for scanned pages |
| Cross-examination | Numeric claims extracted; contradictions detected via embedding clustering (BGE) |
| Citation graph | Reference tracking across the corpus |
| Synthesis | Narrative brief: themes, consensus vs. contrast, methods comparison |
| Critic loop | Optional LLM reflect round re-searches gaps it finds in its own coverage |
| Web supplement | Perplexity-style web section appended to the brief via the [web-search](https://github.com/santanu20/web-search) sibling |
| Graceful without LLM | No Ollama? Phases fall back to deterministic paths; the pipeline still completes |
| Budget-safe | Wall-clock budget enforced; on timeout you get partial results + `timed_out: true`, never a lost run |

## Who it's for

Grad students mapping a field, researchers scoping a review, anyone who needs
to know what the literature actually says (and disagrees about) before writing.
Also built for AI agents: one-JSON-doc output, timestamped artifact
directories, documented exit codes. Runs on any OS with Python (tested on
3.13, Linux); faster with a local Ollama endpoint.

## Install and run as a CLI

```bash
git clone https://github.com/santanu20/scientific-research
cd scientific-research
python scripts/pipeline.py "how does methane hydrate destabilization relate to submarine slope failures" --json
```

On first run the bootstrap creates `.venv/` and installs all dependencies
automatically. Output artifacts (corpus, verified papers, extractions,
correlation, brief, citations.bib) land in a timestamped
`research_outputs/pipeline-<date>-<slug>/` directory. The `--json` flag prints
`{query, research_type, n_papers, n_fulltext_matched, paths, elapsed}` for
scripting; without it you get a human summary. Progress goes to stderr.

Exit codes: `0` ok, `1` failure, `2` busy/usage error, `3` wall-clock budget
exceeded (partial artifacts kept, `timed_out: true` in JSON).

## Install as an agent skill

```bash
# opencode
git clone https://github.com/santanu20/scientific-research ~/.config/opencode/skills/scientific-research

# any other agent (Claude Code, etc.)
# clone into that agent's skills directory the same way
```

`SKILL.md` is the agent-facing surface: the 5-phase pipeline, every phase
script, flags, env toggles, and the output contract. First run bootstraps its
own venv — no manual setup.

Optional but recommended — install the sibling for web coverage:

```bash
git clone https://github.com/santanu20/web-search ~/.config/opencode/skills/web-search
```

Without it, the brief simply skips the web supplement.

## The pipeline

```
query ─> SCOPING ─> DISCOVERY (6 sources) ─> VERIFICATION (multi-resolver DOI/arXiv)
      ─> CORRELATION (claims + contradictions) ─> SYNTHESIS (brief + web supplement)
```

Live-measured end to end (2026-10-07, real network, 24-core CPU): 82 papers
discovered → 12 verified, full brief in 300 s; with web supplement 79 → 10
verified and a 24,000-char brief in 409 s.

## Useful flags

```text
--max N            papers to keep (default 30)
--use-llm          LLM stages on (local Ollama)
--from-year / --to-year / --open-access-only / --type
--reflect N        LLM critic re-search rounds (default 1, 0 = off)
--no-web-pro       drop the web-synthesis supplement
--max-pdf N        open-access PDF downloads (default 10, 0 = none)
--budget S         wall-clock budget in seconds
--out-dir D        custom output directory
--json             machine-readable result (add --with-brief to include text)
```

Run `python scripts/pipeline.py --help` for the full list. Individual phase
scripts (discover, verify, correlate, benchmark, risk-of-bias and
meta-analysis tooling, ...) live in `scripts/` and are also standalone-run;
see `SKILL.md`.

## Tests

```bash
.venv/bin/python -m pytest tests -m "not live" -q
```

Built and tested on Python 3.13. Network tests are marked `live` and excluded
by default.

## Related

- [web-search](https://github.com/santanu20/web-search) — metasearch CLI this
  pipeline uses for its web-synthesis supplement.

## License

MIT

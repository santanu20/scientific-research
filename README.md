# scientific-research

[![CI](https://github.com/santanu20/scientific-research/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/santanu20/scientific-research/actions/workflows/ci.yml)

Deep academic research pipeline as a CLI: literature discovery across Crossref,
OpenAlex, Semantic Scholar, arXiv, EarthArXiv and USGS, DOI/arXiv verification
(multi-resolver, retraction-aware), screening, optional open-access full-text
extraction (PDF + OCR fallback), claim/contradiction analysis, correlation, and
a synthesized research brief. Optional LLM stages run on a local Ollama
endpoint; everything degrades gracefully without one.

## Quickstart

```bash
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

## Useful flags

```text
--max N            papers to keep (default 30)
--use-llm          LLM stages on (local Ollama)
--from-year / --to-year / --open-access-only / --type
--reflect N        LLM critic re-search rounds (default 1, 0 = off)
--no-web-pro       drop the web-synthesis supplement
--budget S         wall-clock budget in seconds
--out-dir D        custom output directory
```

Run `python scripts/pipeline.py --help` for the full list. Individual phase
scripts (discover, verify, correlate, ...) live in `scripts/` and are also
standalone-run; see `SKILL.md`.

## Tests

```bash
.venv/bin/python -m pytest tests -m "not live" -q
```

Built and tested on Python 3.13. Network tests are marked `live` and excluded
by default.

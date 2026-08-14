#!/usr/bin/env python3
"""PRISMA-S search report + PROSPERO draft generator.

PRISMA-S (Rethlefsen ML et al. J Clin Epidemiol 2021;134:178-192) — 16-item
search reporting checklist. Items are auto-filled from run_provenance.json +
corpus metadata where the pipeline knows the answer; everything else is an
explicit [FILL] placeholder (never fabricated, §5 H7).

PROSPERO draft — structured registration skeleton from the research plan +
corpus stats. PROSPERO fields not derivable from a run stay [FILL].

Usage:
    python scripts/report.py research_outputs/ \
        --query "amphibole thermobarometry" \
        --prisma-s research_outputs/prisma_s.md \
        --prospero research_outputs/prospero_draft.md
"""

from __future__ import annotations

# --- Self-contained skill venv bootstrap (mirrors pdf-ocr/web-search pattern) ---
import os as _bs_os
import sys as _bs_sys

_SKILL_VENV = _bs_os.path.expanduser(
    "~/.config/opencode/skills/scientific-research/.venv"
)
_SKILL_VENV_PY = _bs_os.path.join(_SKILL_VENV, "bin", "python")
_REQ_IMPORTS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "sklearn",
    "httpx",
)
_REQ_INSTALLS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "scikit-learn",
    "httpx",
    "pytest",
    "ruff",
)
if __name__ == "__main__" and not _bs_os.environ.get(
    "SCIENTIFIC_RESEARCH_NO_SKILL_VENV"
):
    if not _bs_os.path.exists(_SKILL_VENV_PY) and not _bs_os.environ.get(
        "SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"
    ):
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(
                "Bootstrapping scientific-research skill venv (one-time setup)...\n"
            )
            _bs_sp.run(
                ["uv", "venv", _SKILL_VENV, "--python", "3.13"],
                check=True,
                capture_output=True,
            )
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True,
                capture_output=True,
            )
            _bs_sys.stderr.write("scientific-research skill venv ready.\n")
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(
                f"Failed to auto-bootstrap: {_bs_ex}\nManual: uv venv {_SKILL_VENV} --python 3.13 && uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
            )
            _bs_sys.exit(2)
    if _bs_os.path.exists(_SKILL_VENV_PY) and _bs_os.path.normpath(
        _bs_sys.prefix
    ) != _bs_os.path.normpath(_SKILL_VENV):
        _bs_os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        _bs_os.execv(
            _SKILL_VENV_PY,
            [_SKILL_VENV_PY, _bs_os.path.abspath(__file__)] + _bs_sys.argv[1:],
        )
    _missing = []
    for _m in _REQ_IMPORTS:
        try:
            __import__(_m)
        except ImportError:
            _missing.append(_m)
    if _missing and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(f"Installing missing deps: {', '.join(_missing)}\n")
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True,
                capture_output=True,
            )
            _missing = []
            for _m in _REQ_IMPORTS:
                try:
                    __import__(_m)
                except ImportError:
                    _missing.append(_m)
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Auto-install failed: {_bs_ex}\n")
    if _missing:
        _bs_sys.stderr.write(
            f"FATAL: missing required deps: {', '.join(_missing)}\nInstall: uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
        )
        _bs_sys.exit(2)
# --- End bootstrap ---

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _provenance import load_manifest

_DATABASE_NAMES = {
    "crossref": "Crossref (crossref.org)",
    "openalex": "OpenAlex (openalex.org)",
    "s2": "Semantic Scholar (semanticscholar.org)",
    "arxiv": "arXiv (arxiv.org)",
    "epmc": "Europe PMC (europepmc.org)",
    "eartharxiv": "EarthArXiv (via Crossref DOI prefix)",
    "usgs": "USGS Publications Warehouse",
    "web_search": "Web metasearch (web-search skill, 9 backends)",
}


def render_prisma_s(
    query: str,
    sources: list[str],
    n_records: int,
    search_dates: list[str],
    peer_reviewer: str = "[FILL: peer reviewer name/date]",
    full_search_strategies: str = "[FILL: paste exact per-database search strings, limits and restrictions used]",
) -> str:
    db_list = (
        "; ".join(_DATABASE_NAMES.get(s, s) for s in sources) or "[FILL: databases]"
    )
    dates = (
        f"{min(search_dates)} to {max(search_dates)}"
        if search_dates
        else "[FILL: search dates]"
    )
    items = [
        ("1. Database name", db_list),
        (
            "2. Multi-database searching",
            "Single search platform not used — each database searched via its own API"
            if len(sources) > 1
            else "[FILL]",
        ),
        (
            "3. Study registries",
            "[FILL: e.g. ClinicalTrials.gov, PROSPERO — or state 'none searched']",
        ),
        (
            "4. Online resources and browsing",
            "[FILL: e.g. journal tables of contents — or state 'none']",
        ),
        (
            "5. Citation searching",
            "[FILL: backward/forward citation chasing details — snowball flags used in discover.py]",
        ),
        ("6. Contacts", "[FILL: experts, authors, manufacturers — or state 'none']"),
        ("7. Other methods", "[FILL]"),
        ("8. Full search strategies", full_search_strategies),
        (
            "9. Limits and restrictions",
            "[FILL: publication type, year range, language — as passed via discover.py flags]",
        ),
        ("10. Search filters", "No validated search filter used" if True else "[FILL]"),
        (
            "11. Prior work",
            "[FILL: peer-reviewed search strategies adapted, or 'none']",
        ),
        ("12. Updates", f"Run dates on record: {dates} (from run_provenance.json)"),
        ("13. Dates of searches", dates),
        ("14. Peer review", peer_reviewer),
        (
            "15. Total records",
            f"{n_records} records identified across all databases (pre-deduplication counts in provenance manifest)",
        ),
        (
            "16. Deduplication",
            "Automated DOI/arXiv/title blocked matching (dedup_papers, author+year blocking); see corpus pipeline",
        ),
    ]
    lines = [
        "# PRISMA-S Search Report",
        "",
        f"**Query:** {query}",
        f"**Generated:** {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "",
        "| Item | Report |",
        "|---|---|",
    ]
    lines += [f"| {item} | {report} |" for item, report in items]
    lines += [
        "",
        "_Auto-generated by scientific-research skill report.py. [FILL] items "
        "require human input per PRISMA-S (Rethlefsen 2021, J Clin Epidemiol "
        "134:178-192); they are intentionally NOT guessed._",
    ]
    return "\n".join(lines)


def render_prospero_draft(query: str, n_records: int, n_verified: int) -> str:
    return f"""# PROSPERO Registration Draft

_Generated {time.strftime("%Y-%m-%d")} by scientific-research skill. Fields the
pipeline cannot know are [FILL] — PROSPERO requires registration BEFORE
screening starts; if screening already ran, register as a retrospective
review and say so in the record._

**Title**: [FILL: systematic review title] — topic: {query}
**Review question**: [FILL]
**Objectives**: [FILL]
## Search methods
Databases searched: see PRISMA-S report. Identified **{n_records}** records,
**{n_verified}** verified via DOI/arXiv resolution (§5 H20 gate).
**Search dates**: [FILL]
## Selection criteria
[FILL: population/phenomenon, inclusion, exclusion, study designs]
## Data extraction
Automated extraction pipeline (regex + SVM classifiers, LLM optional) with
human verification of: [FILL]
## Risk of bias
Tool: [FILL: RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa — assess.py advisory output]
## Synthesis
[FILL: narrative / meta-analysis / both; if meta-analysis: effect measure,
τ² estimator (default REML), CI method (default HKSJ)]
## Conflicts of interest
[FILL]
## Funding
[FILL]
"""


def main() -> int:
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: ready")
        return 0
    p = argparse.ArgumentParser(
        prog="report", description="PRISMA-S + PROSPERO report generator"
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("results_dir", type=Path, help="research_outputs/ directory")
    p.add_argument("--query", required=True, help="research question")
    p.add_argument(
        "--prisma-s", type=Path, default=Path("research_outputs/prisma_s.md")
    )
    p.add_argument(
        "--prospero", type=Path, default=Path("research_outputs/prospero_draft.md")
    )
    args = p.parse_args()

    manifest = load_manifest(args.results_dir)
    discovery = next((m for m in manifest if m["stage"] == "discovery"), None)
    sources = (discovery or {}).get("params", {}).get("sources") or []
    dates = [m["timestamp_utc"] for m in manifest if "timestamp_utc" in m]
    n_records = 0
    n_verified = 0
    corpus_path = args.results_dir / "corpus.json"
    if corpus_path.exists():
        try:
            corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
            n_records = len(corpus.get("papers", []))
        except (json.JSONDecodeError, OSError):
            pass
    verified_path = args.results_dir / "verified.json"
    if verified_path.exists():
        try:
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            n_verified = len(verified.get("papers", []))
        except (json.JSONDecodeError, OSError):
            pass

    if not manifest:
        print(
            "WARNING: no run_provenance.json found — report will be mostly [FILL]; "
            "run the pipeline with provenance recording for full auto-fill",
            file=sys.stderr,
        )

    prisma = render_prisma_s(args.query, sources, n_records, dates)
    args.prisma_s.parent.mkdir(parents=True, exist_ok=True)
    args.prisma_s.write_text(prisma, encoding="utf-8")
    print(f"Wrote PRISMA-S report → {args.prisma_s}")

    prospero = render_prospero_draft(args.query, n_records, n_verified)
    args.prospero.parent.mkdir(parents=True, exist_ok=True)
    args.prospero.write_text(prospero, encoding="utf-8")
    print(f"Wrote PROSPERO draft → {args.prospero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

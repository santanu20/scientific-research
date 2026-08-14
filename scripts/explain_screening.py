"""CLI: explain paper screening decisions for a corpus + query.

Reads a corpus.json (output of discover.py) and runs the LLM-only
screening pipeline against it, printing the per-paper decision breakdown
and stage-level statistics. Useful for:

  - Debugging "why was paper X excluded?" questions
  - Tuning screening thresholds before a real pipeline run
  - Auditing screening decisions per PRISMA 2020 (Page MJ et al. 2021 BMJ)

Usage:
    uv run python scripts/explain_screening.py \\
        research_outputs/corpus.json \\
        --query "amphibole thermometery"

Or against an extracted.json (post-extraction):
    uv run python scripts/explain_screening.py \\
        research_outputs/extracted.json \\
        --query "amphibole thermometery" \\
        --format extracted
"""


from __future__ import annotations

# --- Self-contained skill venv bootstrap (mirrors pdf-ocr/web-search pattern) ---
import os as _bs_os, sys as _bs_sys
_SKILL_VENV = _bs_os.path.expanduser("~/.config/opencode/skills/scientific-research/.venv")
_SKILL_VENV_PY = _bs_os.path.join(_SKILL_VENV, "bin", "python")
_REQ_IMPORTS = ("habanero", "pyalex", "semanticscholar", "arxiv", "numpy", "scipy", "sklearn", "httpx")
_REQ_INSTALLS = ("habanero", "pyalex", "semanticscholar", "arxiv", "numpy", "scipy", "scikit-learn", "httpx", "pytest", "ruff")
if __name__ == "__main__" and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_SKILL_VENV"):
    if not _bs_os.path.exists(_SKILL_VENV_PY) and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        import subprocess as _bs_sp
        try:
            _bs_sys.stderr.write("Bootstrapping scientific-research skill venv (one-time setup)...\n")
            _bs_sp.run(["uv", "venv", _SKILL_VENV, "--python", "3.13"], check=True, capture_output=True)
            _bs_sp.run(["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS], check=True, capture_output=True)
            _bs_sys.stderr.write("scientific-research skill venv ready.\n")
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Failed to auto-bootstrap: {_bs_ex}\nManual: uv venv {_SKILL_VENV} --python 3.13 && uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n")
            _bs_sys.exit(2)
    if _bs_os.path.exists(_SKILL_VENV_PY) and _bs_os.path.normpath(_bs_sys.prefix) != _bs_os.path.normpath(_SKILL_VENV):
        _bs_os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        _bs_os.execv(_SKILL_VENV_PY, [_SKILL_VENV_PY, _bs_os.path.abspath(__file__)] + _bs_sys.argv[1:])
    _missing = []
    for _m in _REQ_IMPORTS:
        try: __import__(_m)
        except ImportError: _missing.append(_m)
    if _missing and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        # stale venv: auto-install missing deps once, then re-check
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(f"Installing missing deps: {', '.join(_missing)}\n")
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True, capture_output=True,
            )
            _missing = []
            for _m in _REQ_IMPORTS:
                try: __import__(_m)
                except ImportError: _missing.append(_m)
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Auto-install failed: {_bs_ex}\n")
    if _missing:
        _bs_sys.stderr.write(f"FATAL: missing required deps: {', '.join(_missing)}\nInstall: uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n")
        _bs_sys.exit(2)
# --- End bootstrap ---



import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("explain_screening")


def _load_papers(path: Path, fmt: str) -> list:
    """Load papers from corpus.json or extracted.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if fmt == "auto":
        # corpus.json has {meta, papers: [...]}; extracted.json has {meta, extractions: [...]}
        if "papers" in data:
            fmt = "corpus"
        elif "extractions" in data:
            fmt = "extracted"
        else:
            raise ValueError(f"could not auto-detect format for {path}")
    if fmt == "corpus":
        return data.get("papers", [])
    if fmt == "extracted":
        return data.get("extractions", [])
    raise ValueError(f"unknown format: {fmt}")


class _PaperAdapter:
    """Adapt dict paper records to the PaperRecord-like interface screen_paper expects."""

    def __init__(self, d: dict) -> None:
        self.title = d.get("title", "") or ""
        self.abstract = d.get("abstract", "") or ""
        self.doi = d.get("doi", "") or ""
        self.primary_id = d.get("paper_id", "") or d.get("doi", "") or ""
        # Preserve raw_metadata.concepts if present (corpus.json has it)
        self.raw_metadata = d.get("raw_metadata", {}) or {}


def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="explain_screening",
        description="Explain paper screening decisions for a corpus + query.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--self-check", action="store_true",
                        help="verify deps + key imports, then exit 0")  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("corpus", type=Path, help="corpus.json or extracted.json path")
    p.add_argument(
        "--query",
        required=True,
        help="the research query (must match what the pipeline used)",
    )
    p.add_argument(
        "--format",
        choices=("auto", "corpus", "extracted"),
        default="auto",
        help="input file format (default: auto-detect)",
    )
    p.add_argument(
        "--show-excluded",
        action="store_true",
        help="print full details for each excluded paper",
    )
    p.add_argument(
        "--show-included",
        action="store_true",
        help="print full details for each included paper (default: summary only)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(levelname)-5s %(name)s: %(message)s",
    )

    # Make scripts/ importable (this file lives in scripts/)
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from screen_llm import screen_paper  # noqa: E402

    papers = _load_papers(args.corpus, args.format)
    if not papers:
        print(f"ERROR: no papers in {args.corpus}", file=sys.stderr)
        return 1

    print(f"Query: {args.query!r}")
    print(f"Corpus: {args.corpus} ({len(papers)} papers)")
    print()

    stage_counts: dict[str, int] = {}
    excluded_details: list[tuple[int, dict, str, str]] = []
    included_details: list[tuple[int, dict, str, str]] = []

    for i, p in enumerate(papers, start=1):
        adapter = _PaperAdapter(p)
        decision, stage, reason = screen_paper(adapter, args.query)
        if decision:
            included_details.append((i, p, stage, reason))
        else:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            excluded_details.append((i, p, stage, reason))

    # Summary
    n_excluded = len(excluded_details)
    n_included = len(included_details)
    print("=== SUMMARY ===")
    print(f"  Total:     {len(papers)}")
    print(f"  Included:  {n_included}")
    print(f"  Excluded:  {n_excluded}")
    if stage_counts:
        print(f"  By stage:  {', '.join(f'{s}={n}' for s, n in sorted(stage_counts.items()))}")
    print()

    # Excluded details
    if args.show_excluded and excluded_details:
        print(f"=== EXCLUDED ({n_excluded}) ===")
        for idx, p, stage, reason in excluded_details:
            title = (p.get("title") or "")[:75]
            doi = p.get("doi") or ""
            print(f"  [{idx:3d}] [{stage}] {title}")
            print(f"         doi: {doi}")
            print(f"         reason: {reason}")
            print()

    if args.show_included and included_details:
        print(f"=== INCLUDED ({n_included}) ===")
        for idx, p, stage, reason in included_details:
            title = (p.get("title") or "")[:75]
            print(f"  [{idx:3d}] [{stage}] {title}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

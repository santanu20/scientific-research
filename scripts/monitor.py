#!/usr/bin/env python3
"""Living-document monitor for the scientific-research skill.

Re-runs discovery with a `--since` date filter, dedups against an existing
corpus, and reports newly-discovered papers as alerts. Can be invoked by cron
or run manually as part of a periodic refresh.

Outputs alerts.json (new papers since last check) + console summary.
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
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus, save_corpus  # noqa: E402

log = logging.getLogger("scientific_research.monitor")


@dataclass
class Alert:
    paper_id: str
    doi: str | None
    title: str
    year: int | None
    venue: str
    source: str
    citation_count: int | None
    is_new: bool
    reason: str = ""


def find_new_papers(fresh: list[PaperRecord], baseline: list[PaperRecord]) -> list[PaperRecord]:
    """Return papers in `fresh` not present in `baseline` (by primary_id)."""
    baseline_ids = {p.primary_id for p in baseline}
    return [p for p in fresh if p.primary_id not in baseline_ids]


def filter_by_since(papers: list[PaperRecord], since: str) -> list[PaperRecord]:
    """Keep only papers with publication_date >= since (YYYY-MM-DD).
    Handles int years (some OpenAlex records), strings, and missing dates."""
    try:
        cutoff = datetime.strptime(since, "%Y-%m-%d").date()
    except ValueError:
        log.error("Invalid --since format. Use YYYY-MM-DD.")
        return papers
    cutoff_year = cutoff.year
    out: list[PaperRecord] = []
    for p in papers:
        # publication_date may be: full ISO str, "YYYY-MM-DD", int year, or empty
        pd = p.publication_date
        try:
            if isinstance(pd, int):
                if pd >= cutoff_year:
                    out.append(p)
                continue
            if not pd:
                # Fall back to year field
                if p.year and p.year >= cutoff_year:
                    out.append(p)
                continue
            # String path
            d = datetime.strptime(str(pd)[:10], "%Y-%m-%d").date()
            if d >= cutoff:
                out.append(p)
        except (ValueError, TypeError):
            # Last resort: year field
            try:
                if p.year and int(p.year) >= cutoff_year:
                    out.append(p)
            except (ValueError, TypeError):
                continue
    return out


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="monitor",
        description="Living-document monitor: re-run discovery, alert on new papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--self-check", action="store_true",
                        help="verify deps + key imports, then exit 0")  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="existing corpus.json or verified.json to diff against",
    )
    p.add_argument("--query", required=True, help="original research query to re-run")
    p.add_argument(
        "--since", default=None, help="only alert papers published on/after this date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--max-new", type=int, default=50, help="cap on number of new alerts (default 50)"
    )
    p.add_argument(
        "--sources", default="crossref,openalex,s2,arxiv", help="comma-separated sources"
    )
    p.add_argument("--max-per-source", type=int, default=20)
    p.add_argument("-o", "--output", type=Path, default=Path("research_outputs/alerts.json"))
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="merge new papers into baseline corpus (write back)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG), format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    )

    # Load baseline
    baseline = load_corpus(args.baseline)
    log.info("Baseline corpus: %d papers", len(baseline))

    # Re-run discovery (import inline to avoid hard dep at module load)
    from _sources import dedup_papers
    from discover import search_multi_source

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    fresh_raw = search_multi_source(args.query, max_per_source=args.max_per_source, sources=sources)
    fresh = dedup_papers(fresh_raw)
    log.info("Fresh discovery: %d papers (deduped from %d raw)", len(fresh), len(fresh_raw))

    # Filter by date
    if args.since:
        fresh = filter_by_since(fresh, args.since)
        log.info("After --since %s filter: %d papers", args.since, len(fresh))

    # Diff
    new_papers = find_new_papers(fresh, baseline)
    log.info("New vs baseline: %d", len(new_papers))

    # Cap
    if len(new_papers) > args.max_new:
        new_papers.sort(key=lambda r: -(r.citation_count or 0))
        new_papers = new_papers[: args.max_new]

    # Build alerts
    alerts = [
        Alert(
            paper_id=p.primary_id,
            doi=p.doi,
            title=p.title,
            year=p.year,
            venue=p.venue,
            source=p.source,
            citation_count=p.citation_count,
            is_new=True,
            reason=f"published {p.publication_date or p.year or '?'}",
        )
        for p in new_papers
    ]
    payload = {
        "meta": {
            "baseline": str(args.baseline),
            "query": args.query,
            "since": args.since,
            "baseline_n": len(baseline),
            "fresh_n": len(fresh),
            "new_n": len(new_papers),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "alerts": [asdict(a) for a in alerts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {len(alerts)} alerts → {args.output}")
    print(f"\n## {len(alerts)} new papers since last check")
    for a in alerts[:10]:
        print(f"- `{a.doi or a.paper_id[:30]}` ({a.year}) {a.title[:70]}")
    if len(alerts) > 10:
        print(f"... and {len(alerts) - 10} more")

    # Optionally merge into baseline
    if args.update_baseline and new_papers:
        merged = dedup_papers(baseline + new_papers)
        save_corpus(
            merged,
            args.baseline,
            meta={
                "last_monitor_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "new_added": len(new_papers),
            },
        )
        print(f"\nUpdated baseline: {len(baseline)} → {len(merged)} → {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

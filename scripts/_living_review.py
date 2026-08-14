"""Living review — incremental updates with versioned snapshots.

Supports continuous review updates:
1. Load previous corpus from data/research/<hash>/
2. Search for NEW papers since last run
3. Deduplicate against existing
4. Extract findings for new papers only
5. Merge with previous extractions
6. Re-synthesize brief
7. Save versioned snapshot with diff report
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("scientific_research.living_review")

_RESEARCH_DIR = Path("data/research")


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.lower().encode()).hexdigest()[:16]


def get_last_run_date(query: str) -> str | None:
    """Get the date of the last pipeline run for this query."""
    h = _query_hash(query)
    meta_file = _RESEARCH_DIR / h / "meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            return meta.get("run_date") or meta.get("timestamp")
        except Exception:
            pass
    return None


def load_previous_corpus(query: str) -> list[dict]:
    """Load papers from the previous run."""
    h = _query_hash(query)
    corpus_file = _RESEARCH_DIR / h / "corpus.json"
    if corpus_file.exists():
        try:
            data = json.loads(corpus_file.read_text())
            return data.get("papers", [])
        except Exception:
            pass
    return []


def find_new_papers(
    previous: list[dict],
    current: list[dict],
) -> list[dict]:
    """Identify papers in current that are NOT in previous."""
    prev_dois = {(p.get("doi") or "").lower() for p in previous if p.get("doi")}
    prev_titles = {(p.get("title") or "").lower().strip()[:100] for p in previous if p.get("title")}

    new_papers = []
    for p in current:
        doi = (p.get("doi") or "").lower()
        title = (p.get("title") or "").lower().strip()[:100]
        if doi and doi in prev_dois:
            continue
        if title and title in prev_titles:
            continue
        new_papers.append(p)

    log.info(
        "Living review: %d new papers (of %d current, %d previous)",
        len(new_papers),
        len(current),
        len(previous),
    )
    return new_papers


def save_versioned_snapshot(
    query: str,
    brief: str,
    corpus: list[dict],
    new_papers: list[dict],
    version: int = 1,
) -> Path:
    """Save a versioned snapshot of the review.

    Creates:
    - data/research/<hash>/versions/v{N}_brief.md
    - data/research/<hash>/versions/v{N}_diff.json
    """
    h = _query_hash(query)
    versions_dir = _RESEARCH_DIR / h / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    # Save versioned brief
    brief_file = versions_dir / f"v{version}_brief.md"
    brief_file.write_text(brief, encoding="utf-8")

    # Save diff
    diff = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_papers": len(corpus),
        "new_papers": len(new_papers),
        "new_dois": [p.get("doi") for p in new_papers if p.get("doi")],
        "new_titles": [p.get("title", "")[:100] for p in new_papers],
    }
    diff_file = versions_dir / f"v{version}_diff.json"
    diff_file.write_text(json.dumps(diff, indent=2), encoding="utf-8")

    log.info("Saved version %d: %d total, %d new", version, len(corpus), len(new_papers))
    return brief_file


def get_next_version(query: str) -> int:
    """Get the next version number for this query."""
    h = _query_hash(query)
    versions_dir = _RESEARCH_DIR / h / "versions"
    if not versions_dir.exists():
        return 1
    existing = [f.name for f in versions_dir.glob("v*_brief.md")]
    versions = []
    for name in existing:
        try:
            v = int(name.replace("v", "").replace("_brief.md", ""))
            versions.append(v)
        except ValueError:
            pass
    return max(versions) + 1 if versions else 1


def generate_diff_report(
    previous_brief: str | None,
    current_brief: str,
    new_papers: list[dict],
) -> str:
    """Generate a human-readable diff report.

    Shows:
    - Number of new papers since last update
    - Titles of new papers
    - Brief length change
    """
    lines = ["## Update summary", ""]

    if not previous_brief:
        lines.append("This is the first version of this review.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**New papers added**: {len(new_papers)}")
    lines.append(f"**Previous brief length**: {len(previous_brief)} chars")
    lines.append(f"**Updated brief length**: {len(current_brief)} chars")
    lines.append("")

    if new_papers:
        lines.append("**New papers since last update**:")
        lines.append("")
        for p in new_papers[:20]:
            title = (p.get("title") or "")[:80]
            year = p.get("year") or ""
            doi = p.get("doi") or ""
            lines.append(f"- {title} ({year}) {doi}")
        if len(new_papers) > 20:
            lines.append(f"- ... and {len(new_papers) - 20} more")
        lines.append("")

    return "\n".join(lines)

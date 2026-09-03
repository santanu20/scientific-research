"""Full-text matcher — match discovered papers to local markdown files.

NOTE: This module (research/_fulltext.py) is the LOCAL MARKDOWN MATCHER used
by pipeline.py. It is DISTINCT from research/scripts/_fulltext.py, which is
the PDF SECTION EXTRACTOR (OCR-based, skill-synced). Same name, different
purpose — do not merge or delete either.

When a paper from the web-discovered corpus exists in data/papers/*.md,
extract.py uses the full markdown text instead of just the abstract.
This gives 5-10x more data per paper (methods, results tables, discussion).

Match strategy (first hit wins):
1. DOI in markdown content (grep for "doi.org/10.xxx" or "DOI: 10.xxx")
2. Title fuzzy match (Jaccard overlap > 0.6 on significant title tokens)
3. No match → return None (use abstract-only)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_RE_DOI = re.compile(r"10\.\d{4,}/[^\s\"<>]+", re.IGNORECASE)


def _extract_doi_from_markdown(text: str) -> str | None:
    """Extract the first DOI from markdown text."""
    match = _RE_DOI.search(text[:2000])  # Check first 2000 chars (title + abstract area)
    if match:
        return match.group(0).rstrip(".,;)")
    return None


def _title_jaccard(title1: str, title2: str) -> float:
    """Jaccard similarity between two titles (word-level)."""
    _STOP = frozenset(
        {
            "the",
            "a",
            "an",
            "of",
            "in",
            "and",
            "or",
            "for",
            "to",
            "with",
            "from",
            "by",
            "on",
            "at",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "has",
            "have",
            "had",
            "this",
            "that",
            "these",
            "those",
        }
    )
    t1 = set(
        w.lower().strip(".,;:!?\"'()[]{}")
        for w in title1.split()
        if len(w) > 2 and w.lower() not in _STOP
    )
    t2 = set(
        w.lower().strip(".,;:!?\"'()[]{}")
        for w in title2.split()
        if len(w) > 2 and w.lower() not in _STOP
    )
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def build_local_index(papers_dir: Path, *, ocr_model: str | None = None) -> list[dict]:
    """Build an index of local papers (markdown + PDF) for fast matching.

    Scans both ``*.md`` and ``*.pdf`` files. PDFs are parsed via fitz
    (pymupdf).

    Returns list of dicts:
        {"path": Path, "doi": str | None, "title": str | None, "text": str,
         "filename": str, "source": "md"|"pdf", "is_matched": False}
    """
    if not papers_dir.exists():
        return []

    index: list[dict] = []

    # Markdown files (fast — already text)
    for md_file in sorted(papers_dir.rglob("*.md")):
        # Skip PDF sidecar caches — the PDF loop owns those (pre-existing
        # double-count: one PDF yielded both an 'md' and a 'pdf' entry
        # pointing at the same sidecar).
        if md_file.with_suffix(".pdf").exists():
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        doi = _extract_doi_from_markdown(text)
        title = None
        for line in text.split("\n")[:10]:
            line = line.strip()
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
            if len(line) > 10 and not line.startswith("!"):
                title = line
                break
        index.append(
            {
                "path": md_file,
                "doi": doi,
                "title": title,
                "filename": md_file.stem,
                "source": "md",
                "is_matched": False,
            }
        )

    # PDF files (parse via fitz or pdf_ocr; sidecar .md = parse cache)
    for pdf_file in sorted(papers_dir.rglob("*.pdf")):
        md_path = pdf_file.with_suffix(".md")
        # SIDECAR CACHE (2026-09-03): the .md written below IS the parse
        # cache — but it was never read back, so every research run
        # re-OCR'd the whole local library (live: 15-20 GPU-minutes of
        # vision OCR per run for off-topic scanned PDFs). Read it unless
        # the PDF is newer; one-time parse cost, then milliseconds.
        if md_path.exists() and md_path.stat().st_mtime >= pdf_file.stat().st_mtime:
            try:
                cached = md_path.read_text(encoding="utf-8", errors="replace")
                # strip the provenance header so title/DOI parsing is clean
                if cached.startswith("<!--"):
                    cached = cached.split("-->", 1)[-1].lstrip()
            except OSError:
                cached = ""
            if len(cached.strip()) >= 200:
                index.append(
                    {
                        "path": md_path,
                        "doi": _extract_doi_from_markdown(cached),
                        "title": _extract_title_from_pdf(cached),
                        "filename": pdf_file.stem,
                        "source": "pdf",
                        "is_matched": False,
                    }
                )
                continue
        text = _parse_pdf_to_text(pdf_file, ocr_model=ocr_model)
        doi = _extract_doi_from_markdown(text)
        title = _extract_title_from_pdf(text)
        # Store as .md alongside for downstream consumers — rewrite when
        # stale (a pre-existing guard only wrote when ABSENT, so an
        # outdated sidecar lived forever)
        if (
            not md_path.exists()
            or md_path.stat().st_mtime < pdf_file.stat().st_mtime
        ):
            header = f"<!-- Source: PDF {pdf_file.name} -->\n\n"
            md_path.write_text(header + text[:50000], encoding="utf-8")
        index.append(
            {
                "path": md_path,
                "doi": doi,
                "title": title,
                "filename": pdf_file.stem,
                "source": "pdf",
                "is_matched": False,
            }
        )

    log.info(
        "Local paper index: %d files in %s (%d MD, %d PDF)",
        len(index),
        papers_dir,
        sum(1 for e in index if e["source"] == "md"),
        sum(1 for e in index if e["source"] == "pdf"),
    )
    return index


def _parse_pdf_to_text(pdf_path: Path, *, ocr_model: str | None = None) -> str:
    """Parse PDF to text via fitz (fast) or pdf_ocr (OCR quality)."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        if len(text.strip()) > 200:
            return text
    except Exception:
        log.debug("suppressed Exception in _fulltext.py", exc_info=True)
    # Fallback: pdf_ocr
    try:
        import sys

        if str(ocr_pkg.parent) not in sys.path:
            sys.path.insert(0, str(ocr_pkg.parent))
        raise ImportError("standalone skill: fitz handles text extraction")

        # Page-capped OCR (2026-09-03): research full-text needs the paper's
        # core (title/abstract/intro/methods), not a 300-page monograph.
        # Wall-clock caps alone let the server-side generation outlive the
        # client timeout and wedge the GPU runner (live incident 2026-09-02).
        _OCR_PAGE_CAP = "1-30"
        doc = (
            ocr_parse(pdf_path, fast=True, models=ocr_model, pages=_OCR_PAGE_CAP)
            if ocr_model
            else ocr_parse(pdf_path, fast=True, pages=_OCR_PAGE_CAP)
        )
        return doc.full_text
    except Exception as e:
        log.warning("PDF parse failed for %s: %s", pdf_path.name, e)
        return ""


def _extract_title_from_pdf(text: str) -> str | None:
    """Extract a title from the first meaningful lines of PDF text."""
    for line in text.split("\n")[:15]:
        line = line.strip()
        if len(line) > 15 and not line.isdigit() and "abstract" not in line.lower():
            return line[:200]
    return None


def match_paper_to_local(
    paper_doi: str | None,
    paper_title: str | None,
    local_index: list[dict],
    min_title_score: float = 0.5,
) -> Path | None:
    """Match a corpus paper to a local markdown file.

    Match order:
    1. DOI exact match (case-insensitive)
    2. Title fuzzy match (Jaccard >= min_title_score)
    3. No match → None

    Returns Path to the markdown file, or None.
    """
    # 1. DOI match
    if paper_doi:
        doi_lower = paper_doi.lower().rstrip(".,;)")
        for entry in local_index:
            if entry["doi"] and entry["doi"].lower().rstrip(".,;)") == doi_lower:
                log.debug("DOI match: %s → %s", paper_doi, entry["path"].name)
                return entry["path"]

    # 2. Title match
    if paper_title:
        best_score = 0.0
        best_path: Path | None = None
        for entry in local_index:
            if entry["title"]:
                score = _title_jaccard(paper_title, entry["title"])
                if score > best_score:
                    best_score = score
                    best_path = entry["path"]
        if best_score >= min_title_score:
            log.debug(
                "Title match (%.2f): '%s' → %s",
                best_score,
                paper_title[:40],
                best_path.name if best_path else "?",
            )
            return best_path

    return None


def load_fulltext(path: Path) -> str:
    """Load full markdown text from a local paper file."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        log.warning("Failed to read %s: %s", path, e)
        return ""


def get_local_paper_count(papers_dir: Path) -> int:
    """Count markdown files in the papers directory."""
    if not papers_dir.exists():
        return 0
    return len(list(papers_dir.rglob("*.md")))

"""Full-text PDF section extraction for the research pipeline.

NOTE: This module (research/scripts/_fulltext.py) is the PDF SECTION
EXTRACTOR — OCR-quality extraction + section splitting. It is DISTINCT from
research/_fulltext.py, which is the LOCAL MARKDOWN MATCHER used by
pipeline.py. Same name, different purpose — do not merge or delete either.
This copy is synced with the scientific-research skill; after any edit run
tests/architecture/test_research_scripts_guard.py.

Leverages geokit.pdf_ocr for OCR-quality text extraction, then splits
the text into structured sections (Methods, Results, Discussion, etc.).

When an open-access PDF is available, this module enriches papers with:
- methods_text — analytical procedures, calibration details
- results_text — quantitative data, measurements
- discussion_text — interpretations, limitations
- tables_text — tabulated data extracted by pymupdf

Papers without OA PDFs are unchanged — full-text enrichment is optional.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("scientific_research.fulltext")

_CACHE_DIR = Path.home() / ".cache" / "scientific_research" / "pdfs"

# Section heading patterns — case-insensitive, matches numbered or unnumbered
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "introduction",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:introduction|background|previous\s+work|literature\s+review)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "methods",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:materials?\s+and\s+methods?|methods?|methodology|"
            r"analytical\s+(?:methods?|procedures?)|experimental(?:\s+(?:methods?|section|procedures?))?|"
            r"sample\s+preparation|geological\s+setting|study\s+area)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "results",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:results?|findings?|data\s+analysis|"
            r"analytical\s+results?|geochemical\s+(?:results?|data))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "discussion",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:discussion|interpretation|implications?|"
            r"comparison\s+with)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusions",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:conclusions?|summary|outlook|closing\s+remarks?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "references",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:references?|bibliography|literature\s+cited)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "acknowledgments",
        re.compile(
            r"(?:^|\n)\s*(?:\d+\.?\s*)?(?:acknowledg(?:e)?ments?|funding)\b",
            re.IGNORECASE,
        ),
    ),
]


def _doi_hash(doi: str) -> str:
    """Content-hash from DOI for cache file naming."""
    return hashlib.sha256(doi.lower().encode()).hexdigest()[:32]


def download_pdf(
    doi: str | None,
    oa_url: str | None,
    timeout: float = 30.0,
) -> Path | None:
    """Download an open-access PDF, using cache when available.

    Args:
        doi: Paper DOI (for cache key).
        oa_url: Direct PDF URL (from OpenAlex best_oa_location or Unpaywall).
        timeout: Download timeout in seconds.

    Returns:
        Path to downloaded PDF, or None if unavailable.
    """
    if not doi or not oa_url:
        return None

    cache_key = _doi_hash(doi)
    cache_file = _CACHE_DIR / f"{cache_key}.pdf"
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        return cache_file

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            oa_url,
            headers={"User-Agent": "Mozilla/5.0 (research pipeline)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 1000:
                log.debug("PDF too small for %s: %d bytes", doi, len(data))
                return None
            cache_file.write_bytes(data)
            log.info("Downloaded OA PDF for %s (%d KB)", doi, len(data) // 1024)
            return cache_file
    except Exception as e:
        log.debug("PDF download failed for %s: %s", doi, e)
        return None


def extract_sections(full_text: str) -> dict[str, str]:
    """Split full text into structured sections by heading detection.

    Args:
        full_text: Complete paper text from pdf_ocr.parse().full_text or fitz.

    Returns:
        Dict mapping section names to their text content:
        {"methods": "...", "results": "...", "discussion": "...", "conclusions": "..."}
    """
    if not full_text or len(full_text) < 200:
        return {}

    sections: dict[str, str] = {}

    # Find all section heading positions
    matches: list[tuple[int, str, int]] = []  # (start, section_name, heading_length)

    for section_name, pattern in _SECTION_PATTERNS:
        for m in pattern.finditer(full_text):
            matches.append((m.start(), section_name, len(m.group(0))))

    if len(matches) < 2:
        # Not enough structure — return the whole text as "body"
        return {"body": full_text}

    # Sort by position
    matches.sort(key=lambda x: x[0])

    # Extract text between consecutive headings
    for i, (start, name, _) in enumerate(matches):
        if name in ("references", "acknowledgments"):
            continue  # Don't include these sections

        if i + 1 < len(matches):
            end = matches[i + 1][0]
        else:
            end = len(full_text)

        section_text = full_text[start:end].strip()
        if len(section_text) > 50:
            # Merge with existing section if same name appears multiple times
            if name in sections:
                sections[name] += "\n\n" + section_text
            else:
                sections[name] = section_text

    return sections


def parse_pdf_with_fitz(pdf_path: Path) -> str:
    """Fast text extraction using pymupdf (no OCR fallback).

    Used when the PDF has a text layer. Falls back to empty string on failure.
    """
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        log.debug("fitz extraction failed: %s", e)
        return ""


def parse_pdf_with_ocr(pdf_path: Path, timeout: int = 300) -> str:
    """OCR-quality extraction using geokit.pdf_ocr.

    Handles scanned PDFs and image-based content.
    Falls back to empty string on failure.
    """
    try:
        # Import from geokit package
        import sys

        geokit_src = Path(__file__).resolve().parents[3] / "src"
        if str(geokit_src) not in sys.path:
            sys.path.insert(0, str(geokit_src))

        from geokit.pdf_ocr import parse as ocr_parse

        result = ocr_parse(pdf_path, fast=True, timeout_s=timeout)
        return result.full_text
    except Exception as e:
        log.debug("OCR extraction failed for %s: %s", pdf_path, e)
        return ""


def extract_tables_fitz(pdf_path: Path) -> list[str]:
    """Extract tables from PDF using pymupdf's table detection.

    Returns list of tables rendered as markdown strings.
    """
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        tables = []
        for page_num, page in enumerate(doc):
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    # Convert to markdown
                    rows = tab.extract()
                    if rows and len(rows) > 1:
                        md_rows = [
                            "| " + " | ".join(str(c or "") for c in row) + " |"
                            for row in rows
                        ]
                        header_sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
                        md_table = "\n".join([md_rows[0], header_sep] + md_rows[1:])
                        tables.append(f"<!-- Page {page_num + 1} -->\n{md_table}")
            except Exception:
                pass
        doc.close()
        return tables
    except Exception as e:
        log.debug("Table extraction failed: %s", e)
        return []


def enrich_paper(
    paper: dict,
    pdf_path: Path | None = None,
) -> dict:
    """Enrich a paper dict with full-text sections.

    Args:
        paper: Paper dict with at least 'doi' and optionally 'oa_pdf_url'.
        pdf_path: Pre-downloaded PDF path. If None, will try to download.

    Returns:
        The same paper dict with added keys:
        - methods_text, results_text, discussion_text, conclusions_text
        - full_text (complete text)
        - tables (list of markdown table strings)
        - has_fulltext (bool)
    """
    doi = paper.get("doi") or ""
    oa_url = paper.get("oa_pdf_url") or ""

    # Download PDF if not provided
    if pdf_path is None and (doi or oa_url):
        pdf_path = download_pdf(doi, oa_url if oa_url else None)

    if pdf_path is None or not pdf_path.exists():
        paper["has_fulltext"] = False
        return paper

    # Try fast fitz extraction first
    full_text = parse_pdf_with_fitz(pdf_path)

    # If fitz got too little text, try OCR
    if len(full_text) < 500:
        log.debug("Text layer thin for %s, trying OCR", doi)
        ocr_text = parse_pdf_with_ocr(pdf_path)
        if len(ocr_text) > len(full_text):
            full_text = ocr_text

    if not full_text or len(full_text) < 200:
        paper["has_fulltext"] = False
        return paper

    # Extract sections
    sections = extract_sections(full_text)

    # Add sections to paper dict. NO TRUNCATION by default — full text is full.
    # Caps are configurable via env vars only when memory pressure is real.
    _FT_CAP = int(os.environ.get("SCIENTIFIC_RESEARCH_FULLTEXT_CAP", "0"))
    _SEC_CAP = int(os.environ.get("SCIENTIFIC_RESEARCH_SECTION_CAP", "0"))
    _TABLES_CAP = int(os.environ.get("SCIENTIFIC_RESEARCH_TABLES_CAP", "0"))

    def _cap(s: str, limit: int) -> str:
        return s[:limit] if limit > 0 else s

    paper["full_text"] = _cap(full_text, _FT_CAP)
    paper["methods_text"] = _cap(sections.get("methods", ""), _SEC_CAP)
    paper["results_text"] = _cap(sections.get("results", ""), _SEC_CAP)
    paper["discussion_text"] = _cap(sections.get("discussion", ""), _SEC_CAP)
    paper["conclusions_text"] = _cap(sections.get("conclusions", ""), _SEC_CAP)
    paper["has_fulltext"] = True

    # Extract tables — no cap by default
    tables = extract_tables_fitz(pdf_path)
    if tables:
        capped_tables = tables[:_TABLES_CAP] if _TABLES_CAP > 0 else tables
        paper["tables"] = capped_tables
        paper["tables_text"] = "\n\n".join(capped_tables)

    log.info(
        "Enriched %s: %d chars text, %d sections, %d tables",
        doi or paper.get("title", "?")[:30],
        len(full_text),
        len(sections),
        len(tables),
    )

    return paper


def enrich_papers(
    papers: list[dict],
    max_downloads: int = 20,
    timeout_per_download: float = 30.0,
) -> tuple[int, int]:
    """Enrich a batch of papers with full-text sections.

    Only processes papers with OA PDF URLs. Non-OA papers are unchanged.

    Args:
        papers: List of paper dicts.
        max_downloads: Maximum PDFs to download (prevents rate limiting).
        timeout_per_download: Timeout for each PDF download.

    Returns:
        (n_enriched, n_failed)
    """
    n_enriched = 0
    n_failed = 0
    downloads = 0

    for paper in papers:
        if downloads >= max_downloads:
            log.info("Full-text enrichment: reached max downloads (%d)", max_downloads)
            break

        doi = paper.get("doi") or ""
        oa_url = paper.get("oa_pdf_url") or ""

        if not doi or not oa_url:
            paper["has_fulltext"] = False
            continue

        # Skip if already enriched
        if paper.get("has_fulltext"):
            continue

        try:
            downloads += 1
            enrich_paper(paper)
            if paper.get("has_fulltext"):
                n_enriched += 1
            else:
                n_failed += 1
        except Exception as e:
            log.debug("Enrichment failed for %s: %s", doi, e)
            n_failed += 1
            paper["has_fulltext"] = False

        # Rate limit
        time.sleep(0.5)

    log.info(
        "Full-text enrichment: %d enriched, %d failed, %d skipped (no OA)",
        n_enriched,
        n_failed,
        len(papers) - n_enriched - n_failed,
    )
    return n_enriched, n_failed

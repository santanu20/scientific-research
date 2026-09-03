"""Full-text PDF section extraction for the research pipeline.

NOTE: This module (research/scripts/_fulltext.py) is the PDF SECTION
EXTRACTOR — OCR-quality extraction + section splitting. It is DISTINCT from
research/_fulltext.py, which is the LOCAL MARKDOWN MATCHER used by
pipeline.py. Same name, different purpose — do not merge or delete either.
This copy is synced with the scientific-research skill; after any edit run
tests/architecture/test_research_scripts_guard.py.

Extracts text via PyMuPDF, then splits
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
import re
import time
import urllib.request
from pathlib import Path

from _timeouts import TIMEOUTS  # noqa: E402

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
    if not doi:
        return None

    # If no OA URL provided, query Unpaywall to find one
    if not oa_url:
        try:
            import json as _json

            uw_resp = urllib.request.urlopen(
                f"https://api.unpaywall.org/v2/{doi}?email=research-skill@localhost",
                timeout=TIMEOUTS.unpaywall,
            )
            uw_data = _json.loads(uw_resp.read().decode())
            if uw_data.get("is_oa"):
                # Prefer REPOSITORY locations (universities, arXiv, CORE) —
                # publisher sites block programmatic access (403).
                repo_urls = [
                    (loc.get("url_for_pdf") or loc.get("url"))
                    for loc in uw_data.get("oa_locations", [])
                    if loc.get("host_type") == "repository"
                    and loc.get("url_for_pdf")
                    and "doi.org" not in (loc.get("url_for_pdf") or "")
                ]
                pub_urls = [
                    (loc.get("url_for_pdf") or loc.get("url"))
                    for loc in uw_data.get("oa_locations", [])
                    if loc.get("host_type") == "publisher"
                    and loc.get("url_for_pdf")
                    and "doi.org" not in (loc.get("url_for_pdf") or "")
                ]
                # Try repositories first, then publishers as fallback
                for url in repo_urls + pub_urls:
                    oa_url = url
                    break
                if oa_url:
                    log.debug("Unpaywall found OA PDF for %s: %s", doi, oa_url[:60])
        except Exception:
            log.debug("Optional feature failed in _fulltext.py", exc_info=True)

    if not oa_url:
        return None

    cache_key = _doi_hash(doi)
    cache_file = _CACHE_DIR / f"{cache_key}.pdf"
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        return cache_file

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Use requests for better redirect/cookie handling
        import requests as _requests

        resp = _requests.get(
            oa_url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code == 200 and len(resp.content) > 5000:
            # Verify it's actually a PDF (publishers return HTML error pages)
            if resp.content[:4] != b"%PDF":
                log.debug("Download for %s is not a PDF (got %s)", doi, resp.content[:20])
                return None
            cache_file.write_bytes(resp.content)
            log.info("Downloaded OA PDF for %s (%d KB)", doi, len(resp.content) // 1024)
            return cache_file
        else:
            log.debug("PDF download returned %d for %s", resp.status_code, doi)
            return None
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
    """PDF text extraction (PyMuPDF), content-hash cached.

    Handles scanned PDFs and image-based content. Falls back to empty
    string on failure. Cross-run cache (2026-09-03): research runs
    re-download and re-OCR the SAME OA PDFs — cache keyed by (path,
    mtime, size, page-cap) under ~/.cache/scientific-research/fulltext_ocr
    makes the vision-OCR cost one-time per document version.
    """
    import hashlib

    _OCR_PAGE_CAP = "1-30"  # GPU-wedge mitigation (wall-clock caps alone
    # let server-side generation outlive the client timeout)
    cache_dir = Path.home() / ".cache" / "scientific-research" / "fulltext_ocr"
    try:
        st = pdf_path.stat()
        key = hashlib.sha256(
            f"{pdf_path.resolve()}:{st.st_mtime_ns}:{st.st_size}:{_OCR_PAGE_CAP}".encode()
        ).hexdigest()[:24]
        cache_path = cache_dir / f"{key}.txt"
        if cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            if cached.strip():
                return cached
    except OSError:
        cache_path = None  # type: ignore[assignment]

    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        pages = doc[:30]  # page cap: bounded work per document
        text = "\n\n".join(page.get_text() for page in pages)
        doc.close()
        if text and text.strip():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(text, encoding="utf-8")  # type: ignore[union-attr]
            except OSError:
                log.debug("OCR cache write failed for %s", pdf_path)
        return text
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
                            "| " + " | ".join(str(c or "") for c in row) + " |" for row in rows
                        ]
                        header_sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
                        md_table = "\n".join([md_rows[0], header_sep] + md_rows[1:])
                        tables.append(f"<!-- Page {page_num + 1} -->\n{md_table}")
            except Exception:
                log.debug("Optional feature failed in _fulltext.py", exc_info=True)
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

    # Primary: pdf-ocr — SOTA routing (pdftotext fast path for born-digital,
    # GLM-OCR for scanned). Handles chemistry subscripts (H₂O→H$_{2}$),
    # math regions, fragmented tables per AGENTS.md scientific accuracy.
    full_text = parse_pdf_with_ocr(pdf_path)

    # Fallback: fitz if OCR unavailable (Ollama down) or returned too little
    if len(full_text) < 200:
        log.debug("OCR thin for %s, trying fitz fallback", doi)
        fitz_text = parse_pdf_with_fitz(pdf_path)
        if len(fitz_text) > len(full_text):
            full_text = fitz_text

    if not full_text or len(full_text) < 200:
        paper["has_fulltext"] = False
        return paper

    # Extract sections
    sections = extract_sections(full_text)

    # Add sections to paper dict
    paper["full_text"] = full_text[:20000]  # cap to prevent memory issues
    paper["methods_text"] = sections.get("methods", "")[:5000]
    paper["results_text"] = sections.get("results", "")[:5000]
    paper["discussion_text"] = sections.get("discussion", "")[:5000]
    paper["conclusions_text"] = sections.get("conclusions", "")[:3000]
    paper["has_fulltext"] = True

    # Extract tables
    tables = extract_tables_fitz(pdf_path)
    if tables:
        paper["tables"] = tables[:10]  # cap at 10 tables
        paper["tables_text"] = "\n\n".join(tables[:5])[:5000]

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

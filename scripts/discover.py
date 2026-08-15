#!/usr/bin/env python3
"""Discovery tool for the scientific-research skill.

Multi-source web search across Crossref, OpenAlex, Semantic Scholar, arXiv.
Cohen 2018 snowballing (forward + backward citation chasing).
Deduplication by DOI / arXiv ID / title similarity.

For local PDF extraction, use the dedicated pdf-ocr skill — this script is
web-research only.

Outputs corpus.json suitable for screen.py / verify.py / correlate.py.

References:
    Cohen JF et al. Citation searching and snowballing. J Clin Epidemiol. 2018
        — verified snowballing finds ~2x more relevant papers than DB search alone.

Usage:
    discover.py "machine learning climate" --max 50
    discover.py "PICO query" --sources crossref,openalex
    discover.py "query" --snowball 1
    discover.py --explore "topic landscape"
    discover.py --build-query "P=population I=intervention O=outcome"
"""

from __future__ import annotations

# --- Skill venv bootstrap (shared: see _bootstrap.py) ---
if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()
# --- End bootstrap ---


import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# Make _sources importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from _nlp import (
    expand_query as _expand_query_terms,
)
from _sources import (
    PaperRecord,
    arxiv_search,
    cache_get,
    cache_has,
    cache_put,
    crossref_search,
    dedup_papers,
    openalex_get_by_id,
    openalex_get_cited_by,
    openalex_search,
    openalex_semantic_search,
    render_compact_summary,
    s2_get_references,
    s2_search,
    save_corpus,
)

log = logging.getLogger("scientific_research.discover")

# S2 search + references are tightly rate-limited (separate bucket from
# paper/batch lookup). Drop S2 as a discovery source — use Crossref for
# search. S2 is still used for batch DOI enrichment (Phase 1.5) and
# forward citations (Phase 4) which have a generous separate rate limit.
DEFAULT_SOURCES = "openalex,crossref"
DEFAULT_MAX_PER_SOURCE = 50  # per source; total ~100-150 after dedup

# =============================================================================
# Geological discovery enhancement — journal-targeted Crossref search +
# geological synonym query variants + web-search skill for paper DOIs.
# These are ADDITIONAL parallel sources (not query dilution) — each runs
# independently, results are merged + deduplicated.
# =============================================================================

_GEO_JOURNAL_ISSNS = [
    # Traditional subscription journals (high-impact)
    "0016-7037",  # Geochimica et Cosmochimica Acta
    "0010-7999",  # Contributions to Mineralogy and Petrology
    "0022-3530",  # Journal of Petrology
    "0263-4929",  # Journal of Metamorphic Geology
    "0024-4937",  # Lithos
    "0009-2541",  # Chemical Geology
    "0003-004X",  # American Mineralogist
    "0012-821X",  # Earth and Planetary Science Letters
    "0301-9268",  # Precambrian Research
    "1342-937X",  # Gondwana Research
    "0169-1368",  # Ore Geology Reviews
    "0361-0128",  # Economic Geology
    "0026-4598",  # Mineralium Deposita
    "1674-9871",  # Journal of Earth Science
    "0024-4949",  # Physics and Chemistry of Minerals
    # Open-access earth science journals
    "1869-9510",  # Solid Earth (EGU, fully OA)
    "2611-4244",  # Geoscience Communication (EGU, OA)
    "1866-3516",  # Earth System Science Data (EGU, OA)
    "1991-9603",  # Geoscientific Model Development (EGU, OA)
    "2296-6463",  # Frontiers in Earth Science (OA)
    "2075-163X",  # Minerals (MDPI, OA)
    "2076-3263",  # Geosciences (MDPI, OA)
    "2045-2322",  # Scientific Reports (Nature, OA)
    "1932-6203",  # PLOS ONE (OA)
    "2391-5447",  # Open Geosciences (De Gruyter, OA)
    "2365-5763",  # Acta Geochimica (Springer, OA)
    "1525-2027",  # Geochemistry, Geophysics, Geosystems (AGU)
    "1814-8241",  # Elements (Mineralogical Society)
    "0040-1951",  # Tectonophysics
    "0191-8141",  # Journal of Structural Geology
    "1467-4866",  # Geochemical Transactions (Springer, OA)
    "2472-3452",  # ACS Earth and Space Chemistry
    # Additional OA + high-impact earth science journals
    "2333-5084",  # Earth and Space Science (AGU, OA)
    "2169-9313",  # JGR: Solid Earth (AGU)
    "0956-540X",  # Geophysical Journal International
    "0377-0273",  # Journal of Volcanology and Geothermal Research
    "0258-8900",  # Bulletin of Volcanology
    "0883-2927",  # Applied Geochemistry
    "1866-6280",  # Environmental Earth Sciences (Springer)
    "1553-040X",  # Geosphere (GSA)
    "2586-1132",  # Episodes (IUGS, OA)
    "1885-7971",  # Geologica Acta (OA)
    "2603-4193",  # Journal of Iberian Geology (OA)
    "2572-4525",  # Paleoceanography and Paleoclimatology (AGU)
    "2328-4277",  # Earth's Future (AGU, OA)
    "2471-1403",  # GeoHealth (AGU, OA)
    "2190-4979",  # Earth System Dynamics (EGU, OA)
    "0992-7689",  # Annales Geophysicae (EGU, OA)
    "1561-8633",  # Natural Hazards and Earth System Sciences (EGU, OA)
    "0025-3227",  # Marine Geology
    "2698-5501",  # Geochronology (Copernicus, OA)
    "0016-7606",  # GSA Bulletin
    "0091-7613",  # Geology (GSA)
    "0305-8719",  # Geological Society of London Special Publications
    "1480-3291",  # Canadian Journal of Earth Sciences
    "0812-0099",  # Australian Journal of Earth Sciences
    "0315-0941",  # Geoscience Canada (GAC, OA)
]

# Augment with Crossref-discovered journals (dynamic, ~600 ISSNs)
try:
    from _geodict import get_geo_journals

    _DYNAMIC_JOURNALS = get_geo_journals()
    _GEO_JOURNAL_ISSNS = sorted(set(_GEO_JOURNAL_ISSNS + _DYNAMIC_JOURNALS))
except Exception:
    pass  # keep curated list only

_GEO_SYNONYM_MAP: dict[str, str] = {
    "thermometry": "geothermometry temperature calibration",
    "barometry": "geobarometry pressure calibration",
    "thermobarometry": "geothermobarometry P-T estimation",
    "arc magma": "island arc basalt andesite volcanic arc",
    "ree": "rare earth element REE geochemistry",
    "lile": "large ion lithophile element",
    "hfse": "high field strength element",
    "porphyry copper": "porphyry Cu Mo hydrothermal deposit",
    "alteration": "hydrothermal alteration mineralization",
    "metamorphic": "metamorphism facies grade",
    "partial melting": "anatexis melt generation mantle crust",
    "garnet": "garnet pelitic schist eclogite",
    "biotite": "biotite Fe-Mg exchange mineral",
    "clinopyroxene": "clinopyroxene cpx pyroxene",
    "lherzolite": "lherzolite peridotite mantle xenolith",
    "subduction": "subduction zone slab dehydration fluid",
    "metasomatism": "metasomatism fluid-rock interaction",
    "partition coefficient": "partition coefficient KD distribution",
    "geotherm": "geotherm geothermal gradient heat flow",
}


def _geo_synonym_variant(query: str) -> str | None:
    """Build a geological synonym-expanded query variant.

    Returns None if the query has no geological synonym matches.
    Unlike _expand_search_query (which avoids expansion to prevent
    dilution), this creates a SEPARATE query variant that runs as its
    own parallel Crossref search — results are merged, not substituted.
    """
    q_lower = query.lower()
    additions: list[str] = []
    for term, expansion in _GEO_SYNONYM_MAP.items():
        if term in q_lower:
            # Only add terms NOT already in the query
            for word in expansion.split():
                if word.lower() not in q_lower:
                    additions.append(word)
    if not additions:
        return None
    # Cap at 8 additional words to avoid over-expansion
    return query + " " + " ".join(additions[:8])


# =============================================================================
# Tiered keyword extraction + boolean AND + co-occurrence filter
# (cherry-picked from geokit discover.py — field-agnostic, no domain term lists)
# =============================================================================


def _extract_key_terms(query: str) -> list[str]:
    """Extract key search terms, tiered by importance.

    Tier 1 (REQUIRED — strict AND): proper nouns (capitalized words)
        e.g. "Dongargarh", "Bushveld" — these MUST be present
    Tier 2 (IMPORTANT — should be present): domain-specific nouns
        e.g. "granite", "isotope", "thermometry" — at least ONE should match
    Tier 3 (OPTIONAL — not enforced): broad concepts
        e.g. "research", "study", "analysis" — not required

    Returns Tier 1 + Tier 2 terms (for AND query + co-occurrence).
    Tier 3 terms are dropped from strict matching.
    """
    import re as _re

    _STOP_WORDS = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "its",
        "their",
        "his",
        "her",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "about",
        "into",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
    }

    # Broad concept terms — NOT enforced (too restrictive).
    # Field-agnostic: only generic English broad concepts, NO domain terms.
    _OPTIONAL_TERMS = {
        "research",
        "study",
        "analysis",
        "review",
        "overview",
        "characterization",
        "investigation",
        "understanding",
        "origin",
        "evolution",
        "formation",
        "genesis",
        "comparison",
        "correlation",
        "contrast",
        "difference",
        "similarity",
        "connection",
        "link",
        "association",
        "between",
        "among",
        "versus",
        "vs",
        "relative",
    }

    raw_words = _re.split(r"[\s\-]+", query.strip())

    required_terms: list[str] = []  # Tier 1: proper nouns (strict AND)
    important_terms: list[str] = []  # Tier 2: domain nouns (should match)
    for w in raw_words:
        w_clean = w.strip(".,;:!?()[]\"'")
        w_lower = w_clean.lower()
        if len(w_clean) < 3:
            continue
        if w_lower in _STOP_WORDS:
            continue

        if w_clean[0].isupper():
            if w_clean not in required_terms:
                required_terms.append(w_clean)
        elif w_lower in _OPTIONAL_TERMS:
            continue
        else:
            if w_clean not in important_terms:
                important_terms.append(w_clean)

    key_terms = required_terms + important_terms
    if len(key_terms) > 4:
        key_terms = key_terms[:4]
    return key_terms


def _build_and_query(query: str) -> str:
    """Build a boolean AND query requiring ALL key terms to co-occur.

    "Dongargarh granite" → "Dongargarh AND granite"
    "Nandgaon volcanics geochemistry" → "Nandgaon AND volcanics AND geochemistry"

    This prevents APIs from returning 50k+ papers mentioning just one term.
    """
    key_terms = _extract_key_terms(query)
    if len(key_terms) <= 1:
        return query
    and_query = " AND ".join(key_terms)
    log.info("Boolean AND query: '%s' → '%s'", query[:50], and_query[:80])
    return and_query


def _backfill_abstracts_crossref(papers: list) -> int:
    """Fill missing abstracts from Crossref per-DOI work records (B6).

    Mutates PaperRecord.abstract in place. Returns count recovered.
    Failures are logged once, never fatal — backfill is best-effort.
    """
    import re as _re

    dois = [p.doi for p in papers if p.doi]
    if not dois:
        return 0
    try:
        import habanero

        cr = habanero.Crossref()
        recs = cr.works(ids=dois)
    except Exception as e:  # noqa: BLE001 — network/dep resilience, logged loud
        log.warning("Abstract backfill failed (Crossref batch): %s", e)
        return 0
    items = recs if isinstance(recs, list) else recs.get("message", {}).get("items", [])
    by_doi = {}
    for it in items:
        d = (it.get("DOI") or "").lower()
        ab = it.get("abstract") or ""
        # strip JATS tags Crossref embeds
        ab = _re.sub(r"<[^>]+>", " ", ab)
        ab = _re.sub(r"\s+", " ", ab).strip()
        if d and len(ab) > 80:
            by_doi[d] = ab
    recovered = 0
    for p in papers:
        if not (p.abstract or "").strip() and p.doi:
            ab = by_doi.get(p.doi.lower())
            if ab:
                p.abstract = ab
                recovered += 1
    return recovered


def _enforce_cooccurrence(papers: list, query: str) -> list:
    """Post-filter: IDF-weighted key-term coverage (B1 v2, 2026-08-15).

    Flat thresholds fail asymmetrically: 'amphibole'+'thermobarometry' (rare,
    decisive) get outvoted by 'arc'+'magma' (common in a geo corpus) — the
    filter rejected "Amphibole Thermobarometry: a Thermodynamic Approach"
    while keeping generic arc-magma papers. Fix: weight each key term by its
    IDF across the candidate pool; keep a paper when its matched terms carry
    ≥50% of the query's total IDF mass. Rare-term matches decide relevance;
    common-term matches alone never do. Matches singular/plural forms.
    """
    key_terms = _extract_key_terms(query)
    if len(key_terms) <= 1:
        return papers

    def _forms(t: str) -> list[str]:
        tl = t.lower()
        forms = [tl]
        if tl.endswith("s") and len(tl) > 4:
            forms.append(tl[:-1])
        elif not tl.endswith("s"):
            forms.append(tl + "s")
        return forms

    term_forms = {t: _forms(t) for t in key_terms}
    n_docs = max(len(papers), 1)

    def _doc_text(p) -> str:
        return (
            (getattr(p, "title", "") or "") + " " + (getattr(p, "abstract", "") or "")
        ).lower()

    # IDF over the candidate pool itself
    idf: dict[str, float] = {}
    for t, forms in term_forms.items():
        df = sum(1 for p in papers if any(f in _doc_text(p) for f in forms))
        idf[t] = math.log((n_docs + 1) / (df + 1)) + 1.0  # smoothed, ≥1

    total_mass = sum(idf.values())
    cutoff = 0.5 * total_mass

    filtered = []
    rejected = 0
    for p in papers:
        text = _doc_text(p)
        matched_mass = sum(
            idf[t] for t, forms in term_forms.items() if any(f in text for f in forms)
        )
        if matched_mass >= cutoff:
            filtered.append(p)
        else:
            rejected += 1

    if rejected > 0:
        log.info(
            "Co-occurrence filter (IDF-weighted, cutoff %.2f/%.2f): kept %d/%d, rejected %d",
            cutoff,
            total_mass,
            len(filtered),
            len(papers),
            rejected,
        )
    return filtered


def _crossref_journal_search(query: str, max_results: int = 15) -> list:
    """Search Crossref boosted by geological journal container-titles.

    Uses habanero's query_bibliographic parameter to boost papers from
    top geological journals (GCA, CMP, JP, Lithos, Am Min, Chem Geol).
    """
    from habanero import Crossref

    journal_boost = (
        "Contributions to Mineralogy and Petrology "
        "Geochimica et Cosmochimica Acta "
        "Lithos Journal of Petrology "
        "American Mineralogist Chemical Geology "
        "Journal of Metamorphic Geology Ore Geology Reviews"
    )
    try:
        cr = Crossref(mailto="geokit@dev")
        res = cr.works(
            query=query,
            query_bibliographic=journal_boost,
            limit=max_results,
            sort="relevance",
            select=[
                "DOI",
                "title",
                "author",
                "abstract",
                "published-print",
                "published-online",
                "type",
                "container-title",
                "is-referenced-by-count",
            ],
        )
        items = res.get("message", {}).get("items", [])
        records = []
        for item in items:
            rec = _crossref_item_to_record(item)
            if rec:
                rec.source = "crossref_geo"
                records.append(rec)
        log.info("crossref_geo: %d papers from journal-boosted search", len(records))
        return records
    except Exception as e:
        log.warning("crossref_geo search failed: %s", e)
        return []


def _crossref_item_to_record(item: dict, source: str = "crossref"):
    """Convert a Crossref metadata item to PaperRecord.

    Args:
        item: Crossref message dict.
        source: discovery source label (e.g., "crossref", "web_search",
                "web_search_agentic"). Always paired with Crossref metadata
                enrichment, so sources_seen includes both.
    """
    try:
        from _sources import PaperRecord

        doi = item.get("DOI", "")
        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        abstract = item.get("abstract", "")
        # Strip JATS tags from abstract
        if abstract:
            import re as _re

            abstract = _re.sub(r"<[^>]+>", "", abstract).strip()

        authors = []
        for a in item.get("author", []):
            given = a.get("given", "")
            family = a.get("family", "")
            authors.append(
                {
                    "given": given,
                    "family": family,
                    "name": f"{given} {family}".strip(),
                }
            )

        year = None
        for date_key in ("published-print", "published-online", "issued"):
            date_parts = item.get(date_key, {}).get("date-parts", [[]])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = date_parts[0][0]
                break

        # sources_seen: this paper was discovered via `source` AND enriched via Crossref.
        seen = [source, "crossref"] if source != "crossref" else ["crossref"]
        return PaperRecord(
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=item.get("container-title", [""])[0]
            if item.get("container-title")
            else "",
            type=item.get("type", ""),
            source=source,
            sources_seen=seen,
        )
    except Exception as e:
        log.debug("Crossref item conversion failed: %s", e)
        return None


def web_search_paper_discovery(
    query: str, max_results: int = 15, force_refresh: bool = False
) -> list:
    """SOTA paper discovery via web-search skill subprocess.

    Uses the web-search skill's full infrastructure: 9 metasearch backends
    (DDG/Google/Brave/Yandex/Yahoo/Mojeek/Wikipedia/Startpage/Grokipedia),
    4-layer bot-block bypass (cookie/primp/curl_cffi/Playwright stealth),
    9-layer rate-limit defense, per-URL extract cache, domain tiering.

    Returns PaperRecords (resolved through Crossref for metadata).
    Requires the web-search skill venv (auto-bootstraps on first call).

    Activation: --use-web-search flag (registration gated in search_multi_source).
    Pass force_refresh=True to bypass web-search's per-URL extract cache.
    """

    import subprocess as _sp

    SCRIPT = os.path.expanduser("~/.config/opencode/skills/web-search/web_search.py")
    if not os.path.exists(SCRIPT):
        log.warning("web-search skill not found at %s — skipping", SCRIPT)
        return []

    # Propagate force_refresh → web-search subprocess (bypasses per-URL extract cache)
    sub_env = dict(os.environ)
    if force_refresh:
        sub_env["WEB_SEARCH_NO_CACHE"] = "1"

    search_query = f"{query} doi abstract"
    try:
        proc = _sp.run(
            [
                "python3",
                SCRIPT,
                "text",
                search_query,
                "--max-results",
                str(max_results * 3),
                "--sort",
                "authority",
                "--extract",
                "0",
                "--call-budget",
                "90",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=sub_env,
        )
    except _sp.TimeoutExpired:
        log.warning("web-search subprocess timed out for: %s", query[:60])
        return []

    if proc.returncode != 0:
        log.warning(
            "web-search failed (exit %d): %s", proc.returncode, proc.stderr[:200]
        )
        return []

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        log.warning("web-search returned non-JSON: %s", str(ex)[:100])
        return []

    results = data.get("results", [])
    if not results:
        return []

    # Extract DOIs from result URLs/snippets/titles
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
    found_dois: set[str] = set()
    for r in results:
        text = f"{r.get('href', '')} {r.get('body', '')} {r.get('title', '')}"
        for m in doi_pattern.finditer(text):
            found_dois.add(m.group().rstrip(".,;)").lower())

    if not found_dois:
        log.debug("web_search: %d results but 0 DOIs", len(results))
        return []

    log.info(
        "web_search: %d unique DOIs from %d results, resolving via Crossref",
        len(found_dois),
        len(results),
    )

    # Resolve DOIs via Crossref for metadata
    resolved: list = []
    from habanero import Crossref  # HARD dep

    cr = Crossref(mailto="geokit@dev")
    for doi in list(found_dois)[:max_results]:
        try:
            item = cr.works(ids=doi)
            if item and item.get("message"):
                record = _crossref_item_to_record(item["message"], source="web_search")
                if record:
                    resolved.append(record)
        except Exception:
            continue

    log.info("web_search: resolved %d/%d papers", len(resolved), len(found_dois))
    return resolved


def web_search_agentic_discovery(
    query: str, max_results: int = 15, force_refresh: bool = False
) -> list:
    """Deep site-specific paper discovery via web-search agentic --adaptive mode.

    Uses Crawl4AI's AdaptiveCrawler to semantically crawl the top search hit
    (typically a major repository: arxiv.org, biorxiv.org, publisher site)
    with the user's query as relevance filter. Returns ONLY pages that
    answer the query — high precision, lower recall than text mode.

    Slower than web_search_paper_discovery (browser + adaptive crawl, 30-120s)
    but finds papers in JS-rendered sites, conference proceedings, niche
    journals, and behind search forms that text mode misses.

    Output includes `confidence` score (0-1) — semantic relevance of crawled
    content to the query.

    Activation: --use-web-search-agentic flag.
    """
    import subprocess as _sp

    SCRIPT = os.path.expanduser("~/.config/opencode/skills/web-search/web_search.py")
    if not os.path.exists(SCRIPT):
        log.warning("web-search skill not found at %s — skipping agentic", SCRIPT)
        return []

    # Propagate force_refresh → web-search subprocess
    sub_env = dict(os.environ)
    if force_refresh:
        sub_env["WEB_SEARCH_NO_CACHE"] = "1"

    agentic_query = f"{query} paper doi abstract"
    try:
        proc = _sp.run(
            [
                "python3",
                SCRIPT,
                "agentic",
                agentic_query,
                "--adaptive",
                "--max-iter",
                "2",
                "--top-n-extract",
                "3",
                "--call-budget",
                "120",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=sub_env,
        )
    except _sp.TimeoutExpired:
        log.warning("web-search agentic timed out for: %s", query[:60])
        return []

    if proc.returncode != 0:
        log.warning(
            "web-search agentic failed (exit %d): %s",
            proc.returncode,
            proc.stderr[:200],
        )
        return []

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        log.warning("web-search agentic returned non-JSON: %s", str(ex)[:100])
        return []

    # Collect text from ALL result tiers: search hits, extracted pages, adaptive crawl
    text_sources: list[str] = []
    for r in data.get("results", []):
        text_sources.append(
            f"{r.get('href', '')} {r.get('body', '')} {r.get('title', '')}"
        )
    for e in data.get("extracted", []):
        text_sources.append((e.get("content") or "")[:5000])
    ap = data.get("adaptive", {})
    confidence = None
    if isinstance(ap, dict):
        confidence = ap.get("confidence")
        for p in ap.get("pages", []):
            text_sources.append((p.get("content") or "")[:5000])

    # Extract DOIs from all collected text
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
    found_dois: set[str] = set()
    for text in text_sources:
        for m in doi_pattern.finditer(text):
            found_dois.add(m.group().rstrip(".,;)").lower())

    if not found_dois:
        log.debug(
            "web_search_agentic: 0 DOIs (confidence=%s, %d text sources)",
            confidence,
            len(text_sources),
        )
        return []

    log.info(
        "web_search_agentic: %d unique DOIs (confidence=%s)",
        len(found_dois),
        confidence,
    )

    # Resolve DOIs via Crossref for metadata
    resolved: list = []
    from habanero import Crossref  # HARD dep

    cr = Crossref(mailto="geokit@dev")
    for doi in list(found_dois)[:max_results]:
        try:
            item = cr.works(ids=doi)
            if item and item.get("message"):
                record = _crossref_item_to_record(
                    item["message"], source="web_search_agentic"
                )
                if record:
                    resolved.append(record)
        except Exception:
            continue

    log.info(
        "web_search_agentic: resolved %d/%d papers (confidence=%s)",
        len(resolved),
        len(found_dois),
        confidence,
    )
    return resolved


# =============================================================================

# Strong non-geoscience signals — if present, paper is NOT geoscience
_GEO_EXCLUSION = re.compile(
    r"\b(?:magnetic\s+garnet|yttrium\s+iron\s+garnet|\bYIG\b|"
    r"spintronic|spintronics|"
    r"semiconductor|thin\s+film|epitax\w*|"
    r"magneto-optic|magnetoelastic|"
    r"microwave\s+(?:device|circuit|filter|resonator)|antenna|"
    r"photonic|plasmonic|metamaterial|"
    r"neural\s+network|deep\s+learning|machine\s+learning|"
    r"graph\s+neural|computer\s+vision|image\s+segmentation|"
    r"robot(?:ics|ic)?|autonomous|"
    r"lithium.(?:ion|battery)|electrode|cathode|anode|"
    r"quantum\s+(?:dot|well|hall|cascade|computing)|"
    r"superconduct(?:or|ivity|ing)|"
    r"ferrimagnet\w*|antiferromagnet\w*|ferromagnet\w*|"
    r"topological\s+insulator|"
    r"nano(?:wire|particle|tube|rod|crystal)|"
    r"biosensor|gas\s+sensor|"
    r"\bLED\b|photodetector|solar\s+cell|"
    r"Czochralski|Bridgman|flux\s+growth|"
    r"laser\s+(?:medium|crystal|host|cooling)|"
    r"optical\s+(?:pump|fiber|fibre|waveguide|cavity|interconnect)|"
    r"rare.Earth\s+doped|Nd:\w|Yb:\w|Er:\w|Tm:\w|"
    r"upconversion|downconversion|"
    r"scintillator|phosphor|luminescen|"
    r"Perovskite\s+solar|perovskite\s+(?:device|LED|detector)|"
    r"waste.water|pollut.?(?:removal|degradation)|"
    r"catalytic|photocatalytic|electrocatalytic|"
    r"anode.material|cathode.material|battery.material|"
    r"clinical\s+(?:trial|study|outcome)|patient\s+(?:outcome|survival)|"
    r"disease|therapeutic|tumor|cancer|drug\s+delivery|"
    r"dosage|treatment\s+group|placebo|symptom|diagnosis|"
    r"surgical|implant|prosthesis|stent|"
    r"algorithm\s+(?:design|analysis|complexity|implementation)|"
    r"software\s+(?:engineer|framework|architecture|design\s+pattern)|"
    r"benchmark\s+dataset|\bGPU\b|\bCPU\b|database\s+query|"
    r"compiler|runtime\s+(?:error|environment)|"
    r"blockchain|cryptocurrency|smart\s+contract|"
    r"copolymer|cross.link|polymerization|monomer|"
    r"aerodynam|propulsion|turbine\s+blade|combustion\s+chamber|nozzle|"
    r"food\s+(?:science|quality|safety)|nutritional|dietary|antioxidant|"
    r"fermentation\s+(?:process|product)|vitamin|"
    r"wireless\s+(?:network|communication|sensor)|5G|\bLTE\b|"
    r"power\s+electronics|motor\s+drive|invertor|"
    r"image\s+(?:processing|recognition|classif)|"
    r"natural\s+language|sentiment\s+analysis|chatbot|"
    r"lattice\s+(?:QCD|gauge|model|field|theory|gas|action|dynamics)|"
    r"\bIsing\s+(?:model|exactly|universal|critical|spin|2d|3d)\b|"
    r"\bBose\s+(?:gas|Einstein|condensat|Hubbard)\b|"
    r"\bself.avoiding\s+walk\b|"
    r"\bCoulomb\s+gas\b|"
    r"\bpolymer\s+(?:adsorption|chain|model|solution|melt|brush|blend)\b|"
    r"\bpartition\s+function\b|"
    r"\bHamiltonian\s+(?:dynamics|system|matrix|mechanics)\b|"
    r"\bthermodynamic\s+limit\b|"
    r"\brenormali[sz]ation\s+group\b|"
    r"\bmean\s+field\s+(?:theory|approximation|model)\b|"
    r"\batmospheric\s+(?:model|chemistry|transport|circulation|science)\b|"
    r"\bclimate\s+model\b|"
    r"\bnumerical\s+weather\s+prediction\b|"
    r"\btropospheric\b|"
    r"\bstratospheric\b|"
    r"\badsorption\s+isotherm\b|"
    r"\bexactly\s+solvable\b|"
    r"\bprotein\s+(?:folding|structure|interaction|binding)\b|"
    r"\bcell\s+(?:membrane|signaling|cycle|division|culture|line)\b|"
    r"\bDNA\s+(?:sequencing|repair|replication)\b|"
    r"\bRNA\s+(?:sequencing|splicing|interference)\b|"
    r"\benzyme\s+(?:kinetics|catalysis)\b|"
    r"\bantibody\b|"
    r"\bapoptosis\b|"
    r"\bmetabolomics\b|"
    r"\bproteomics\b|"
    r"\bgenomics\b|"
    # Additional stat-mech / math / physics terms
    r"\bYang.Lee\b|"
    r"\bhard.core\s+(?:model|lattice|particle|gas|interaction)\b|"
    r"\bfermion\w*\b|"
    r"\bquark\w*\b|"
    r"\bgraph\s+theory\b|"
    r"\bcombinatorial\b|"
    r"\bmultimedia\s+fugacity\s+model\b|"
    r"\bhard.sphere\b|"
    r"\bLennard.Jones\b|"
    r"\bVan.der.Waals\b|"
    r"\bMonte\s+Carlo\s+(?:simulation|study).*(?:lattice|spin|gas|polymer|Ising|particle)\b|"
    r"\bcondensed\s+matter\b|"
    r"\bmany.body\s+(?:problem|system|theory|physics)\b|"
    r"\bquantum\s+(?:field\s+theory|chromodynamic|electrodynam|gravity)\b|"
    r"\bstochastic\s+(?:process|differential|equation)\b|"
    r"\bErd\w*s.R\u00e9nyi\b|"
    r"\bdegree\s+distribution.*(?:graph|network|node)\b|"
    r"\bscale.free\s+network\b|"
    r"\bpercolation\s+(?:theory|threshold|model|cluster)\b|"
    r"\basbestos\b|"
    r"\bchrysotile\b|"
    r"\bmesothelioma\b|"
    r"\bxenon\s+(?:compound|sodium|chemistry)\b|"
    r"\bdislocation\s+creep\b|"
    r"\blattice.preferred\s+orientation\b"
    r")\b",
    re.IGNORECASE,
)

# Geoscience query signals — when present in query, activate geo filtering
_GEO_QUERY_SIGNALS = re.compile(
    r"\b(?:thermobar|geotherm|petrolog|metamorph|geochron|tecton|"
    r"crustal|mantle|mineral\s+assemblage|facies|"
    r"schist|gneiss|eclogite|amphibolite|granulite|peridotite|"
    r"subduct|orogen|fault\s+rock|mylonite|shear\s+zone|"
    r"pressure.temperature|P.T\s+path|kbar|"
    r"garnet|pyroxene|amphibole|olivine|feldspar|"
    r"mica|quartz|epidote|diamond|coesite|"
    r"titanite|monazite|zircon|apatite|spinel|corundum|"
    r"geothermobarometr|geospeedometr|diffusion\s+chronometry|"
    r"phase\s+equilibri|pseudosection|THERMOCALC|Perple_X|"
    r"inclusion|entrapment\s+pressure|"
    r"metapelit|metabas|ultrahigh.pressure|\bUHP\b|"
    r"retrograde|prograde|exhumation|"
    r"seism|earthquake|magnitude|\bVp\b|\bVs\b|"
    r"volcan|eruption|magma|lava|"
    r"sediment|stratigraph|depositional|facies|"
    r"basin|delta|submarine|"
    r"groundwater|aquifer|hydrogeolog|"
    r"mineraliz|ore\s+deposit|hydrothermal|"
    r"structural\s+geolog|stress\s+inversion|paleostress|strain|"
    r"paleomagnet|magnetic\s+fabric|"
    r"gravity\s+(?:survey|anomal)|magnetic\s+anomal|"
    r"borehole|drilling|\bODP\b|\bIODP\b|"
    r"geochem|isotope|\bSr\b.*\bNd\b|trace\s+element|"
    r"weathering|erosion|geomorpholog|"
    r"diagen|cementation|porosity|permeab|"
    r"sulfur\s+fugacit\w*|sulphur\s+fugacit\w*|oxygen\s+fugacit\w*|"
    r"\bredox\b|redox\s+state|redox\s+condition|"
    r"sulfide|sulphide|pyrrhotite|pentlandite|anhydrite|"
    r"scapolite|magmatic\s+volatile|degassing|ore.forming|"
    r"\bMELTS\b|alphaMELTS|sulfur\s+concentrat|"
    r"\bfS2\b|\bfO2\b|"
    r"\bFMQ\b|fayalite.magnetite.quartz|"
    r"\bNNO\b|\bIW\b|\bQFM\b)"
    r"\b",
    re.IGNORECASE,
)

# Geoscience paper signals — paper must have at least one to be relevant
_GEO_PAPER_SIGNALS = re.compile(
    r"\b(?:metamorph|petrolog|geolog|geochron|tecton|crustal|mantle|"
    r"facies|mineral\s+assemblage|"
    r"schist|gneiss|eclogite|amphibolite|granulite|peridotite|basalt|"
    r"subduct|orogen|fault|shear|cleavage|fabric|"
    r"pressure.temperature|P.T\s+(?:path|estimate|condition)|"
    r"kbar|MPa(?!.Pa)|GPa(?!.Pa)|"
    r"garnet\s+(?:bearing|growth|zoning|composit|resorption)|"
    r"Fe.Mg\s+(?:exchange|partition|thermomet)|"
    r"geothermobarometr|geospeedometr|"
    r"phase\s+equilibri|pseudosection|THERMOCALC|Perple_X|"
    r"inclusion\s+(?:in|host)|entrapment|Raman\s+spectroscop|"
    r"retrograde|prograde|metapelit|metabas|"
    r"\bUHP\b|ultrahigh.pressure|exhumation|"
    r"clinopyroxene|orthopyroxene|hornblende|biotite|muscovite|chlorite|"
    r"staurolite|cordierite|sillimanite|kyanite|andalusite|"
    r"crust|lithosphere|asthenosphere|"
    r"sulfur\s+fugacit\w*|sulphur\s+fugacit\w*|oxygen\s+fugacit\w*|"
    r"\bfS2\b|\bfO2\b|"
    r"redox\s+(?:state|condition|buffer)|"
    r"sulfide\s+saturation|pyrrhotite|pentlandite|anhydrite|"
    r"magmatic\s+volatile|degassing|ore.forming|"
    r"\bMELTS\b|alphaMELTS|sulfur\s+(?:concentrat|speciat)|"
    r"\bFMQ\b|fayalite.magnetite.quartz|"
    r"\bNNO\b|\bIW\b|\bQFM\b|"
    r"magma.*(?:chamber|evolution|ascent)|arc\s+magma|"
    r"mantle.*(?:xenolith|peridotite|source)|"
    r"xenolith|chalcophile|hydrothermal.*sulfide|"
    r"EPMA|LA.ICP.MS|Mossbauer|XANES|microprobe|"
    r"\bmantle\b|\bcrust\b|\btectonic\b|structural\s+geolog|"
    r"fault\s+zone|shear\s+zone|\bigneous\b|\bvolcanic\b|"
    r"\bsedimentary\b|metamorphi|\bolivine\b|\bperidotite\b|"
    r"\bgarnets?\b|\bpyroxenes?\b|\bmagma\b|\bmagmas\b|\bmelts?\b|"
    r"amphiboles?\b|thermobarometr\w*|geothermomet\w*|hornblendes?\b|"
    r"experimental\s+petrolog|petrolog|geochem|geophysic|"
    r"volcanolog|sedimentolog|paleomagnet|geochronolog|"
    r"mineraliz|hydrothermal|ore\s+deposit)\b",
    re.IGNORECASE,
)


# Ambiguous terms shared between geology and physics/chemistry/biology
_AMBIGUOUS_TERMS = frozenset(
    {
        "fugacity",
        "viscosity",
        "diffusion",
        "stress",
        "strain",
        "conductivity",
        "permeability",
        "porosity",
        "density",
        "gradient",
        "anisotropy",
        "partition",
        "solubility",
        "adsorption",
        "absorption",
        "saturation",
        "crystallization",
        "precipitation",
        "weathering",
        "erosion",
        "magnetization",
        "susceptibility",
        "attenuation",
        "tomography",
        "inversion",
        "fracture",
    }
)

_STRONG_GEO_CONTEXT = re.compile(
    r"\b(?:garnet|pyroxene|amphibole|olivine|feldspar|mica|quartz|"
    r"metamorphi|igneou|sedimentar|volcani|"
    r"mantle|crustal|orogeni|subduct|"
    r"eclogite|granulite|amphibolite|peridotite|basalt|"
    r"thermobarometr|geothermobarometr|pseudosection|"
    r"seismolog|volcanolog|sedimentolog|geochronolog)\b",
    re.IGNORECASE,
)


def _expand_search_query(query: str) -> str:
    """Expand query with disambiguating context for ambiguous non-geoscience terms.

    Intent-aware expansion (synonyms, landmarks, method terms) is intentionally
    NOT used for API search — it was found to dilute the query with 50+ terms,
    reducing Crossref results from 10 to 3 and OpenAlex from 10 to 0.

    Post-search ranking handles relevance without hurting discovery yield:
    - intent_filter (pipeline.py:142) — removes peripheral concepts
    - intent_boost (_ranking.py) — boosts landmark authors + must-have terms
    - semantic_relevance_scores (pipeline.py:190) — TF-IDF/BGE on original query
    """
    if not query:
        return query

    # Ontology expansion (for ambiguous terms only)
    q_lower = query.lower()
    if not _GEO_QUERY_SIGNALS.search(q_lower):
        return query
    if _STRONG_GEO_CONTEXT.search(q_lower):
        return query
    query_words = set(q_lower.split())
    if query_words & _AMBIGUOUS_TERMS:
        from _ontology import expand_query_with_ontology  # internal module

        expanded = expand_query_with_ontology(query)
        if expanded != query:
            return expanded
        return (
            query
            + " geology petrology geochemistry mantle magma mineral metamorphic volcanic"
        )
    return query


# =============================================================================
# Domain relevance filtering — removes off-topic papers (e.g., magnetic
# garnets when the query is about metamorphic garnet thermobarometry)
# =============================================================================


def _is_domain_relevant(paper: object, query: str) -> bool:
    """Check if paper is relevant to the query's scientific domain.

    For geoscience queries, filters out materials-science, CS, and physics
    papers that share mineral names (e.g., 'garnet') but belong to unrelated
    fields.  For non-geoscience queries, always returns True.
    """
    if not query:
        return True
    q_lower = query.lower()

    # Only apply domain filtering for geoscience queries
    if not _GEO_QUERY_SIGNALS.search(q_lower):
        return True

    # Get paper text
    title = ""
    abstract = ""
    if hasattr(paper, "title"):
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
    elif isinstance(paper, dict):
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
    text = f"{title} {abstract}"

    # Count geoscience signals FIRST (needed for soft exclusions below)
    geo_signal_count = len(_GEO_PAPER_SIGNALS.findall(text))

    # Exclude papers with strong non-geoscience signals. BUG FIX 2026-08-15:
    # ML/DL/NN terms were HARD exclusions — they rejected legitimate
    # ML-applied-to-geoscience papers ("Machine learning thermobarometry...")
    # which a modern review must include. ML terms are now SOFT: excluded
    # only when geoscience evidence is thin (< 2 signals). All other
    # exclusion terms stay hard.
    _ML_SOFT = ("machine learning", "deep learning", "neural network",
                "graph neural", "computer vision", "image segmentation")
    for m in _GEO_EXCLUSION.finditer(text):
        term = m.group(0).lower()
        if term in _ML_SOFT:
            if geo_signal_count < 2:
                return False
            continue  # ML term + strong geo evidence = methods paper, keep
        return False  # hard exclusion
    query_words = set(q_lower.split())
    ambiguous_overlap = query_words & _AMBIGUOUS_TERMS
    is_ambiguous = bool(ambiguous_overlap)
    # Scale threshold by ambiguity: more ambiguous = require more evidence
    if is_ambiguous:
        min_signals = 2
    else:
        min_signals = 1
    if geo_signal_count < min_signals:
        return False

    # Additional check: ontology domain penalty (catches patterns _GEO_EXCLUSION misses)
    from _ontology import domain_penalty  # internal module

    if domain_penalty(text) == 0.0:
        return False

    return True


# =============================================================================
# Multi-source search
# =============================================================================




def search_multi_source(
    query: str,
    max_per_source: int = 15,
    sources: list[str] | None = None,
    semantic: bool = False,
    force_refresh: bool = False,
    per_source_timeout: float = 30.0,
    filters: dict | None = None,
    use_web_search: bool = False,
    use_web_search_agentic: bool = False,
) -> list[PaperRecord]:
    """Search across multiple sources IN PARALLEL, return merged list.

    Each source runs in its own thread with a hard timeout. One hanging
    source cannot block the pipeline — it is abandoned after
    per_source_timeout seconds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FuturesTimeout

    expanded_query = _expand_search_query(query)

    sources = sources or DEFAULT_SOURCES.split(",")
    out: list[PaperRecord] = []

    # Agentic crawl takes 60-180s (browser + AdaptiveCrawler); bump per-source timeout.
    if use_web_search_agentic:
        per_source_timeout = max(per_source_timeout, 180.0)

    # Build source→callable map
    # Convert generic filters to API-specific format
    cr_filter: dict = {}
    oa_filter: dict = {}
    if filters:
        if filters.get("year_from"):
            cr_filter["from-pub-date"] = f"{filters['year_from']}-01-01"
            oa_filter["from_publication_date"] = f"{filters['year_from']}-01-01"
        if filters.get("year_to"):
            cr_filter["until-pub-date"] = f"{filters['year_to']}-12-31"
            oa_filter["to_publication_date"] = f"{filters['year_to']}-12-31"
        if filters.get("open_access_only"):
            cr_filter["has-full-text"] = "true"
            oa_filter["is_oa"] = "true"
        if filters.get("publication_type"):
            cr_filter["type"] = filters["publication_type"]
            oa_filter["type"] = filters["publication_type"]

    source_calls: dict[str, Any] = {}
    if "crossref" in sources:
        source_calls["crossref"] = lambda: crossref_search(
            expanded_query, max_results=max_per_source, filter_dict=cr_filter or None
        )
    if "openalex" in sources:
        # Two-mode OpenAlex search: semantic (AI relevance) + keyword (filterable).
        # Semantic search uses OpenAlex embeddings for better recall; keyword
        # search supports year/OA/type filters. Run BOTH when filters are set
        # and merge results for maximum coverage + precision.
        if oa_filter or (
            filters and (filters.get("year_from") or filters.get("year_to"))
        ):
            # Filters require keyword search (semantic API ignores filters)
            source_calls["openalex"] = lambda: openalex_search(
                expanded_query,
                max_results=max_per_source,
                force_refresh=force_refresh,
                filters=oa_filter or None,
            )
            # Also run semantic search WITHOUT filters for recall, then merge
            source_calls["openalex_semantic"] = lambda: openalex_semantic_search(
                expanded_query, max_results=max_per_source
            )
        else:
            # No filters — semantic search gives better recall
            source_calls["openalex"] = lambda: openalex_semantic_search(
                expanded_query, max_results=max_per_source
            )
    if "s2" in sources:
        source_calls["s2"] = lambda: s2_search(
            expanded_query, max_results=max_per_source
        )
    if "eartharxiv" in sources:
        from _sources import eartharxiv_search

        source_calls["eartharxiv"] = lambda: eartharxiv_search(
            expanded_query, max_results=max_per_source
        )
    if "usgs" in sources:
        from _sources import usgs_search

        source_calls["usgs"] = lambda: usgs_search(
            expanded_query, max_results=max_per_source
        )
    if "arxiv" in sources:
        source_calls["arxiv"] = lambda: arxiv_search(
            expanded_query, max_results=max_per_source
        )
    if "epmc" in sources:
        from _sources import epmc_search

        source_calls["epmc"] = lambda: epmc_search(
            expanded_query, max_results=max_per_source
        )

    # Web-search skill integration — SOTA paper discovery via subprocess.
    # Activated by --use-web-search flag or SCIENTIFIC_RESEARCH_ENABLE_WEB_SEARCH=1.
    # Adds 9 metasearch backends + 4-layer bot-block bypass + per-URL extract cache.
    if (
        use_web_search
        or os.environ.get("SCIENTIFIC_RESEARCH_ENABLE_WEB_SEARCH", "0") == "1"
    ):
        source_calls["web_search"] = lambda: web_search_paper_discovery(
            query, max_results=min(max_per_source, 15), force_refresh=force_refresh
        )

    # Web-search AGENTIC integration — Crawl4AI AdaptiveCrawler on top hit.
    # Slower (browser + semantic crawl, 30-120s) but finds papers in JS-rendered
    # sites and behind search forms that text mode misses. Adds confidence score.
    if use_web_search_agentic:
        source_calls["web_search_agentic"] = lambda: web_search_agentic_discovery(
            query, max_results=min(max_per_source, 10), force_refresh=force_refresh
        )

    # Geological discovery enhancement — intent-aware source selection.
    # For geo queries: add journal-targeted search, synonym variants, web-search.
    # Intent determines which sources are PRIORITIZED (higher max_results).
    if _GEO_QUERY_SIGNALS.search(query.lower()):
        # Parse intent for source prioritization
        intent_topics: list[str] = []
        try:
            from _intent import parse_intent

            intent = parse_intent(query)
            if intent.material:
                intent_topics.append(intent.material)
            if intent.method:
                intent_topics.append(intent.method)
        except Exception:
            pass

        # 1. Journal-targeted Crossref search — boost from 57 geo journals
        source_calls["crossref_geo"] = lambda: _crossref_journal_search(
            query, max_results=max_per_source
        )

        # 2. Geological synonym variant — separate Crossref call
        synonym_q = _geo_synonym_variant(query)
        if synonym_q and synonym_q != expanded_query:
            source_calls["crossref_synonym"] = lambda: crossref_search(
                synonym_q, max_results=max_per_source, filter_dict=cr_filter or None
            )

        # 3. Web-search skill (text mode) — finds papers missed by Crossref/OpenAlex.
        # Auto-registered when --use-web-search flag or env var set (see top of function).

        # 4. USGS publications — public-domain earth science reports
        if "usgs" not in source_calls:
            from _sources import usgs_search

            source_calls["usgs"] = lambda: usgs_search(
                expanded_query, max_results=min(max_per_source, 10)
            )

        # Intent-based logging
        if intent_topics:
            log.info(
                "Geo discovery: intent=%s → prioritized sources: crossref_geo, synonym, web_search, usgs",
                intent_topics,
            )

        # 4. USGS publications — public-domain earth science reports
        if "usgs" not in source_calls:
            from _sources import usgs_search

            source_calls["usgs"] = lambda: usgs_search(
                expanded_query, max_results=min(max_per_source, 10)
            )

    log.info(
        "Searching %d sources in parallel (timeout=%.0fs each): %s",
        len(source_calls),
        per_source_timeout,
        ", ".join(source_calls),
    )

    # Use ThreadPoolExecutor WITHOUT context manager — __exit__ blocks waiting
    # for ALL threads including hanging ones (S2 library retries 429s internally).
    pool = ThreadPoolExecutor(max_workers=len(source_calls))
    future_to_source = {pool.submit(fn): name for name, fn in source_calls.items()}
    try:
        for future in as_completed(future_to_source, timeout=per_source_timeout + 5):
            src_name = future_to_source[future]
            try:
                results = future.result(timeout=5)
                out.extend(results)
                log.info("  %-12s → %d results", src_name, len(results))
            except Exception as e:
                if src_name == "s2":
                    from _sources import _s2_record_failure  # internal module

                    _s2_record_failure()
                log.warning("  %-12s → FAILED: %s", src_name, e)
    except FuturesTimeout:
        for future, src_name in future_to_source.items():
            if not future.done():
                log.warning(
                    "  %-12s → TIMEOUT (abandoned after %.0fs)",
                    src_name,
                    per_source_timeout,
                )
                if src_name == "s2":
                    from _sources import _s2_record_failure  # internal module

                    _s2_record_failure()
                future.cancel()
    finally:
        # shutdown(wait=False) abandons hanging threads — they don't block caller.
        # cancel_futures=True cancels not-yet-started tasks.
        pool.shutdown(wait=False, cancel_futures=True)

    return out


# =============================================================================
# Cohen 2018 snowballing
# =============================================================================
def snowball_backward(
    seed: list[PaperRecord], max_per_seed: int = 10, depth: int = 1
) -> list[PaperRecord]:
    """Backward snowballing: fetch papers cited by seeds.
    Uses OpenAlex referenced_works (OpenAlex IDs) + S2 references."""
    out: list[PaperRecord] = []
    seen_ids = {p.primary_id for p in seed}
    for p in seed:
        refs: list[PaperRecord] = []
        if p.openalex_id and p.referenced_works:
            try:
                log.info(
                    "Backward OA refs for %s (%d)",
                    p.doi or p.openalex_id,
                    len(p.referenced_works),
                )
                for rid in p.referenced_works[:max_per_seed]:
                    if cache_has(rid):
                        rec = cache_get(rid)
                    else:
                        rec = openalex_get_by_id(rid)
                        if rec:
                            cache_put(rec)
                    if rec:
                        refs.append(rec)
            except Exception as e:
                log.warning("OA referenced_works failed: %s", e)
        if p.s2_paper_id:
            try:
                s2_refs = s2_get_references(p.s2_paper_id, max_results=max_per_seed)
                refs.extend(s2_refs)
            except Exception as e:
                log.warning("S2 references failed: %s", e)
        for r in refs:
            if r.primary_id not in seen_ids:
                seen_ids.add(r.primary_id)
                out.append(r)
    if depth > 1 and out:
        deeper = snowball_backward(out, max_per_seed=max_per_seed, depth=depth - 1)
        for r in deeper:
            if r.primary_id not in seen_ids:
                seen_ids.add(r.primary_id)
                out.append(r)
    return out


def snowball_forward(
    seed: list[PaperRecord], max_per_seed: int = 10, depth: int = 1
) -> list[PaperRecord]:
    """Forward snowballing: fetch papers that cite seeds.
    Uses OpenAlex cited_by_api_url + S2 forward citations."""
    out: list[PaperRecord] = []
    seen_ids = {p.primary_id for p in seed}
    for p in seed:
        citing: list[PaperRecord] = []
        if p.openalex_id:
            try:
                log.info("Forward OA cites for %s", p.doi or p.openalex_id)
                citing.extend(
                    openalex_get_cited_by(p.openalex_id, max_results=max_per_seed)
                )
            except Exception as e:
                log.warning("OA cited_by failed: %s", e)
        # S2 citations REMOVED — causes 429 cascades. OpenAlex cited_by
        # (above) is the primary forward-citation source.
        for r in citing[:max_per_seed]:
            if r.primary_id not in seen_ids:
                seen_ids.add(r.primary_id)
                out.append(r)
    if depth > 1 and out:
        deeper = snowball_forward(out, max_per_seed=max_per_seed, depth=depth - 1)
        for r in deeper:
            if r.primary_id not in seen_ids:
                seen_ids.add(r.primary_id)
                out.append(r)
    return out


# =============================================================================
# Boolean query builder (PICO)
# =============================================================================
def build_boolean_query(
    population: str = "",
    intervention: str = "",
    outcome: str = "",
    comparator: str = "",
    study_type: str = "",
) -> str:
    """Build a Boolean query from PICO components.
    Each component is AND'd. Synonyms within a component are OR'd."""
    parts: list[str] = []
    for val in [population, intervention, comparator, outcome]:
        if val:
            syn = _expand_synonyms(val)
            if len(syn) > 1:
                parts.append("(" + " OR ".join(syn) + ")")
            else:
                parts.append(syn[0])
    if study_type:
        parts.append(study_type)
    return " AND ".join(parts)


_SYNONYMS = {
    "cancer": ["cancer", "tumor", "tumour", "neoplasm", "oncology"],
    "machine learning": ["machine learning", "ML", "deep learning", "neural network"],
    "climate": ["climate", "climatic", "weather", "meteorological"],
    "rct": ["randomized controlled trial", "RCT", "randomised"],
    "meta-analysis": ["meta-analysis", "metaanalysis", "systematic review"],
}


def _expand_synonyms(term: str) -> list[str]:
    key = term.lower().strip()
    return _SYNONYMS.get(key, [term])


# =============================================================================
# Automated query expansion (ported from litsearchr)
# =============================================================================
def auto_expand_query(seed_papers: list[PaperRecord], max_terms: int = 15) -> list[str]:
    """Expand a search query using FakeRAKE + co-occurrence network.

    Ported from litsearchr (Grames et al. 2019):
    1. Extract candidate terms from seed paper titles + abstracts (FakeRAKE)
    2. Build co-occurrence network
    3. Rank by strength (weighted degree)
    4. Find changepoint cutoff
    5. Return top terms above cutoff

    Returns list of expanded search terms to OR into the original query.
    """
    if not seed_papers:
        return []
    seed_texts = []
    for p in seed_papers:
        text = (p.title or "") + " " + (p.abstract or "")
        if text.strip():
            seed_texts.append(text.strip())
    if not seed_texts:
        return []
    expanded = _expand_query_terms(seed_texts, max_terms=max_terms, min_freq=2)
    log.info(
        "Query expansion: %d terms from %d seed papers", len(expanded), len(seed_texts)
    )
    return expanded


# =============================================================================
# Saturation-detecting snowballing (Cohen 2018 + saturation heuristic)
# =============================================================================
def snowball_with_saturation(
    seed: list[PaperRecord],
    max_depth: int = 3,
    max_per_seed: int = 10,
    min_new_ratio: float = 0.15,
    saturation_min_batch: int = 5,
) -> tuple[list[PaperRecord], list[dict]]:
    """Recursive snowballing with saturation detection.

    Performs forward + backward snowballing iteratively until either:
    - max_depth reached, OR
    - new papers found < min_new_ratio of current batch (saturation)

    Returns (all_new_papers, depth_log) where depth_log tracks:
      [{"depth": 1, "backward": N, "forward": N, "new": N, "ratio": float}, ...]
    """
    all_new: list[PaperRecord] = []
    seen_ids = {p.primary_id for p in seed}
    depth_log: list[dict] = []
    current_batch = seed[: max_per_seed * 2]  # configurable cap

    for depth in range(1, max_depth + 1):
        if not current_batch:
            log.info("Snowball: no papers to expand at depth %d", depth)
            break

        log.info("Snowball depth %d: %d papers to expand", depth, len(current_batch))
        backward = snowball_backward(current_batch, max_per_seed=max_per_seed, depth=1)
        forward = snowball_forward(
            current_batch[:10], max_per_seed=max_per_seed, depth=1
        )
        combined = dedup_papers(backward + forward)

        # Filter out already-seen
        new_this_depth: list[PaperRecord] = []
        for p in combined:
            if p.primary_id not in seen_ids:
                seen_ids.add(p.primary_id)
                new_this_depth.append(p)

        batch_size = max(len(current_batch), 1)
        ratio = len(new_this_depth) / batch_size

        depth_log.append(
            {
                "depth": depth,
                "backward_found": len(backward),
                "forward_found": len(forward),
                "new_unique": len(new_this_depth),
                "ratio": round(ratio, 3),
            }
        )

        log.info(
            "Snowball depth %d: %d backward + %d forward → %d new (ratio %.2f)",
            depth,
            len(backward),
            len(forward),
            len(new_this_depth),
            ratio,
        )

        all_new.extend(new_this_depth)

        # Saturation check: if we found enough papers but ratio is low, stop
        if len(new_this_depth) < saturation_min_batch and ratio < min_new_ratio:
            log.info(
                "Snowball saturated at depth %d (ratio %.2f < %.2f, batch %d < %d)",
                depth,
                ratio,
                min_new_ratio,
                len(new_this_depth),
                saturation_min_batch,
            )
            break

        # Next depth: expand the new papers we just found
        current_batch = new_this_depth[: max_per_seed * 2]

    return all_new, depth_log


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="discover",
        description="Multi-source academic paper discovery + Cohen 2018 snowballing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("query", help="search query")
    p.add_argument(
        "--max", type=int, default=100, help="max papers in final corpus (default 100)"
    )
    p.add_argument(
        "--require-abstract",
        action="store_true",
        default=True,
        help="only keep papers with abstracts (default True — improves extraction quality)",
    )
    p.add_argument(
        "--no-require-abstract",
        action="store_false",
        dest="require_abstract",
        help="keep papers without abstracts (may reduce extraction quality)",
    )
    p.add_argument(
        "--max-per-source",
        type=int,
        default=DEFAULT_MAX_PER_SOURCE,
        help="max results per source before dedup (default 15)",
    )
    p.add_argument(
        "--sources", default=DEFAULT_SOURCES, help=f"comma-separated: {DEFAULT_SOURCES}"
    )
    p.add_argument(
        "--semantic",
        action="store_true",
        help="use OpenAlex semantic search (paragraph-length query)",
    )
    p.add_argument(
        "--use-web-search",
        action="store_true",
        help="add web-search skill as a discovery source (9 metasearch backends + "
        "4-layer bot-block bypass + per-URL extract cache). Finds papers "
        "Crossref/OpenAlex/S2 miss: conference papers, niche journals, preprints. "
        "Also activates via SCIENTIFIC_RESEARCH_ENABLE_WEB_SEARCH=1 env var.",
    )
    p.add_argument(
        "--use-web-search-agentic",
        action="store_true",
        help="deep-research mode: web-search agentic --adaptive crawls top hit with "
        "Crawl4AI semantic filter (confidence-scored). Slower (30-120s) but finds "
        "papers in JS-rendered sites and behind search forms. Use for hard-to-find "
        "papers when --use-web-search is insufficient.",
    )
    p.add_argument(
        "--auto-web-search-threshold",
        type=int,
        default=5,
        metavar="N",
        help="auto-enable web_search supplement when discovery yields fewer than N "
        "papers. Catches sparse topics where Crossref/OpenAlex/S2 miss the field. "
        "Set 0 to disable. Default 5.",
    )
    p.add_argument(
        "--snowball",
        type=int,
        default=0,
        metavar="N",
        help="snowball depth (1 = forward+backward 1 hop, default 0)",
    )
    p.add_argument(
        "--snowball-max",
        type=int,
        default=10,
        help="max papers per seed for snowballing (default 10)",
    )
    p.add_argument(
        "--snowball-saturation",
        action="store_true",
        help="use saturation-detecting snowballing (stops when new paper ratio drops)",
    )
    p.add_argument(
        "--snowball-depth-max",
        type=int,
        default=3,
        help="max depth for saturation snowballing (default 3)",
    )
    p.add_argument(
        "--expand",
        action="store_true",
        help="auto-expand query using FakeRAKE + co-occurrence from initial results",
    )
    p.add_argument(
        "--expand-max-terms",
        type=int,
        default=15,
        help="max terms for query expansion (default 15)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("research_outputs/corpus.json"),
        help="output corpus.json path",
    )
    p.add_argument(
        "--explore", action="store_true", help="landscape mode: shallow scan, no dedup"
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="skip knowledge base, fetch fresh from APIs (overrides cached search results)",
    )
    p.add_argument(
        "--build-query",
        action="store_true",
        help="build Boolean query from PICO and exit (no search)",
    )
    p.add_argument("--pico-population", help="PICO P component")
    p.add_argument("--pico-intervention", help="PICO I component")
    p.add_argument("--pico-comparator", help="PICO C component")
    p.add_argument("--pico-outcome", help="PICO O component")
    p.add_argument("--pico-study-type", help="study type filter (RCT, cohort, etc.)")
    p.add_argument(
        "--summary",
        action="store_true",
        help="also write corpus_summary.md (compact LLM-facing summary)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument(
        "--from-year", type=int, default=None, help="earliest publication year"
    )
    p.add_argument("--to-year", type=int, default=None, help="latest publication year")
    p.add_argument(
        "--open-access-only", action="store_true", help="only discover OA papers"
    )
    p.add_argument(
        "--type",
        default="",
        help="publication type filter (journal-article, book-chapter, etc.)",
    )
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    # PICO Boolean query builder mode
    if args.build_query:
        q = build_boolean_query(
            population=args.pico_population or "",
            intervention=args.pico_intervention or "",
            comparator=args.pico_comparator or "",
            outcome=args.pico_outcome or "",
            study_type=args.pico_study_type or "",
        )
        print(q)
        return 0

    if not args.query:
        p.error("must provide query (or --build-query with PICO flags)")

    started = time.time()
    log.info("=== WEB DISCOVERY ===")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    # Build search filters from CLI args
    cli_filters: dict = {}
    if args.from_year or args.to_year:
        cli_filters["year_from"] = args.from_year
        cli_filters["year_to"] = args.to_year
    if args.open_access_only:
        cli_filters["open_access_only"] = True
    if args.type:
        cli_filters["publication_type"] = args.type
    if args.require_abstract:
        cli_filters["require_abstract"] = True

    results = search_multi_source(
        args.query,
        max_per_source=int(args.max_per_source * 1.5)
        if args.require_abstract
        else args.max_per_source,
        sources=sources,
        semantic=args.semantic,
        force_refresh=args.force_refresh,
        filters=cli_filters or None,
        use_web_search=args.use_web_search,
        use_web_search_agentic=args.use_web_search_agentic,
    )
    log.info("Web: %d raw across sources", len(results))

    # Dedup
    if args.explore:
        log.info("Landscape mode: skipping dedup")
        final = results
    else:
        log.info("=== DEDUP ===")
        final = dedup_papers(results)
        log.info("Dedup: %d → %d", len(results), len(final))

    # B6 abstract backfill: OpenAlex leaves abstract_inverted_index empty for
    # copyright-filtered publishers (observed 8/12 empty). Crossref's
    # per-DOI work record often carries the abstract even when the search
    # result didn't. One batched habanero call recovers what the APIs split.
    missing = [p for p in final if p.doi and not (p.abstract or "").strip()]
    if missing:
        recovered = _backfill_abstracts_crossref(missing)
        if recovered:
            log.info(
                "Abstract backfill: %d/%d recovered via Crossref DOI",
                recovered,
                len(missing),
            )

    # Auto-supplement via web_search when post-dedup corpus is sparse.
    # Triggered when: (a) threshold > 0, (b) user didn't already enable
    # web_search/agentic explicitly, (c) final count below threshold.
    # Catches niche topics where Crossref/OpenAlex/S2 miss the field entirely.
    if (
        args.auto_web_search_threshold > 0
        and not args.use_web_search
        and not args.use_web_search_agentic
        and len(final) < args.auto_web_search_threshold
    ):
        log.info(
            "Sparse post-dedup (%d < threshold %d) — auto-supplementing via web_search",
            len(final),
            args.auto_web_search_threshold,
        )
        try:
            supplement = web_search_paper_discovery(
                args.query,
                max_results=max(10, args.auto_web_search_threshold * 2),
                force_refresh=args.force_refresh,
            )
            if supplement:
                log.info("Auto-supplement: +%d papers from web_search", len(supplement))
                final.extend(supplement)
        except Exception as ex:
            log.warning("Auto-supplement via web_search failed: %s", ex)

    # Snowball (Cohen 2018)
    if args.snowball and final:
        if args.snowball_saturation:
            log.info(
                "=== SATURATION SNOWBALLING (max_depth=%d) ===", args.snowball_depth_max
            )
            snowballed, depth_log = snowball_with_saturation(
                final[: args.snowball_max * 2],
                max_depth=args.snowball_depth_max,
                max_per_seed=args.snowball_max,
            )
            log.info(
                "Saturation snowball: %d total new papers across %d depths",
                len(snowballed),
                len(depth_log),
            )
            for entry in depth_log:
                log.info(
                    "  depth %d: %d new (ratio %.2f)",
                    entry["depth"],
                    entry["new_unique"],
                    entry["ratio"],
                )
        else:
            log.info("=== SNOWBALLING (depth %d) ===", args.snowball)
            backward = snowball_backward(
                final[: args.snowball_max * 2],
                max_per_seed=args.snowball_max,
                depth=args.snowball,
            )
            forward = snowball_forward(
                final[: args.snowball_max],
                max_per_seed=args.snowball_max,
                depth=args.snowball,
            )
            snowballed = dedup_papers(backward + forward)
            log.info(
                "Snowball: %d backward + %d forward → %d unique",
                len(backward),
                len(forward),
                len(snowballed),
            )
        final = dedup_papers(final + snowballed)

    # Query expansion (litsearchr-style)
    if args.expand and final:
        log.info("=== QUERY EXPANSION ===")
        expanded_terms = auto_expand_query(
            final[: args.snowball_max * 2], max_terms=args.expand_max_terms
        )
        if expanded_terms:
            log.info("Expanded terms: %s", ", ".join(expanded_terms[:10]))
            # Run a second search with expanded terms
            expanded_query = " OR ".join(expanded_terms[: args.expand_max_terms])
            log.info("Running expanded search: %s...", expanded_query[:100])
            expanded_results = search_multi_source(
                expanded_query,
                max_per_source=args.max_per_source,
                sources=sources,
                semantic=True,
            )
            log.info("Expanded search: %d raw results", len(expanded_results))
            final = dedup_papers(final + expanded_results)

    # Filter to papers with abstracts (improves extraction quality)
    if args.require_abstract and final:
        before = len(final)
        final = [p for p in final if p.abstract and len(p.abstract.strip()) > 50]
        log.info(
            "Abstract filter: %d → %d (removed %d without abstracts)",
            before,
            len(final),
            before - len(final),
        )

    # Domain relevance (geoscience queries only) — wired 2026-08-15.
    # _is_domain_relevant was defined but NEVER called from the CLI flow
    # (its only caller _apply_post_filters was dead code, removed); junk was
    # caught only by the IDF co-occurrence filter. ML/DL terms are soft
    # exclusions now (see _is_domain_relevant), so ML-applied-to-geo
    # papers survive.
    if final and not args.explore and _GEO_QUERY_SIGNALS.search(args.query.lower()):
        before_domain = len(final)
        final = [p for p in final if _is_domain_relevant(p, args.query)]
        if len(final) < before_domain:
            log.info(
                "Domain filter: %d → %d (removed %d off-domain)",
                before_domain, len(final), before_domain - len(final),
            )

    # Co-occurrence filter (Tier 3 cherry-pick) — keep only papers
    # mentioning ALL key terms. Matches singular/plural forms.
    if final and not args.explore:
        before_cooccur = len(final)
        final = _enforce_cooccurrence(final, args.query)
        if len(final) < before_cooccur:
            log.info(
                "Co-occurrence: %d → %d (removed %d missing key terms)",
                before_cooccur,
                len(final),
                before_cooccur - len(final),
            )

    # Multi-criteria ranking (replaces crude sort-by-citation)
    # Scores on: semantic relevance (30%) + citation influence (25%) +
    #            network centrality (25%) + recency (20%)
    # Then applies diversity filter (max 3/author, max 5/venue) + recency balance
    if len(final) > args.max:
        from _ranking import rank_papers

        log.info("=== RANKING %d papers (top %d) ===", len(final), args.max)
        ranked = rank_papers(args.query, final, max_n=args.max)
        final = [p for p, _, _ in ranked]
        log.info("Ranked: %d → %d (diversity-filtered)", len(ranked), len(final))
    elif final:
        final.sort(key=lambda r: -(r.citation_count or 0))
        log.info("Small corpus — sorted by citation_count (no ranking needed)")

    # Save corpus
    total_discovered = len(results)
    save_corpus(
        final,
        args.output,
        meta={
            "query": args.query,
            "sources": args.sources,
            "snowball_depth": args.snowball,
            "snowball_max_per_seed": args.snowball_max,
            "total_discovered": total_discovered,
            "final_count": len(final),
            "duration_sec": round(time.time() - started, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(f"Wrote {len(final)} papers → {args.output}")

    if args.summary:
        summary_path = args.output.parent / "corpus_summary.md"
        summary_path.write_text(render_compact_summary(final, max_papers=len(final)))
        print(f"Wrote compact summary → {summary_path}")

    # Generate PRISMA flow diagram
    try:
        from _render import render_prisma_png

        prisma_path = str(args.output.parent / "prisma.png")
        render_prisma_png(
            identification=total_discovered,
            duplicates_removed=total_discovered - len(final),
            screened=len(final),
            excluded_at_screening=0,
            included=len(final),
            output_path=prisma_path,
        )
        print(f"Wrote PRISMA diagram → {prisma_path}")
    except Exception as e:
        log.warning("PRISMA diagram generation failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())

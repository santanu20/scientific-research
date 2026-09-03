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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Make _sources importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from _nlp import (
    expand_query as _expand_query_terms,
)
from _sources import (
    _EMAIL,
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
DEFAULT_SOURCES = "web_search,openalex,crossref"
DEFAULT_MAX_PER_SOURCE = 50  # per source; total ~100-150 after dedup

# =============================================================================
# Geological discovery enhancement — journal-targeted Crossref search +
# geological synonym query variants + web-search skill for paper DOIs.
# These are ADDITIONAL parallel sources (not query dilution) — each runs
# independently, results are merged + deduplicated.
# =============================================================================

# Journal-ISSN boost infrastructure removed — its only consumer
# (_crossref_journal_search) was deleted under the dynamic-context mandate.
# _GEO_SYNONYM_MAP removed — static geology vocabulary (dynamic-context mandate, 2026-08-22).


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
        "along",
        "across",
        "between",
        "among",
        "within",
        "through",
        "throughout",
        "toward",
        "towards",
        "via",
        "per",
        "over",
        "under",
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
        "comparative",
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

    # Dynamic Title-Case detection: when most words are capitalized
    # (e.g. "Comparative Petrographic Analysis of the ... Dykes"),
    # capitalization carries ZERO information — treating every word as a
    # proper noun poisoned the AND-query with generic terms and truncated
    # real place-name discriminators away (2026-08-22 fix).
    alpha_words = [w for w in raw_words if w and w[0].isalpha()]
    title_case = (
        sum(1 for w in alpha_words if w[0].isupper()) / len(alpha_words) > 0.6
        if alpha_words
        else False
    )

    required_terms: list[str] = []  # Tier 1: proper nouns (strict AND)
    important_terms: list[str] = []  # Tier 2: domain nouns (should match)
    for w in raw_words:
        w_clean = w.strip(".,;:!?()[]\"'")
        w_lower = w_clean.lower()
        if len(w_clean) < 3:
            continue
        if w_lower in _STOP_WORDS:
            continue

        # Generic-concept filter runs FIRST: a broad-concept word stays
        # optional even when the query is Title Case ("Analysis" must
        # never masquerade as a proper noun).
        if w_lower in _OPTIONAL_TERMS:
            continue

        if not title_case and w_clean[0].isupper():
            if w_clean not in required_terms:
                required_terms.append(w_clean)
        else:
            if w_lower not in important_terms:
                important_terms.append(w_lower)

    key_terms = required_terms + important_terms
    return key_terms


def _build_and_query(query: str) -> str:
    """Build a boolean AND query requiring ALL key terms to co-occur.

    "Dongargarh granite" → "Dongargarh AND granite"
    "Nandgaon volcanics geochemistry" → "Nandgaon AND volcanics AND geochemistry"

    This prevents APIs from returning 50k+ papers mentioning just one term.
    """
    # Shape here, not in the extractor: co-occurrence needs the FULL term
    # list; the API AND-query wants only the top few discriminators
    # (proper-noun tier sorts ahead of content words).
    key_terms = _extract_key_terms(query)[:4]
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
    except Exception as e:
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
        """Surface-form variants (language morphology only — no vocabulary).

        Substring matching means a SHORTER variant also catches its own
        derivatives ("basalt" matches "basaltic"), so suffix-stripped
        bases strictly widen recall without false-negative risk.
        """
        tl = t.lower()
        forms = {tl}
        if tl.endswith("ies") and len(tl) > 5:
            forms.update({tl[:-3] + "y", tl[:-1]})
        elif tl.endswith("s") and len(tl) > 4:
            forms.add(tl[:-1])
        elif not tl.endswith("s"):
            forms.add(tl + "s")
        if tl.endswith("ical") and len(tl) > 6:
            forms.add(tl[:-2])  # geological → geologic
        if tl.endswith("ing") and len(tl) > 6:
            forms.add(tl[:-3])          # weathering → weather (matches weathered)
        if tl.endswith("ed") and len(tl) > 5:
            forms.add(tl[:-2])          # weathered → weather
        if tl.endswith("ic") and len(tl) > 4:
            forms.add(tl[:-2])          # basaltic → basalt
            forms.add(tl[:-1])          # ...and basalti- substring safety
        return list(forms)

    term_forms = {t: _forms(t) for t in key_terms}
    n_docs = max(len(papers), 1)

    def _doc_text(p) -> str:
        return ((getattr(p, "title", "") or "") + " " + (getattr(p, "abstract", "") or "")).lower()

    # IDF over the candidate pool itself. Terms with ZERO document
    # frequency are unverifiable against this pool — keeping them in the
    # denominator inflated the cutoff with phantom mass and let generic-
    # word papers slip through while unreachable place names punished
    # everything (2026-08-22). They are dropped and logged instead.
    dfs: dict[str, int] = {}
    for t, forms in term_forms.items():
        dfs[t] = sum(1 for p in papers if any(f in _doc_text(p) for f in forms))
    # A term needs >=2 pool confirmations to act as a scorer. On SMALL
    # pools df==1 generic prose ("selected", "area") wears an IDF crown
    # and can even become a HEAD term — so df==1 terms are dropped
    # whenever at least two multi-confirmed terms exist to score against;
    # below that, no statistics exist and the filter stands down.
    multi_confirmed = sum(1 for d in dfs.values() if d >= 2)

    idf: dict[str, float] = {}
    dropped_unverified: list[str] = []
    for t, forms in term_forms.items():
        df = dfs[t]
        if df == 0 or (df == 1 and multi_confirmed >= 2):
            dropped_unverified.append(t)
            continue
        idf[t] = math.log((n_docs + 1) / (df + 1)) + 1.0  # smoothed, ≥1

    if dropped_unverified:
        log.info(
            "Co-occurrence: %d query term(s) absent from whole pool — "
            "excluded from scoring: %s",
            len(dropped_unverified),
            ", ".join(dropped_unverified),
        )
    if not idf:
        # Pool shares ZERO verifiable query vocabulary — retrieval failure.
        # Reject loudly (empty list triggers the designed 2→1 re-scope loop)
        # instead of waving junk through (restores B1 pin semantics).
        log.warning("Co-occurrence: no query term found anywhere in pool")
        return []
    if len(idf) == 1:
        return papers

    total_mass = sum(idf.values())

    # Coverage bar: mass share AND distinct-term count. Mass alone over-
    # requires when several mid-frequency words split the denominator
    # (Title-Case queries ship 6-7 terms); the count guard keeps single-
    # common-word papers out.
    COVERAGE = 0.45
    MIN_MATCHED = 2
    if log.isEnabledFor(logging.DEBUG):
        for t_ in sorted(idf, key=idf.get, reverse=True):
            df_ = sum(
                1 for p in papers
                if any(f in _doc_text(p) for f in term_forms[t_])
            )
            log.debug("coocc term %-18s df=%d idf=%.2f", t_, df_, idf[t_])

    filtered = []
    rejected = 0
    for p in papers:
        text = _doc_text(p)
        matched_terms = [
            t for t, forms in term_forms.items()
            if t in idf and any(f in text for f in forms)
        ]
        matched_mass = sum(idf[t] for t in matched_terms)
        # Head-term rule: at least one match among the two highest-IDF
        # verified terms. Generic-word-only matches ("selected", "area")
        # can no longer clear the bar on dilution arithmetic alone.
        head_ok = True
        if idf and matched_terms:
            ranked_terms = sorted(idf, key=idf.get, reverse=True)
            head_set = set(ranked_terms[:2])
            head_ok = bool(head_set & set(matched_terms))
        elif not matched_terms:
            head_ok = False
        if (
            matched_mass >= COVERAGE * total_mass
            and len(matched_terms) >= MIN_MATCHED
            and head_ok
        ):
            filtered.append(p)
        else:
            rejected += 1

    if rejected > 0:
        log.info(
            "Co-occurrence filter (coverage %.0f%% of %.2f + >=2 terms): kept %d/%d, rejected %d",
            COVERAGE * 100,
            total_mass,
            len(filtered),
            len(papers),
            rejected,
        )
    return filtered


def _crossref_journal_search(query: str, max_results: int = 15) -> list:
    """Second Crossref pass (relevance-sorted duplicate of the main query).

    Historical note: this used to inject a hardcoded petrology-journal
    bibliographic boost into EVERY geoscience query, which starved
    hydro/geomorph/environmental topics of their true hits (2026-08-22).
    The boost is gone; the dynamic domain gate downstream owns relevance.
    """
    from habanero import Crossref

    try:
        cr = Crossref(mailto=_EMAIL)
        res = cr.works(
            query=query,
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
        # habanero works() is typed dict | list; query form returns dict.
        res_msg = res if isinstance(res, dict) else {}
        items = res_msg.get("message", {}).get("items", [])
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
            parts = item.get(date_key, {}).get("date-parts", [[]])[0] or []
            if not parts or not parts[0]:
                continue
            # JOL platforms (NepJOL etc.) deposit digitized back-issues
            # with exact (Y, 1, 1) online/issued placeholders — live case
            # 10.3126/bdg.v11i0.1544: issued 1970-01-01 while created is
            # 2008-12-05 and the abstract ends "Vol. 11, 2008" (2026-09-02).
            # Skip placeholders; fall back to `created` when nothing real
            # remains.
            if date_key != "published-print" and parts == [parts[0], 1, 1]:
                continue
            year = parts[0]
            break
        if year is None:
            _created = (item.get("created", {}).get("date-parts", [[]]) or [[]])[0]
            if _created and _created[0]:
                log.info(
                    "Crossref placeholder publication date on %s — using created year %s",
                    doi or (title[:60] if title else "unknown record"),
                    _created[0],
                )
                year = _created[0]

        # sources_seen: this paper was discovered via `source` AND enriched via Crossref.
        seen = [source, "crossref"] if source != "crossref" else ["crossref"]
        return PaperRecord(
            doi=doi,
            title=title,
            abstract=abstract,
            authors=authors,
            year=year,
            venue=item.get("container-title", [""])[0] if item.get("container-title") else "",
            type=item.get("type", ""),
            source=source,
            sources_seen=seen,
        )
    except Exception as e:
        log.debug("Crossref item conversion failed: %s", e)
        return None


def _clean_harvested_doi(raw: str) -> str:
    """Strip trailing punctuation/brackets that regex harvesting picks up."""
    doi = raw.rstrip(".,;:)\\]}\"'").lower()
    while doi and doi.count("(") > doi.count(")"):
        doi = doi[:-1]
    return doi


def _content_tokens(query: str) -> set[str]:
    """Content tokens (>=4 chars) of the query — dynamic, no vocabulary."""
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop = set(ENGLISH_STOP_WORDS)
    except ImportError:  # pragma: no cover - sklearn is a hard dep
        stop = set()

    return {tok for tok in re.findall(r"[a-z0-9]+", query.lower()) if len(tok) >= 4 and tok not in stop}


def _query_overlap_ok(record: object, query_tokens: set[str]) -> bool:
    """Dynamic relevance gate: record text must share >=1 content token.

    With <=3 query tokens (very narrow questions), require zero-tolerance
    is too strict — any single overlap still passes. No field lists used.
    Fuzzy tier (2026-09-03): a query token typo'd by one edit must still
    count — web search FINDS papers for typo'd queries via fuzzy results,
    then this gate hard-dropped them because substring overlap cannot see
    a 1-edit mismatch ('godawari' vs a 'godavari basin' title). Mirrors
    the screening matcher (fold + Damerau-Levenshtein<=1).
    """
    if not query_tokens:
        return True
    text = ((getattr(record, "title", "") or "") + " " + (getattr(record, "abstract", "") or "")).lower()
    if any(tok in text for tok in query_tokens):
        return True
    from _honesty import _dl_within1, fold_text

    text_toks = set(re.findall(r"[a-z0-9]+", fold_text(text)))
    for tok in query_tokens:
        ft = fold_text(tok)
        if any(
            abs(len(t2) - len(ft)) <= 1 and _dl_within1(ft, t2) for t2 in text_toks
        ):
            return True
    return False



def _locate_web_search_script() -> str:
    """Locate the OMP web-search skill CLI (standalone-skill fallback)."""
    candidates = (
        "~/.omp/agent/skills/web-search/web_search.py",
        "~/.config/opencode/skills/web-search/web_search.py",
    )
    for c in candidates:
        p = Path(os.path.expanduser(c))
        if p.exists():
            return str(p)
    return ""


def _web_text_engine(
    query: str, *, max_results: int = 15, call_budget: int = 90, no_cache: bool = False
) -> tuple[int, dict]:
    """Dual-mode web text search (2026-09-03 sync port).

    Runs the OMP web-search skill CLI (optional sibling; degrades to
    an error return when absent).
    Returns (rc, {"results": [...]}) shaped like the native engine.
    """
    import subprocess as _sp

    script = _locate_web_search_script()
    if not script or not os.path.exists(script):
        return 1, {"error": "web-search skill not found"}
    env = dict(os.environ)
    if no_cache:
        env["WEB_SEARCH_NO_CACHE"] = "1"
    try:
        proc = _sp.run(
            ["python3", script, "text", query, "--max-results", str(max_results),
             "--sort", "relevance", "--extract", "0", "--call-budget", str(call_budget)],
            capture_output=True, text=True, timeout=120, check=False, env=env,
        )
    except _sp.TimeoutExpired:
        return 1, {"error": "web-search subprocess timed out"}
    if proc.returncode != 0:
        return proc.returncode, {"error": proc.stderr[:200]}
    try:
        return 0, json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        return 1, {"error": f"non-JSON: {ex}"}


def web_search_paper_discovery(
    query: str,
    max_results: int = 15,
    force_refresh: bool = False,
    domain_hint: str = "",
) -> list:
    """Paper discovery via web search (web-search skill CLI bridge).

    In-process port of the web-search engine (2026-09-01): 8+ metasearch
    backends with rotation, 4-layer bot-block bypass (cookie/primp/
    curl_cffi/Playwright stealth — browser tier engages automatically),
    adaptive rate limiting, circuit breakers, TTL disk cache.

    Returns PaperRecords (resolved through Crossref for metadata).
    Always registered as a default source (DEFAULT_SOURCES); the
    --use-web-search flag force-adds it when `sources` was narrowed.
    Pass force_refresh=True to bypass the engine's disk cache.
    """
    try:
        # SCHOLARLY-ANCHORED query (2026-09-03, live: raw place-name
        # queries pulled Chandrapur-district spider/herbicide papers into
        # a geology candidate pool — the engine reranks WEB pages, not
        # papers, so place collisions ride through DOI harvesting). The
        # anchor is DYNAMIC: the detected research domain plus a scholarly
        # scope token, never a hardcoded vocabulary.
        _anchor = " ".join(
            t
            for t in (domain_hint.strip(), "research paper")
            if t and t.lower() not in query.lower()
        )
        web_query = f"{query} {_anchor}".strip() if _anchor else query
        if _anchor:
            log.info("web_search query anchored: %r", web_query[:90])
        rc, data = _web_text_engine(
            web_query,
            max_results=max_results * 3,
            call_budget=90,
            no_cache=force_refresh,
        )
    except _EngineError as e:
        # Source-level degrade: one blocked/failed source must not kill the
        # multi-source search (per-source isolation contract).
        log.warning("web_search source degraded (%s): %s", type(e).__name__, str(e)[:150])
        return []
    if rc == 3:
        # Engine "genuine empty": content floor dropped everything usable.
        log.info("web_search: 0 usable results for %s (content floor)", query[:60])
        return []
    if rc != 0:
        log.warning(
            "web-search engine failed (rc=%d): %s",
            rc,
            str(data.get("error") or data.get("note") or "unknown")[:200],
        )
        return []

    results = data.get("results", [])
    if not results:
        return []

    # Extract DOIs from result URLs/snippets/titles.
    # Snippets routinely cite OTHER papers' DOIs, so every harvested DOI is
    # later gated by a query-overlap check after Crossref resolution (F4).
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
    found_dois: set[str] = set()
    for r in results:
        text = f"{r.get('href', '')} {r.get('body', '')} {r.get('title', '')}"
        for m in doi_pattern.finditer(text):
            found_dois.add(_clean_harvested_doi(m.group()))

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

    cr = Crossref(mailto=_EMAIL)
    query_tokens = _content_tokens(query)
    dropped_offtopic = 0
    for doi in list(found_dois)[:max_results]:
        try:
            item = cr.works(ids=doi)
            # habanero works(ids=) typed dict | list; DOI path returns dict.
            if isinstance(item, dict) and item.get("message"):
                record = _crossref_item_to_record(item["message"], source="web_search")
                if record is None:
                    continue
                if not _query_overlap_ok(record, query_tokens):
                    dropped_offtopic += 1
                    continue
                resolved.append(record)
        except Exception:
            continue

    if dropped_offtopic:
        log.info(
            "web_search relevance gate: dropped %d citation-noise paper(s) (zero query-term overlap)",
            dropped_offtopic,
        )
    log.info("web_search: resolved %d/%d papers", len(resolved), len(found_dois))
    return resolved


def web_search_agentic_discovery(query: str, max_results: int = 15, force_refresh: bool = False) -> list:
    """Deep site-specific paper discovery via native engine agentic mode.

    Adaptive crawl discovery (unavailable standalone — see body). Uses, when present,
    semantically crawl the top search hit (typically a major repository:
    arxiv.org, biorxiv.org, publisher site) with the user's query as
    relevance filter. Returns ONLY pages that answer the query — high
    precision, lower recall than text mode.

    Slower than web_search_paper_discovery (browser + adaptive crawl,
    30-120s) but finds papers in JS-rendered sites, conference proceedings,
    niche journals, and behind search forms that text mode misses.

    Output includes `confidence` score (0-1) — semantic relevance of
    crawled content to the query.

    Activation: --use-web-search-agentic flag.
    """
    # Standalone skill: the adaptive-crawl tier is not part of this
    # implementation — text-mode discovery covers recall here.
    log.warning("web_search_agentic: crawl tier unavailable in standalone skill")
    return []

    try:
        rc, data = _engine_agentic(
            query,
            adaptive=True,
            max_iter=2,
            top_n_extract=3,
            call_budget=120,
            no_cache=force_refresh,
            quiet=True,
        )
    except _EngineError as e:
        log.warning("web_search_agentic source degraded (%s): %s", type(e).__name__, str(e)[:150])
        return []
    if rc != 0:
        log.warning(
            "web-search agentic failed (rc=%d): %s", rc, str(data.get("error"))[:200]
        )
        return []

    # Collect text from ALL result tiers: search hits, extracted pages, adaptive crawl
    text_sources: list[str] = []
    for r in data.get("results", []):
        text_sources.append(f"{r.get('href', '')} {r.get('body', '')} {r.get('title', '')}")
    for e in data.get("extracted", []):
        text_sources.append(e.get("content") or "")
    ap = data.get("adaptive", {})
    confidence = None
    if isinstance(ap, dict):
        confidence = ap.get("confidence")
        for p in ap.get("pages", []):
            text_sources.append(p.get("content") or "")

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

    cr = Crossref(mailto=_EMAIL)
    for doi in list(found_dois)[:max_results]:
        try:
            item = cr.works(ids=doi)
            # habanero works(ids=) typed dict | list; DOI path returns dict.
            if isinstance(item, dict) and item.get("message"):
                record = _crossref_item_to_record(item["message"], source="web_search_agentic")
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


# Geoscience query signals — when present in query, activate geo filtering


# Ambiguous terms shared between geology and physics/chemistry/biology


def _expand_search_query(query: str) -> str:
    """Return the query unchanged.

    2026-08-22 dynamic-context mandate: the former geology-ontology
    expansion (geo-signal gates + ambiguous-term append of eight geology
    words) injected static discipline context into API searches. Ranking,
    facet coverage and the topical gate own relevance dynamically now.
    Kept as identity for call-site stability.
    """
    return query


# =============================================================================
# Domain relevance filtering — removes off-topic papers (e.g., magnetic
# garnets when the query is about metamorphic garnet thermobarometry)
# =============================================================================


def _is_domain_relevant(paper: Any, query: str) -> bool:
    """Dynamic topical-consistency gate (field-agnostic since 2026-08-22).

    Replaces the former geology-signal tables: a paper is relevant iff it
    shares enough of THE QUERY'S OWN content vocabulary — at least 20% of
    the query's content tokens, minimum 2. No discipline word lists
    participate; the same rule governs every field.
    """
    if not query:
        return True
    text = (
        (getattr(paper, "title", "") or "")
        + " "
        + (getattr(paper, "abstract", "") or "")
    ).lower()
    q_tokens = sorted(_content_tokens(query))
    if not q_tokens:
        return True
    need = max(2, -(-len(q_tokens) * 20 // 100))  # ceil(20%), floor 2
    hits = sum(1 for tok in q_tokens if tok in text)
    if hits < need:
        log.debug(
            "Domain gate: rejected (%d/%d query-term hits): %.70s",
            hits,
            need,
            text.strip() or "<no title/abstract>",
        )
        return False
    return True


# =============================================================================
# Multi-source search
# =============================================================================


def reset_ddgs_circuit() -> None:
    """Reset the DDGS circuit breaker (tests, forks).

    Thin delegate to the rate-limit registry's ddgs breaker — kept HERE
    (not in _websearch) because the smoke harness and sibling tests import
    it from ``discover`` alongside the source resets (audit fix 2026-08-23:
    the symbol was expected but never defined anywhere).
    """
    from _ratelimits import get_registry

    get_registry().ddgs.reset()


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
    original_query: str | None = None,
    domain_hint: str = "",
) -> list[PaperRecord]:
    """Search across multiple sources IN PARALLEL, return merged list.

    Each source runs in its own thread with a hard timeout. One hanging
    source cannot block the pipeline — it is abandoned after
    per_source_timeout seconds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from concurrent.futures import TimeoutError as FuturesTimeout

    expanded_query = _expand_search_query(query)
    if original_query and filters is not None:
        # provenance for orchestrators: the USER query behind this
        # (possibly expanded) search — downstream co-occurrence/intent
        # filters should key on this, not the expanded string
        filters = dict(filters)
        filters["original_query"] = original_query

    # Sequence (covariant) - pyright infers the split() branch as list[LiteralString].
    active_sources: Sequence[str] = sources if sources else DEFAULT_SOURCES.split(",")
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
    if "crossref" in active_sources:
        source_calls["crossref"] = lambda: crossref_search(
            expanded_query, max_results=max_per_source, filter_dict=cr_filter or None
        )
    if "openalex" in active_sources:
        # Two-mode OpenAlex search: semantic (AI relevance) + keyword (filterable).
        # Semantic search uses OpenAlex embeddings for better recall; keyword
        # search supports year/OA/type filters. Run BOTH when filters are set
        # and merge results for maximum coverage + precision.
        if oa_filter or (filters and (filters.get("year_from") or filters.get("year_to"))):
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
            source_calls["openalex"] = lambda: openalex_semantic_search(expanded_query, max_results=max_per_source)
    if "s2" in active_sources:
        source_calls["s2"] = lambda: s2_search(expanded_query, max_results=max_per_source)
    if "eartharxiv" in active_sources:
        from _sources import eartharxiv_search

        source_calls["eartharxiv"] = lambda: eartharxiv_search(expanded_query, max_results=max_per_source)
    if "usgs" in active_sources:
        from _sources import usgs_search

        source_calls["usgs"] = lambda: usgs_search(expanded_query, max_results=max_per_source)
    if "arxiv" in active_sources:
        source_calls["arxiv"] = lambda: arxiv_search(expanded_query, max_results=max_per_source)
    if "epmc" in active_sources:
        from _sources import epmc_search

        source_calls["epmc"] = lambda: epmc_search(expanded_query, max_results=max_per_source)

    # Web-search discovery — PRIMARY source (2026-09-01: websearch engine
    # engine, in-process port). Registered via DEFAULT_SOURCES ("websearch");
    # the flag / env var force-add it when the caller narrowed `sources`.
    if (
        "web_search" in active_sources
        or use_web_search
        or os.environ.get("SCIENTIFIC_RESEARCH_ENABLE_WEB_SEARCH", "0") == "1"
    ):
        source_calls["web_search"] = lambda: web_search_paper_discovery(
            query,
            max_results=min(max_per_source, 15),
            force_refresh=force_refresh,
            domain_hint=domain_hint,
        )

    # Web-search AGENTIC integration — Crawl4AI AdaptiveCrawler on top hit.
    # Slower (browser + semantic crawl, 30-120s) but finds papers in JS-rendered
    # sites and behind search forms that text mode misses. Adds confidence score.
    if use_web_search_agentic:
        source_calls["web_search_agentic"] = lambda: web_search_agentic_discovery(
            query, max_results=min(max_per_source, 10), force_refresh=force_refresh
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
def snowball_backward(seed: list[PaperRecord], max_per_seed: int = 10, depth: int = 1) -> list[PaperRecord]:
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


def snowball_forward(seed: list[PaperRecord], max_per_seed: int = 10, depth: int = 1) -> list[PaperRecord]:
    """Forward snowballing: fetch papers that cite seeds.
    Uses OpenAlex cited_by_api_url + S2 forward citations."""
    out: list[PaperRecord] = []
    seen_ids = {p.primary_id for p in seed}
    for p in seed:
        citing: list[PaperRecord] = []
        if p.openalex_id:
            try:
                log.info("Forward OA cites for %s", p.doi or p.openalex_id)
                citing.extend(openalex_get_cited_by(p.openalex_id, max_results=max_per_seed))
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
    log.info("Query expansion: %d terms from %d seed papers", len(expanded), len(seed_texts))
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
        forward = snowball_forward(current_batch[:10], max_per_seed=max_per_seed, depth=1)
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
    p.add_argument("--max", type=int, default=100, help="max papers in final corpus (default 100)")
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
    p.add_argument("--sources", default=DEFAULT_SOURCES, help=f"comma-separated: {DEFAULT_SOURCES}")
    p.add_argument(
        "--semantic",
        action="store_true",
        help="use OpenAlex semantic search (paragraph-length query)",
    )
    p.add_argument(
        "--use-web-search",
        action="store_true",
        help="force-add websearch source when --sources narrowed it out "
        "(websearch is a DEFAULT source since 2026-09-01: native engine, "
        "8+ metasearch backends + bot-block bypass). Also activates via "
        "SCIENTIFIC_RESEARCH_ENABLE_WEB_SEARCH=1 env var.",
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
    p.add_argument("--explore", action="store_true", help="landscape mode: shallow scan, no dedup")
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
    p.add_argument("--from-year", type=int, default=None, help="earliest publication year")
    p.add_argument("--to-year", type=int, default=None, help="latest publication year")
    p.add_argument("--open-access-only", action="store_true", help="only discover OA papers")
    p.add_argument(
        "--type",
        default="",
        help="publication type filter (journal-article, book-chapter, etc.)",
    )
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    # force=True: library imports may install their own root handler,
    # which silently disabled -v entirely (only WARNING+ escaped).
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        force=True,
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
        max_per_source=int(args.max_per_source * 1.5) if args.require_abstract else args.max_per_source,
        sources=sources,
        semantic=args.semantic,
        force_refresh=args.force_refresh,
        filters=cli_filters or None,
        use_web_search=args.use_web_search,
        use_web_search_agentic=args.use_web_search_agentic,
    )
    log.info("Web: %d raw across sources", len(results))

    # Phase-0 context builder — dynamic query decomposition (W1 wiring).
    # Extra per-entity/pairwise probes run ONLY when the primary pass came
    # back thin relative to the ask; original-query-first stays untouched.
    _ctx = None
    try:
        from _context import build_research_context

        _ctx = build_research_context(args.query, use_network=True)
        # Raw-count heuristic: filtering typically collapses a pool 5-10x,
        # so a pool under ~4x the ask usually yields a thin final corpus.
        if (
            len(_ctx.search_strategies) > 1
            and len(results) < args.max * 4
        ):
            probe_sources = [s for s in sources if s in ("crossref", "openalex")]
            for strat in _ctx.search_strategies[1:3]:
                try:
                    extra = search_multi_source(
                        strat,
                        max_per_source=max(5, args.max_per_source // 2),
                        sources=probe_sources,
                        force_refresh=args.force_refresh,
                        filters=cli_filters or None,
                    )
                    log.info(
                        "Strategy probe %r: +%d candidates",
                        strat[:60],
                        len(extra),
                    )
                    results.extend(extra)
                except Exception as pe:
                    log.warning("Strategy probe failed %r: %s", strat[:60], pe)
            if _ctx.niche:
                log.info(
                    "Niche locality-anchored topic — consider --use-web-search "
                    "for regional-journal coverage"
                )
    except Exception as e:
        log.warning("Context builder unavailable — single-strategy search: %s", e)

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

    # Early abstract filter (R1 reorder 2026-08-21): title-only seeds must
    # not drive snowballing/expansion — they waste API calls and pull
    # reference-graph noise. Idempotent; the post-merge filter below still
    # runs to catch no-abstract snowball/expansion arrivals.
    if args.require_abstract and final:
        before = len(final)
        final = [p for p in final if p.abstract and len(p.abstract.strip()) > 50]
        log.info(
            "Early abstract filter: %d → %d (removed %d without abstracts)",
            before,
            len(final),
            before - len(final),
        )

    # Snowball (Cohen 2018)
    if args.snowball and final:
        if args.snowball_saturation:
            log.info("=== SATURATION SNOWBALLING (max_depth=%d) ===", args.snowball_depth_max)
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
        expanded_terms = auto_expand_query(final[: args.snowball_max * 2], max_terms=args.expand_max_terms)
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

    # Post-merge abstract filter (snowball/expansion arrivals)
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
    if final and not args.explore:
        before_domain = len(final)
        final = [p for p in final if _is_domain_relevant(p, args.query)]
        if len(final) < before_domain:
            log.info(
                "Domain filter: %d → %d (removed %d off-domain)",
                before_domain,
                len(final),
                before_domain - len(final),
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
    # Ranking runs at EVERY corpus size — the dynamic relevance floor owns
    # junk exclusion on tiny corpora too (polysemy strays previously rode
    # in via the citation-sort shortcut).
    if final:
        from _ranking import rank_papers

        log.info("=== RANKING %d papers (top %d) ===", len(final), args.max)
        ranked = rank_papers(args.query, final, max_n=args.max)
        final = [p for p, _, _ in ranked]
        log.info("Ranked: %d papers (relevance floor + diversity)", len(final))

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

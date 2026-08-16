#!/usr/bin/env python3
"""Shared source wrappers, cache, dedup, retry for the scientific-research skill.

Wraps 4 academic APIs (Crossref, OpenAlex, Semantic Scholar, arXiv) into a
unified ``PaperRecord`` schema. Provides content-hash JSON cache for cross-run
deduplication, exponential-backoff retry, and a compact summary renderer that
keeps LLM-facing output small (~150 tokens/paper).

Fail-loud dep check (AGENTS.md §5 H2). No silent fallback.

Dependencies (install in any project venv that uses the skill):
    uv add habanero pyalex semanticscholar arxiv

Cache layout:
    ~/.cache/scientific-research/<sha256(identifier)>/paper.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Run-scoped rate-limit circuit breakers + centralized timeouts (Tier 3 wiring).
# Replaces the module-level breaker globals that persisted across pipeline runs,
# causing failure cascades. See _ratelimits.py for the SOTA design rationale.
from _ratelimits import CircuitBreaker, get_registry
from _timeouts import TIMEOUTS

# --- Dependency check (fail loud per §5 H2) ---
_MISSING: list[str] = []
try:
    from habanero import Crossref, cn, counts  # type: ignore
except ImportError:
    _MISSING.append("habanero")
try:
    import pyalex  # type: ignore
    from pyalex import Works  # type: ignore
except ImportError:
    _MISSING.append("pyalex")
try:
    from semanticscholar import SemanticScholar  # type: ignore
except ImportError:
    _MISSING.append("semanticscholar")
try:
    import arxiv  # type: ignore
except ImportError:
    _MISSING.append("arxiv")

if _MISSING:
    raise ImportError(
        "scientific-research requires: " + ", ".join(_MISSING) + ".\n"
        "Install:  uv add " + " ".join(_MISSING) + "\n"
        "Or:       pip install " + " ".join(_MISSING)
    )

# --- Polite-pool configuration ---
_EMAIL = os.environ.get("SCIENTIFIC_RESEARCH_EMAIL", "researcher@example.com")
# pyalex 0.21+: use `pyalex.config` dict (older versions used `pyalex.settings`)
_pyalex_cfg = getattr(pyalex, "config", None) or getattr(pyalex, "settings", None)
if _pyalex_cfg is not None:
    try:
        _pyalex_cfg["email"] = _EMAIL
        _pyalex_cfg["max_retries"] = 3
    except (TypeError, KeyError):
        _pyalex_cfg.email = _EMAIL
        _pyalex_cfg.max_retries = 3

log = logging.getLogger("scientific_research.sources")

# --- Knowledge base directory (permanent, XDG data) ---
CACHE_DIR = Path(
    os.environ.get(
        "SCIENTIFIC_RESEARCH_KB",
        str(Path.home() / ".local" / "share" / "scientific-research"),
    )
)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PaperRecord — unified schema across all sources
# =============================================================================
@dataclass
class PaperRecord:
    """Unified paper record across Crossref/OpenAlex/S2/arXiv sources."""

    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    s2_paper_id: str | None = None
    pmid: str | None = None
    title: str = ""
    abstract: str = ""
    full_text: str | None = None  # optional full text (geokit PDF pipeline)
    authors: list[dict] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    publication_date: str = ""
    type: str = ""
    type_crossref: str = ""
    language: str = ""
    citation_count: int | None = None
    influential_citation_count: int | None = None
    referenced_works_count: int | None = None
    is_retracted: bool = False
    is_open_access: bool = False
    oa_pdf_url: str | None = None
    concepts: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    referenced_works: list[str] = field(default_factory=list)
    cited_by_api_url: str | None = None
    funders: list[dict] = field(default_factory=list)
    awards: list[str] = field(default_factory=list)
    mesh: list[str] = field(default_factory=list)
    source: str = ""
    sources_seen: list[str] = field(default_factory=list)
    local_path: str | None = None
    extraction_method: str = ""
    raw_metadata: dict = field(default_factory=dict)
    fetched_at: str = ""

    @property
    def primary_id(self) -> str:
        """Best identifier for dedup / cache key (deterministic, stable)."""
        if self.doi:
            return "doi:" + self.doi.lower()
        if self.arxiv_id:
            return "arxiv:" + self.arxiv_id
        if self.pmid:
            return "pmid:" + str(self.pmid)
        if self.openalex_id:
            return self.openalex_id
        if self.s2_paper_id:
            return "s2:" + self.s2_paper_id
        h = hashlib.sha256(self.title.lower().encode()).hexdigest()[:16]
        return "title:" + h

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PaperRecord:
        known = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# Cache helpers
# =============================================================================
def cache_key(identifier: str) -> Path:
    h = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return CACHE_DIR / h / "paper.json"


def cache_get(identifier: str) -> PaperRecord | None:
    p = cache_key(identifier)
    if p.exists():
        try:
            return PaperRecord.from_dict(json.loads(p.read_text()))
        except Exception as e:
            log.warning("cache read failed for %s: %s", identifier, e)
    return None


def cache_put(paper: PaperRecord) -> None:
    p = cache_key(paper.primary_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            _sanitize_json(paper.to_dict()), indent=2, ensure_ascii=False, default=str
        )
    )


def _sanitize_json(obj: Any) -> Any:
    """Recursively convert non-JSON-native types (datetime, sets, custom
    objects) to JSON-safe equivalents. Prevents serialization crashes when
    upstream APIs (S2 library) return rich objects in raw_metadata."""
    import datetime

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return obj.decode("latin-1", errors="replace")
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(x) for x in obj]
    # Last resort: stringify unknown objects
    try:
        return str(obj)
    except Exception as e:
        log.debug("JSON sanitization failed for %s: %s", type(obj).__name__, e)
        return None


def cache_has(identifier: str) -> bool:
    return cache_key(identifier).exists()


# =============================================================================
# Retry with explicit policy (max attempts, backoff, jitter — §5 H16)
# Rate-limit aware: checks Retry-After header and 429 status codes
# =============================================================================
def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter_max: float = 0.25,
    rate_limit_aware: bool = True,
):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            attempt, last_err = 0, None
            while attempt < max_attempts:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    attempt += 1
                    if attempt >= max_attempts:
                        break

                    # Calculate delay
                    delay = base_delay * (backoff_factor ** (attempt - 1))

                    # Rate-limit awareness: check for 429 / Retry-After
                    if rate_limit_aware:
                        err_str = str(e).lower()
                        if (
                            "429" in err_str
                            or "rate" in err_str
                            or "too many" in err_str
                        ):
                            # Rate limited — use longer delay
                            delay = max(
                                delay, 3.0
                            )  # circuit breakers are primary defense
                            log.warning(
                                "RATE LIMITED on %s (attempt %d/%d) — waiting %.1fs",
                                fn.__name__,
                                attempt,
                                max_attempts,
                                delay,
                            )
                        elif "timeout" in err_str or "timed out" in err_str:
                            # Timeout — use longer delay
                            delay = max(delay, 5.0)

                    delay += jitter_max * ((hash(str(e)) % 1000) / 1000.0)
                    log.warning(
                        "attempt %d/%d failed for %s: %s (retry in %.1fs)",
                        attempt,
                        max_attempts,
                        fn.__name__,
                        e,
                        delay,
                    )
                    time.sleep(delay)
            raise RuntimeError(
                f"{fn.__name__} failed after {max_attempts} attempts"
            ) from last_err

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


# =============================================================================
# Crossref (habanero)
# =============================================================================
_crossref_client = Crossref(mailto=_EMAIL, timeout=TIMEOUTS.crossref_doi)


def _strip_jats(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _crossref_to_record(item: dict) -> PaperRecord:
    authors: list[dict] = []
    for a in item.get("author", []):
        name = " ".join(filter(None, [a.get("given"), a.get("family")]))
        if name:
            authors.append(
                {
                    "name": name,
                    "orcid": (a.get("ORCID") or "").replace("http://orcid.org/", "")
                    or None,
                    "affiliation": [
                        af.get("name")
                        for af in a.get("affiliation", [])
                        if af.get("name")
                    ],
                }
            )
    issued = item.get("issued", {}).get("date-parts", [[None]])
    year = issued[0][0] if issued and issued[0] else None
    funders = [
        {"name": f.get("name"), "doi": f.get("DOI"), "award": f.get("award", [])}
        for f in item.get("funder", [])
    ]
    type_cr = item.get("type", "") or ""
    return PaperRecord(
        doi=item.get("DOI"),
        title=(item.get("title") or [""])[0],
        abstract=_strip_jats(item.get("abstract", "")),
        authors=authors,
        year=year,
        venue=(item.get("container-title") or [""])[0],
        publication_date=(
            item.get("published-print", {}).get("date-parts", [[""]])[0][0]
            if item.get("published-print")
            else ""
        ),
        type=type_cr,
        type_crossref=type_cr,
        language=item.get("language", ""),
        funders=funders,
        awards=[a for f in item.get("funder", []) for a in f.get("award", [])],
        referenced_works_count=item.get("references-count"),
        is_retracted=(
            "retraction" in type_cr.lower()
            or any(
                r.get("type") == "retraction"
                for r in (item.get("relation", {}) or {}).get("is-review-of", [])
            )
        ),
        source="crossref",
        sources_seen=["crossref"],
        raw_metadata=item,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


@retry_with_backoff(max_attempts=3)
def crossref_search(
    query: str, max_results: int = 25, filter_dict: dict | None = None
) -> list[PaperRecord]:
    """Search Crossref works. ``filter_dict`` follows Crossref filter syntax.

    Respects Crossref circuit breaker — returns empty list if open.
    Enforces minimum interval between calls to stay in polite pool.
    Opens circuit breaker on consecutive 429s (shared with DOI resolution).
    """
    cb = _crossref_cb()
    if not cb.available:
        log.debug("Crossref circuit open — skipping search")
        return []
    cb.pace()
    try:
        res = _crossref_client.works(
            query=query, limit=max_results, filter=filter_dict or {}
        )
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "too many" in err_str:
            cb.record_failure()
        raise
    cb.record_success()
    return [_crossref_to_record(it) for it in res.get("message", {}).get("items", [])]


# Crossref circuit breaker — delegates to run-scoped RateLimitRegistry.
# Threshold + min_interval live in _ratelimits.RateLimitRegistry.crossref.


def _crossref_cb() -> CircuitBreaker:
    """Active Crossref circuit breaker from the run-scoped registry."""
    return get_registry().crossref


def reset_crossref_circuit() -> None:
    """Reset Crossref circuit breaker for new pipeline runs (backward-compat wrapper)."""
    _crossref_cb().reset()


def crossref_resolve_doi(doi: str) -> PaperRecord | None:
    """Resolve a DOI via Crossref. Returns None if not found / unresolved.

    Circuit breaker opens after consecutive 429s to avoid hammering a
    rate-limited API.
    """
    cb = _crossref_cb()
    if not cb.available:
        return None
    try:
        res = _crossref_client.works(ids=doi)
        cb.record_success()
        return _crossref_to_record(res["message"])
    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            cb.record_failure()
        log.warning("Crossref DOI resolve failed for %s: %s", doi, e)
        return None


# =============================================================================
# OpenAlex (pyalex + raw API)
# =============================================================================
_OPENALEX_SELECT = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "publication_date",
        "ids",
        "language",
        "type",
        "type_crossref",
        "open_access",
        "authorships",
        "cited_by_count",
        "referenced_works_count",
        "referenced_works",
        "cited_by_api_url",
        "is_retracted",
        "best_oa_location",
        "concepts",
        "topics",
        "keywords",
        "mesh",
        "funders",
        "awards",
        "primary_location",
        "abstract_inverted_index",
    ]
)


def _openalex_reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)


def _openalex_to_record(item: dict) -> PaperRecord:
    authors: list[dict] = []
    for a in item.get("authorships", []):
        author = a.get("author") or {}
        name = author.get("display_name", "")
        if name:
            authors.append(
                {
                    "name": name,
                    "orcid": (author.get("orcid") or "").replace(
                        "https://orcid.org/", ""
                    )
                    or None,
                    "openalex_id": author.get("id"),
                    "affiliation": [
                        inst.get("display_name")
                        for inst in a.get("institutions", [])
                        if inst.get("display_name")
                    ],
                    "countries": a.get("countries", []),
                }
            )
    oa = item.get("open_access") or {}
    best_oa = item.get("best_oa_location") or {}
    concepts = [
        c.get("display_name", "")
        for c in item.get("concepts", [])
        if c.get("display_name")
    ]
    topics = [
        t.get("display_name", "")
        for t in item.get("topics", [])
        if t.get("display_name")
    ]
    keywords = [
        k.get("display_name", "")
        for k in (item.get("keywords") or [])
        if k.get("display_name")
    ]
    mesh = [
        m.get("descriptor_name", "")
        for m in (item.get("mesh") or [])
        if m.get("descriptor_name")
    ]
    funders = [
        {"name": f.get("display_name"), "openalex_id": f.get("id"), "doi": f.get("doi")}
        for f in (item.get("funders") or [])
    ]
    doi_raw = item.get("doi") or ""
    doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
    primary = item.get("primary_location") or {}
    venue = (primary.get("source") or {}).get("display_name", "") if primary else ""
    return PaperRecord(
        doi=doi,
        openalex_id=item.get("id"),
        title=item.get("title") or item.get("display_name", ""),
        abstract=_openalex_reconstruct_abstract(item.get("abstract_inverted_index")),
        authors=authors,
        year=item.get("publication_year"),
        venue=venue,
        publication_date=item.get("publication_date", "") or "",
        type=item.get("type", ""),
        type_crossref=item.get("type_crossref", ""),
        language=item.get("language", ""),
        citation_count=item.get("cited_by_count"),
        referenced_works_count=item.get("referenced_works_count"),
        referenced_works=item.get("referenced_works", []),
        cited_by_api_url=item.get("cited_by_api_url"),
        is_retracted=bool(item.get("is_retracted", False)),
        is_open_access=bool(oa.get("is_oa", False)),
        oa_pdf_url=best_oa.get("pdf_url"),
        concepts=concepts + topics,
        keywords=keywords,
        mesh=mesh,
        funders=funders,
        awards=list(item.get("awards") or []),
        source="openalex",
        sources_seen=["openalex"],
        raw_metadata=item,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# OpenAlex rate limiting + circuit breaker (delegates to run-scoped registry).
_OPENALEX_EMAIL = "scientific-research@geokit.dev"  # polite pool


def _openalex_cb() -> CircuitBreaker:
    """Active OpenAlex circuit breaker from the run-scoped registry."""
    return get_registry().openalex


def _openalex_raw_get(url: str) -> dict | None:
    cb = _openalex_cb()

    # Circuit breaker — stop if too many consecutive 429s
    if not cb.available:
        return None

    # Rate limiting — enforce minimum interval between calls
    cb.pace()

    # Add polite email to URL for higher rate limit
    if "?" in url:
        url = url + "&mailto=" + _OPENALEX_EMAIL
    else:
        url = url + "?mailto=" + _OPENALEX_EMAIL

    import httpx

    try:
        r = httpx.get(url, timeout=TIMEOUTS.crossref_search)
    except Exception as e:
        log.warning("OpenAlex request failed: %s", e)
        return None

    if r.status_code == 404:
        return None
    if r.status_code == 429:
        cb.record_failure()
        if not cb.available:
            return None
        # Wait and retry once
        retry_after = min(
            float(r.headers.get("retry-after", "5")), 5.0
        )  # cap at 5s — fail fast
        log.warning("OpenAlex 429 — waiting %.0fs", retry_after)
        time.sleep(retry_after)
        try:
            r = httpx.get(url, timeout=TIMEOUTS.crossref_search)
        except Exception as e:
            log.debug("OpenAlex retry request failed: %s", e)
            return None
        if r.status_code == 429:
            return None

    # Success — reset counter
    cb.record_success()
    try:
        r.raise_for_status()
    except Exception as e:
        log.warning("OpenAlex error: %s", e)
        return None
    return r.json()


def reset_openalex_circuit() -> None:
    """Reset OpenAlex circuit breaker for new pipeline runs (backward-compat wrapper)."""
    _openalex_cb().reset()


@retry_with_backoff(max_attempts=3)
def openalex_search(
    query: str,
    max_results: int = 25,
    filters: dict | None = None,
    force_refresh: bool = False,
) -> list[PaperRecord]:
    """Search OpenAlex works with cursor pagination for large result sets.

    Checks search cache first — if this query was searched recently (<30 days),
    loads paper IDs from cache and fetches records from paper cache (0 API calls).
    """
    # Check knowledge base
    try:
        from _search_cache import search_cache_get, search_cache_put

        cached_ids, age_days = search_cache_get(
            query, "openalex", max_results, force_refresh=force_refresh
        )
        if cached_ids:
            papers = [cache_get(pid) for pid in cached_ids if cache_has(pid)]
            if papers and len(papers) >= len(cached_ids) * 0.8:
                log.info(
                    "Knowledge base HIT: '%s' → %d papers (%d days old, 0 API calls)",
                    query[:40],
                    len(papers),
                    age_days,
                )
                return papers
            else:
                log.debug(
                    "Search cache partial hit: %d/%d papers in cache",
                    len(papers) if papers else 0,
                    len(cached_ids),
                )
    except ImportError:
        pass

    # API fetch
    if max_results <= 200:
        q = Works().search(query)
        if filters:
            q = q.filter(**filters)
        q = q.select(_OPENALEX_SELECT)
        items = q.get(per_page=min(max_results, 200))
        records = [_openalex_to_record(it) for it in items[:max_results]]
        for p in records:
            cache_put(p)
        try:
            search_cache_put(
                query, "openalex", max_results, [p.primary_id for p in records]
            )
        except (ImportError, NameError):
            pass
        return records

    # Cursor pagination for large result sets
    all_items = []
    cursor = "*"
    per_page = 200
    while len(all_items) < max_results:
        try:
            params = {
                "search": query,
                "per_page": per_page,
                "cursor": cursor,
                "select": _OPENALEX_SELECT,
            }
            if filters:
                params["filter"] = ",".join(f"{k}:{v}" for k, v in filters.items())

            url = "https://api.openalex.org/works?" + "&".join(
                f"{k}={v}" for k, v in params.items()
            )
            data = _openalex_raw_get(url)
            if not data:
                break

            results = data.get("results", [])
            if not results:
                break

            all_items.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            log.debug("OpenAlex pagination: fetched %d/%d", len(all_items), max_results)
        except Exception as e:
            log.warning("OpenAlex pagination error: %s", e)
            break

    records = [_openalex_to_record(it) for it in all_items[:max_results]]
    # Cache results
    for p in records:
        cache_put(p)
    try:
        search_cache_put(
            query, "openalex", max_results, [p.primary_id for p in records]
        )
    except (ImportError, NameError):
        pass
    return records


@retry_with_backoff(max_attempts=3)
def openalex_semantic_search(text: str, max_results: int = 25) -> list[PaperRecord]:
    """Semantic search via OpenAlex AI embeddings (paragraph-length queries)."""
    q = Works().similar(text).select(_OPENALEX_SELECT)
    # B4: semantic endpoint rejects per_page > 50 (snowball passed 75+ →
    # 3-attempt failure loop). Clamp here; pagination not supported by
    # .similar() anyway.
    items = q.get(per_page=min(max_results, 50))
    return [_openalex_to_record(it) for it in items[:max_results]]


@retry_with_backoff(max_attempts=3)
def openalex_get_by_doi(doi: str) -> PaperRecord | None:
    # Check paper cache first
    if cache_has(doi):
        cached = cache_get(doi)
        if cached and cached.abstract:
            return cached
    url = "https://api.openalex.org/works/doi:" + doi + "?select=" + _OPENALEX_SELECT
    try:
        d = _openalex_raw_get(url)
        record = _openalex_to_record(d) if d else None
        if record:
            cache_put(record)
        return record
    except Exception as e:
        log.warning("OpenAlex DOI lookup failed for %s: %s", doi, e)
        return None


@retry_with_backoff(max_attempts=3)
def openalex_get_by_id(openalex_id: str) -> PaperRecord | None:
    url = (
        "https://api.openalex.org/works/" + openalex_id + "?select=" + _OPENALEX_SELECT
    )
    try:
        d = _openalex_raw_get(url)
        return _openalex_to_record(d) if d else None
    except Exception as e:
        log.warning("OpenAlex ID lookup failed for %s: %s", openalex_id, e)
        return None


@retry_with_backoff(max_attempts=3)
def openalex_get_cited_by(openalex_id: str, max_results: int = 50) -> list[PaperRecord]:
    """Forward citations via filter=cites:<W_id>."""
    wid = openalex_id.replace("https://openalex.org/", "")
    url = (
        "https://api.openalex.org/works?filter=cites:"
        + wid
        + "&per-page="
        + str(max_results)
        + "&select="
        + _OPENALEX_SELECT
    )
    try:
        d = _openalex_raw_get(url) or {}
        return [_openalex_to_record(it) for it in d.get("results", [])]
    except Exception as e:
        log.warning("OpenAlex cited_by failed for %s: %s", openalex_id, e)
        return []


# =============================================================================
# Semantic Scholar
# =============================================================================
_s2_client = SemanticScholar(
    timeout=TIMEOUTS.semantic_scholar,
    api_key=os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
)
# S2 rate limit: 100 req/5min unauthenticated, 5000 req/5min with key.
# Pacing interval now lives in _ratelimits.RateLimitRegistry.s2.min_interval
# (computed via _s2_min_interval, respects S2_API_KEY env).

_S2_FIELDS = (
    "title,abstract,tldr,year,venue,authors,citationCount,"
    "influentialCitationCount,referenceCount,openAccessPdf,"
    "fieldsOfStudy,publicationTypes,externalIds,journal,"
    "publicationDate,isOpenAccess"
)

# Subset valid for citation/reference endpoints. S2 API rejects `tldr`,
# `openAccessPdf`, `journal` on nested paper objects (citingPaper.* / paper.*).
# Verified live 2026-07-08: passing 'citingPaper.tldr' returns 400.
_S2_CITATION_FIELDS = (
    "title,abstract,year,venue,authors,citationCount,"
    "influentialCitationCount,referenceCount,fieldsOfStudy,"
    "publicationTypes,externalIds,publicationDate,isOpenAccess"
)

# ── S2 Circuit Breaker — delegates to run-scoped RateLimitRegistry ────────
# After threshold consecutive failures, ALL S2 calls return empty results
# instantly — no API hit, no retry, no pacing delay. This prevents cascading
# 30s+ timeouts when S2 is unreachable. State lives in get_registry().s2.


def _s2_cb() -> CircuitBreaker:
    """Active S2 circuit breaker from the run-scoped registry."""
    return get_registry().s2


def _s2_available() -> bool:
    """Check if S2 circuit is closed (available). Backward-compat wrapper."""
    return _s2_cb().available


def _s2_record_success() -> None:
    """Reset failure counter on successful S2 call. Backward-compat wrapper."""
    _s2_cb().record_success()


def _s2_record_failure() -> None:
    """Track failures and open circuit after threshold. Backward-compat wrapper."""
    _s2_cb().record_failure()


def reset_s2_circuit() -> None:
    """Reset circuit breaker for rate-limit recovery.
    Does NOT undo force_skip_s2() — user preference persists."""
    _s2_cb().reset()


def enable_s2_for_enrichment() -> None:
    """Re-enable S2 for abstract enrichment IF it was force-skipped.

    Only clears force_skip — does NOT reset a circuit that opened from
    actual 429 rate-limiting. If S2 rate-limited us during discovery,
    it will rate-limit us during enrichment too.
    """
    _s2_cb().enable()


def force_skip_s2() -> None:
    """Skip ALL S2 calls for the rest of the session.
    Used when user didn't select S2 as a source."""
    _s2_cb().force_skip()


def _obj_to_dict(o: Any) -> dict:
    if isinstance(o, dict):
        return dict(o)
    if hasattr(o, "__dict__") and not isinstance(o, type):
        return {
            k: getattr(o, k, None)
            for k in dir(o)
            if not k.startswith("_") and not callable(getattr(o, k, None))
        }
    return {}


def _s2_to_record(p: Any) -> PaperRecord:
    d = _obj_to_dict(p)
    ext = d.get("externalIds") or {}
    tldr = d.get("tldr") or {}
    abstract = d.get("abstract") or ""
    if not abstract and isinstance(tldr, dict):
        abstract = tldr.get("text", "")
    oa_pdf = d.get("openAccessPdf") or {}
    authors: list[dict] = []
    for a in d.get("authors") or []:
        ad = _obj_to_dict(a)
        name = ad.get("name", "")
        if name:
            authors.append({"name": name, "s2_author_id": ad.get("authorId")})
    return PaperRecord(
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        s2_paper_id=d.get("paperId"),
        pmid=str(ext.get("PubMed")) if ext.get("PubMed") else None,
        title=d.get("title", ""),
        abstract=abstract,
        authors=authors,
        year=d.get("year"),
        venue=d.get("venue", "") or "",
        publication_date=d.get("publicationDate", "") or "",
        type=", ".join(d.get("publicationTypes") or []),
        citation_count=d.get("citationCount"),
        influential_citation_count=d.get("influentialCitationCount"),
        referenced_works_count=d.get("referenceCount"),
        is_open_access=bool(d.get("isOpenAccess", False)),
        oa_pdf_url=oa_pdf.get("url") if isinstance(oa_pdf, dict) else None,
        fields_of_study=list(d.get("fieldsOfStudy") or []),
        source="s2",
        sources_seen=["s2"],
        raw_metadata=d,
        fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


@retry_with_backoff(max_attempts=1)
def s2_search(query: str, max_results: int = 25) -> list[PaperRecord]:
    """Search Semantic Scholar via direct REST API (no library retries)."""
    if not _s2_available():
        return []
    _s2_pace()
    try:
        import requests as _requests

        resp = _requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": max_results, "fields": _S2_FIELDS},
            timeout=TIMEOUTS.semantic_scholar,
        )
        if resp.status_code == 429:
            _s2_record_failure()
            log.warning("S2 search rate-limited (429) for '%s'", query[:50])
            return []
        resp.raise_for_status()
        data = resp.json().get("data", [])
        _s2_record_success()
        return [_s2_to_record(p) for p in data]
    except Exception as e:
        _s2_record_failure()
        log.warning("S2 search failed for '%s': %s", query[:50], e)
        return []


def _s2_pace() -> None:
    """Sleep to respect S2 rate limit (delegates to run-scoped breaker pacing)."""
    _s2_cb().pace()


@retry_with_backoff(max_attempts=1)
def s2_get_paper(identifier: str) -> PaperRecord | None:
    """Identifier formats: 'DOI:X', 'ArXiv:X', 'CorpusId:X', 'PMID:X', or paperId.
    Single attempt — S2 library retries 429s internally."""
    if not _s2_available():
        return None
    # Normalize arXiv IDs: strip version suffix (v1, v2) — S2 indexes base IDs only
    if identifier.upper().startswith("ARXIV:"):
        base = identifier.split(":", 1)[1]
        base = re.sub(r"v\d+$", "", base)  # strip trailing v1, v2, etc.
        identifier = "ARXIV:" + base
    _s2_pace()
    try:
        import requests as _requests

        resp = _requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{identifier}",
            params={"fields": _S2_FIELDS},
            timeout=TIMEOUTS.semantic_scholar,
        )
        if resp.status_code == 429:
            _s2_record_failure()
            log.warning("S2 paper lookup rate-limited (429) for %s", identifier)
            return None
        if resp.status_code == 404:
            # Paper not in S2 database — normal, NOT a transient failure
            log.debug("S2 paper not found (404) for %s", identifier)
            return None
        resp.raise_for_status()
        _s2_record_success()
        return _s2_to_record(resp.json())
    except Exception as e:
        err_str = str(e)
        if "404" in err_str:
            # HTTPError from raise_for_status — paper not found, not a failure
            log.debug("S2 paper not found (404) for %s", identifier)
            return None
        _s2_record_failure()
        log.warning("S2 paper lookup failed for %s: %s", identifier, e)
        return None


@retry_with_backoff(max_attempts=1)
def s2_get_papers_batch(identifiers: list[str]) -> dict[str, PaperRecord]:
    """Batch-resolve multiple paper IDs in ONE API call.

    Uses S2's POST /paper/batch endpoint (up to 500 IDs per request).
    Returns {identifier: PaperRecord} for found papers. Missing papers
    are omitted from the dict. This replaces N individual s2_get_paper
    calls with 1 batch call — avoids rate-limit 429s entirely.

    Identifier formats: 'DOI:X', 'ArXiv:X', 'CorpusId:X', 'PMID:X', or paperId.
    """
    if not identifiers or not _s2_available():
        return {}
    _s2_pace()
    try:
        import requests as _requests

        # S2 batch accepts up to 500 IDs — chunk if needed
        all_results: dict[str, PaperRecord] = {}
        for chunk_start in range(0, len(identifiers), 500):
            chunk = identifiers[chunk_start : chunk_start + 500]
            resp = _requests.post(
                "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": _S2_FIELDS},
                json={"ids": chunk},
                timeout=TIMEOUTS.crossref_search,
            )
            if resp.status_code == 429:
                _s2_record_failure()
                log.warning("S2 batch rate-limited (429) for %d IDs", len(chunk))
                return all_results  # return partial results
            resp.raise_for_status()
            _s2_record_success()
            data = resp.json()
            # Response is array of paper objects (null for not-found)
            for i, paper_data in enumerate(data or []):
                if paper_data is None:
                    continue  # paper not in S2
                record = _s2_to_record(paper_data)
                all_results[chunk[i]] = record
        log.info(
            "S2 batch: resolved %d/%d papers in %d API call(s)",
            len(all_results),
            len(identifiers),
            (len(identifiers) + 499) // 500,
        )
        return all_results
    except Exception as e:
        _s2_record_failure()
        log.warning("S2 batch lookup failed: %s — falling back to individual", e)
        return {}


@retry_with_backoff(max_attempts=1)
def s2_get_references(paper_id: str, max_results: int = 50) -> list[PaperRecord]:
    """Backward citations (papers this paper references).
    S2 API uses `citedPaper.X` field prefix; Python lib wraps it as `.paper` attr."""
    if not _s2_available():
        return []
    _s2_pace()
    try:
        fields = [f"citedPaper.{f}" for f in _S2_CITATION_FIELDS.split(",")]
        import requests as _requests

        resp = _requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
            params={"fields": ",".join(fields), "limit": max_results},
            timeout=TIMEOUTS.semantic_scholar,
        )
        if resp.status_code == 429:
            _s2_record_failure()
            log.warning("S2 references rate-limited (429) for %s", paper_id)
            return []
        resp.raise_for_status()
        _s2_record_success()
        out: list[PaperRecord] = []
        for r in (resp.json().get("data") or [])[:max_results]:
            cp = r.get("citedPaper") or r.get("paper") or r
            if cp:
                out.append(_s2_to_record(cp))
        return out
    except Exception as e:
        _s2_record_failure()
        log.warning("S2 references failed for %s: %s", paper_id, e)
        return []


@retry_with_backoff(max_attempts=1)
def s2_get_citations(
    paper_id: str, max_results: int = 50
) -> list[tuple[PaperRecord, str]]:
    """Forward citations via direct REST API (no library retries)."""
    if not _s2_available():
        return []
    _s2_pace()
    try:
        import requests as _requests

        fields = ",".join(
            [f"citingPaper.{f}" for f in _S2_CITATION_FIELDS.split(",")]
            + ["contexts", "intents"]
        )
        resp = _requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
            params={"fields": fields, "limit": max_results},
            timeout=TIMEOUTS.semantic_scholar,
        )
        if resp.status_code == 429:
            _s2_record_failure()
            log.warning("S2 citations rate-limited (429) for %s", paper_id)
            return []
        resp.raise_for_status()
        _s2_record_success()
        out: list[tuple[PaperRecord, str]] = []
        for r in (resp.json().get("data") or [])[:max_results]:
            cp = r.get("citingPaper") or r.get("paper") or r
            if not cp:
                continue
            contexts = r.get("contexts") or []
            intents = r.get("intents") or []
            ctx_str = " ".join(contexts) if contexts else ""
            intent_str = ",".join(intents) if intents else ""
            full = (f"[intents:{intent_str}] {ctx_str}").strip()
            out.append((_s2_to_record(cp), full))
        return out
    except Exception as e:
        _s2_record_failure()
        log.warning("S2 citations failed for %s: %s", paper_id, e)
        return []


@retry_with_backoff(max_attempts=1)
def s2_get_recommended(paper_id: str, max_results: int = 25) -> list[PaperRecord]:
    if not _s2_available():
        return []
    _s2_pace()
    try:
        import requests as _requests

        resp = _requests.get(
            f"https://api.semanticscholar.org/recommendations/v1/papers/{paper_id}",
            params={"fields": _S2_FIELDS, "limit": max_results},
            timeout=TIMEOUTS.semantic_scholar,
        )
        if resp.status_code == 429:
            _s2_record_failure()
            return []
        resp.raise_for_status()
        _s2_record_success()
        return [_s2_to_record(p) for p in resp.json().get("recommendedPapers", [])]
    except Exception as e:
        log.warning("S2 recommendations failed for %s: %s", paper_id, e)
        return []


# =============================================================================
# arXiv — circuit breaker + pacing (arXiv recommends max 1 req/s)
# =============================================================================
_arxiv_consecutive_failures: int = 0
_ARXIV_CIRCUIT_THRESHOLD: int = 3
_arxiv_circuit_open: bool = False
_arxiv_last_call: float = 0.0
_ARXIV_MIN_INTERVAL: float = 1.0


def reset_arxiv_circuit() -> None:
    """Reset arXiv circuit breaker for new pipeline runs."""
    global _arxiv_consecutive_failures, _arxiv_circuit_open, _arxiv_last_call
    _arxiv_consecutive_failures = 0
    _arxiv_circuit_open = False
    _arxiv_last_call = 0.0


@retry_with_backoff(max_attempts=3)
def arxiv_search(query: str, max_results: int = 25) -> list[PaperRecord]:
    """Search arXiv. ``query`` supports arXiv field-prefix syntax.

    Respects arXiv circuit breaker — returns empty list if open.
    Enforces 1s minimum interval between calls.
    """
    global _arxiv_consecutive_failures, _arxiv_circuit_open, _arxiv_last_call
    if _arxiv_circuit_open:
        log.debug("arXiv circuit open — skipping search")
        return []
    elapsed = time.time() - _arxiv_last_call
    if elapsed < _ARXIV_MIN_INTERVAL:
        time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
    _arxiv_last_call = time.time()
    try:
        client = arxiv.Client()
        # Add arXiv category hints for earth science queries (boosts relevance
        # without filtering — papers outside these categories still appear).
        arxiv_query = query
        if any(
            w in query.lower()
            for w in (
                "earthquake",
                "seismic",
                "tectonic",
                "geophys",
                "mantle",
                "subduction",
                "crust",
                "lithosphere",
                "volcan",
                "magma",
            )
        ):
            arxiv_query = f"{query} cat:physics.geo-ph"
        search = arxiv.Search(
            query=arxiv_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[PaperRecord] = []
        for r in client.results(search):
            authors = [{"name": str(a)} for a in r.authors]
            arxiv_id = r.entry_id.split("/abs/")[-1]
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
            doi = getattr(r, "doi", None)
            out.append(
                PaperRecord(
                    doi=doi,
                    arxiv_id=arxiv_id,
                    title=r.title,
                    abstract=r.summary,
                    authors=authors,
                    year=int(r.published.year) if r.published else None,
                    publication_date=(
                        r.published.strftime("%Y-%m-%d") if r.published else ""
                    ),
                    venue="arXiv",
                    type="preprint",
                    is_open_access=True,
                    oa_pdf_url=r.pdf_url,
                    source="arxiv",
                    sources_seen=["arxiv"],
                    fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    raw_metadata={
                        "entry_id": r.entry_id,
                        "categories": list(r.categories),
                        "links": [str(link) for link in r.links],
                    },
                )
            )
        _arxiv_consecutive_failures = 0
        return out
    except Exception:
        _arxiv_consecutive_failures += 1
        if _arxiv_consecutive_failures >= _ARXIV_CIRCUIT_THRESHOLD:
            _arxiv_circuit_open = True
            log.warning(
                "arXiv circuit breaker OPENED after %d consecutive failures",
                _arxiv_consecutive_failures,
            )
        raise


# =============================================================================
# Citation export via habanero content negotiation
# =============================================================================
def export_citation(doi: str, fmt: str = "bibtex", style: str = "apa") -> str:
    """Export a single citation via doi.org content negotiation.
    fmt: 'bibtex' | 'ris' | 'text' | 'citeproc-json' | 'datacite-json'
    style: only for fmt='text' (CSL style name)."""
    return cn.content_negotiation(ids=doi, format=fmt, style=style)


def citation_count(doi: str) -> int | None:
    try:
        return counts.citation_count(doi=doi)
    except Exception as e:
        log.warning("citation_count failed for %s: %s", doi, e)
        return None


# =============================================================================
# Title similarity dedup
# =============================================================================
def _normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", (t or "").lower())).strip()


def title_similarity(a: str, b: str) -> float:
    """Jaccard token similarity (0-1). Cheap, deterministic."""
    ta = set(_normalize_title(a).split())
    tb = set(_normalize_title(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _author_dedup_key(author: dict) -> str:
    """Stable key for author dedup: last name + first initial.
    Handles full-name ('Joel Ita') vs abbreviated ('J. Ita') forms."""
    name = (author.get("name") or "").strip().lower()
    if not name:
        # Fall back to other keys (orcid, openalex_id) if name missing
        return (
            author.get("orcid")
            or author.get("openalex_id")
            or author.get("s2_author_id")
            or json.dumps(author, sort_keys=True)
        )
    # Last name = last whitespace token; first initial = first char of first token
    parts = name.replace(".", " ").split()
    if len(parts) >= 2:
        return f"{parts[-1]}|{parts[0][0]}"
    return parts[0] if parts else name


def _dedup_authors(authors: list[dict]) -> list[dict]:
    """Dedup author dicts across sources (Crossref vs OpenAlex vs S2) by
    normalized last name + first initial. Prefers fuller name spellings."""
    by_key: dict[str, dict] = {}
    for a in authors:
        key = _author_dedup_key(a)
        if key not in by_key:
            by_key[key] = a
        else:
            # Prefer the entry with the longer (more complete) name
            existing_name = by_key[key].get("name", "")
            new_name = a.get("name", "")
            if len(new_name) > len(existing_name):
                # Merge: keep new name but preserve other fields (orcid, etc.)
                merged = {**by_key[key], **a}
                merged["name"] = new_name
                by_key[key] = merged
    return list(by_key.values())


def _merge_records(a: PaperRecord, b: PaperRecord) -> PaperRecord:
    """Merge two records for same paper. Prefers non-empty values."""
    a_dict, b_dict = a.to_dict(), b.to_dict()
    merged: dict = {}
    for k in a_dict:
        av, bv = a_dict[k], b_dict.get(k)
        if k == "sources_seen":
            merged[k] = sorted(set((av or []) + (bv or [])))
        elif k == "raw_metadata":
            merged[k] = {**(av or {}), **(bv or {})}
        elif av and not bv:
            merged[k] = av
        elif bv and not av:
            merged[k] = bv
        elif k == "authors" and isinstance(av, list) and isinstance(bv, list):
            # Special handling — dedup by last+initial, prefer full names
            merged[k] = _dedup_authors(av + bv)
        elif isinstance(av, list) and isinstance(bv, list):
            seen, combined = set(), []
            for item in av + bv:
                key = (
                    json.dumps(item, sort_keys=True)
                    if isinstance(item, (dict, list))
                    else str(item)
                )
                if key not in seen:
                    seen.add(key)
                    combined.append(item)
            merged[k] = combined
        else:
            merged[k] = av if av is not None else bv
    merged["source"] = "merged"
    return PaperRecord.from_dict(merged)


def _is_supplemental_doi(doi: str) -> bool:
    """Check if DOI is supplemental material (not a real paper).

    Patterns: .s001, .s002, .supp, _si, supplement, supporting
    """
    d = doi.lower()
    return bool(
        re.search(r"\.s\d{3}\b", d)
        or ".supp" in d
        or "_si" in d
        or "supplement" in d
        or "supporting" in d
    )


def _dedup_block(p: PaperRecord) -> tuple[str, str]:
    """Blocking key for dedup: (first-author surname prefix, year).

    Berra 2023 dedupe methodology: title fuzzy-matching must be blocked on
    author+year to prevent false merges of different papers sharing generic
    titles ("Introduction to..."). Returns ("", "") when either part missing.
    """
    authors = p.authors or []
    surname = ""
    if authors and isinstance(authors[0], dict):
        name = (authors[0].get("name") or authors[0].get("family") or "").strip()
        if name:
            surname = re.sub(r"[^a-z]", "", name.lower())[:4]
    year = str(p.year) if p.year else ""
    return (surname, year)


def dedup_papers(
    papers: Iterable[PaperRecord], title_threshold: float = 0.85
) -> list[PaperRecord]:
    """Dedup by DOI (exact) + arXiv ID (exact) + BLOCKED title similarity.

    Filters supplemental DOIs (.s001, .supp, ...). Merges metadata across
    sources. Title fuzzy-match runs ONLY within (author-surname, year) blocks;
    records missing author/year fall back to a stricter threshold (0.92)
    against same-year (or all) entries — generic-title false merges are the
    failure mode this blocks (Berra 2023 dedupe methodology).
    """
    STRICT = min(title_threshold + 0.07, 0.95)
    clusters: dict[str, PaperRecord] = {}
    blocks: dict[
        tuple[str, str], list[tuple[str, str]]
    ] = {}  # block → [(norm, cluster_id)]
    fallback_index: list[tuple[str, str, str]] = []  # (norm, year, cluster_id)

    def _title_match(p: PaperRecord, norm: str) -> str | None:
        surname, year = _dedup_block(p)
        if surname and year:
            for existing_norm, cid in blocks.get((surname, year), []):
                if title_similarity(norm, existing_norm) >= title_threshold:
                    return cid
            # B3: preprint reposts across repositories/servers land in
            # different YEARS (posted 2021 on one server, 2022 on another).
            # Same first-author block + near-exact title (≥0.95) = same paper
            # regardless of the year stamp. Surname scoping keeps this safe
            # against generic-title false merges.
            for (e_surname, _e_year), entries in blocks.items():
                if e_surname != surname:
                    continue
                for existing_norm, cid in entries:
                    if title_similarity(norm, existing_norm) >= 0.95:
                        return cid
            return None
        # incomplete block: stricter threshold, same-year scope when possible
        for existing_norm, e_year, cid in fallback_index:
            if year and e_year and e_year != year:
                continue
            if title_similarity(norm, existing_norm) >= STRICT:
                return cid
        return None

    def _register(p: PaperRecord, norm: str, cid: str) -> None:
        surname, year = _dedup_block(p)
        if surname and year:
            blocks.setdefault((surname, year), []).append((norm, cid))
        fallback_index.append((norm, year, cid))

    for p in papers:
        # Filter supplemental material DOIs
        if p.doi and _is_supplemental_doi(p.doi):
            continue
        norm = _normalize_title(p.title)
        # 1. Blocked title similarity FIRST (catches DOI-less duplicates)
        if norm:
            matched_id = _title_match(p, norm)
            if matched_id:
                clusters[matched_id] = _merge_records(clusters[matched_id], p)
                continue
        # 2. Exact DOI match
        if p.doi:
            key = "doi:" + p.doi.lower()
            if key in clusters:
                clusters[key] = _merge_records(clusters[key], p)
                continue
            clusters[key] = p
            if norm:
                _register(p, norm, key)
            continue
        # 3. Exact arXiv ID match
        if p.arxiv_id:
            key = "arxiv:" + p.arxiv_id
            if key in clusters:
                clusters[key] = _merge_records(clusters[key], p)
                continue
            clusters[key] = p
            if norm:
                _register(p, norm, key)
            continue
        # 4. New paper — register by primary_id. Identical-title distinct papers
        # share the title-hash key; collision without a blocked title match =
        # different papers → disambiguate instead of silently dropping one.
        key = p.primary_id
        if key in clusters:
            key = f"{key}#{sum(1 for k in clusters if k.startswith(key))}"
        clusters[key] = p
        if norm:
            _register(p, norm, key)
    return list(clusters.values())


# =============================================================================
# Compact summary renderer (LLM-facing)
# =============================================================================
def render_compact_summary(
    papers: Iterable[PaperRecord],
    max_papers: int = 50,
    include_abstract: bool = True,
    abstract_chars: int = 300,
) -> str:
    """Render compact markdown summary (~150 tokens per paper)."""
    papers = list(papers)[:max_papers]
    lines = [f"# Corpus summary ({len(papers)} papers)\n"]
    for i, p in enumerate(papers, 1):
        auth = ", ".join(a["name"] for a in p.authors[:3])
        if len(p.authors) > 3:
            auth += f" ... +{len(p.authors) - 3}"
        ident = p.doi or p.arxiv_id or p.openalex_id or "?"
        lines.append(f"## {i}. {p.title}")
        lines.append(
            f"- **ID**: `{ident}` | **Year**: {p.year or '?'} | "
            f"**Venue**: {p.venue or '?'} | **Source**: {p.source}"
        )
        if auth:
            lines.append(f"- **Authors**: {auth}")
        if p.citation_count is not None:
            infl = (
                f" | Influential: {p.influential_citation_count}"
                if p.influential_citation_count
                else ""
            )
            lines.append(f"- **Citations**: {p.citation_count}{infl}")
        if p.is_retracted:
            lines.append("- **WARNING: RETRACTED**")
        if p.is_open_access and p.oa_pdf_url:
            lines.append(f"- **OA PDF**: {p.oa_pdf_url}")
        if p.concepts:
            lines.append(f"- **Concepts**: {', '.join(p.concepts[:5])}")
        if p.fields_of_study:
            lines.append(f"- **Fields**: {', '.join(p.fields_of_study)}")
        if p.local_path:
            lines.append(f"- **Local PDF**: {p.local_path}")
        if include_abstract and p.abstract:
            ab = p.abstract[:abstract_chars]
            if len(p.abstract) > abstract_chars:
                ab += "..."
            lines.append(f"- **Abstract**: {ab}")
        lines.append("")
    return "\n".join(lines)


def load_corpus(path: str | Path) -> list[PaperRecord]:
    """Load corpus.json file → list[PaperRecord]."""
    data = json.loads(Path(path).read_text())
    return [PaperRecord.from_dict(d) for d in data.get("papers", [])]


def save_corpus(
    papers: Iterable[PaperRecord], path: str | Path, meta: dict | None = None
) -> None:
    """Save corpus to JSON with optional metadata."""
    payload = {
        "meta": meta or {},
        "papers": [_sanitize_json(p.to_dict()) for p in papers],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )


if __name__ == "__main__":
    print("scientific-research _sources module loaded OK")
    print(f"Cache dir: {CACHE_DIR}")
    print("Sources: crossref, openalex, s2, arxiv")


# =============================================================================
# Europe PMC (biomedical + life sciences; PMC OA subset; no API key required)
# Verified live 2026-08-14: REST search endpoint, resultType=core fields
#   title, authorString, abstractText, doi, pmid, pubYear, isOpenAccess (Y/N),
#   journalInfo.journal.title, pmcid (PMC full-text id)
# =============================================================================


def _epmc_to_record(it: dict) -> PaperRecord:
    authors = []
    for name in (it.get("authorString") or "").split(","):
        name = name.strip()
        if name:
            authors.append({"name": name})
    journal_info = it.get("journalInfo") or {}
    journal = (journal_info.get("journal") or {}).get("title") or ""
    pub_year = str(it.get("pubYear") or journal_info.get("yearOfPublication") or "")
    keywords = [str(k) for k in (it.get("keywordList") or []) if k]
    return PaperRecord(
        doi=it.get("doi"),
        pmid=str(it["pmid"]) if it.get("pmid") else None,
        title=it.get("title") or "",
        abstract=it.get("abstractText") or "",
        authors=authors,
        year=int(pub_year) if pub_year.isdigit() else None,
        venue=journal,
        type=it.get("pubType") or "",
        is_open_access=(it.get("isOpenAccess") == "Y"),
        language=it.get("language") or "",
        mesh=keywords,
        source="epmc",
        raw_metadata={
            "epmc": {k: it.get(k) for k in ("id", "pmcid", "source", "citedByCount")}
        },
    )


@retry_with_backoff(max_attempts=3)
def epmc_search(query: str, max_results: int = 25) -> list[PaperRecord]:
    """Search Europe PMC (REST, no key). PubMed + PMC + preprints + agricolA."""
    from urllib.parse import quote_plus

    import httpx

    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query={quote_plus(query)}"
        f"&format=json&resultType=core&pageSize={min(max_results, 100)}&cursorMark=*"
    )
    try:
        r = httpx.get(url, timeout=TIMEOUTS.openalex)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Europe PMC search failed: %s", e)
        return []
    hits = (data.get("resultList") or {}).get("result") or []
    records = []
    for it in hits:
        if not (it.get("doi") or it.get("pmid")):
            continue  # skip records without resolvable identifiers (§5 H20)
        records.append(_epmc_to_record(it))
        if len(records) >= max_results:
            break
    for p in records:
        cache_put(p)
    try:
        from _search_cache import search_cache_put

        search_cache_put(query, "epmc", max_results, [p.primary_id for p in records])
    except (ImportError, NameError):
        pass
    return records


def epmc_get_by_doi(doi: str) -> PaperRecord | None:
    """Fetch a single record by DOI via Europe PMC."""
    import httpx

    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=DOI:{doi}&format=json&resultType=core&pageSize=1"
    )
    try:
        r = httpx.get(url, timeout=TIMEOUTS.openalex)
        r.raise_for_status()
        hits = (r.json().get("resultList") or {}).get("result") or []
        return _epmc_to_record(hits[0]) if hits else None
    except Exception as e:
        log.debug("Europe PMC DOI lookup failed for %s: %s", doi, e)
        return None


# =============================================================================
# EarthArXiv (OSF preprints — earth science specific)
# =============================================================================


def eartharxiv_search(query: str, max_results: int = 15) -> list[PaperRecord]:
    """Search EarthArXiv preprints via Crossref (DOI prefix 10.31223).

    EarthArXiv was originally hosted on OSF but moved to CDL hosting.
    The OSF API v2 ``/preprints/`` endpoint does not support text search
    — the ``title_or_abstract`` filter field is invalid and returns 400.
    Crossref indexes EarthArXiv preprints under DOI prefix ``10.31223``
    and supports full-text ``query`` search with the ``prefix`` filter.

    Rate-limit management: delegates to ``crossref_search``, which respects
    the shared Crossref circuit breaker (skips when open) and uses
    ``@retry_with_backoff`` for 429s (3 attempts, exponential backoff,
    3 s minimum for rate-limit errors). No suppression — every failure
    is logged at WARNING level.
    """
    if not _crossref_cb().available:
        log.debug("Crossref circuit open — skipping EarthArXiv search")
        return []
    try:
        papers = crossref_search(
            query, max_results=max_results, filter_dict={"prefix": "10.31223"}
        )
    except RuntimeError as e:
        log.warning("EarthArXiv search failed via Crossref: %s", e)
        return []
    for p in papers:
        p.source = "eartharxiv"
        p.sources_seen = ["eartharxiv"]
        if not p.venue:
            p.venue = "EarthArXiv"
    log.info("EarthArXiv: %d results for '%s'", len(papers), query[:40])
    return papers


# =============================================================================
# USGS Publications (DOI prefix 10.3133 — public domain)
# =============================================================================


def usgs_search(query: str, max_results: int = 15) -> list[PaperRecord]:
    """Search USGS publications via Crossref (DOI prefix 10.3133).

    USGS publications are public domain and freely available. Crossref
    indexes them under DOI prefix ``10.3133`` with full-text search.

    Rate-limit management: delegates to ``crossref_search``, which respects
    the shared Crossref circuit breaker and ``@retry_with_backoff``.
    """
    if not _crossref_cb().available:
        log.debug("Crossref circuit open — skipping USGS search")
        return []
    try:
        papers = crossref_search(
            query, max_results=max_results, filter_dict={"prefix": "10.3133"}
        )
    except RuntimeError as e:
        log.warning("USGS search failed via Crossref: %s", e)
        return []
    for p in papers:
        p.source = "usgs"
        p.sources_seen = ["usgs"]
        if not p.venue:
            p.venue = "USGS"
    log.info("USGS: %d results for '%s'", len(papers), query[:40])
    return papers

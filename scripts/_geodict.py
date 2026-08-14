"""Geological dictionary — Wikidata SPARQL powered term database.

Downloads minerals (IMA-approved species), mineral groups, and rock types
from Wikidata. Replaces manual hardcoded term lists with 8000+ terms.

Cache: ~/.cache/geokit/geo_dictionary.json (refreshed every 30 days).
Fallback: bundled curated terms if Wikidata unreachable.

Sources:
  - Minerals: Wikidata Q12089225 (mineral species, IMA-approved)
  - Mineral groups: Wikidata Q7946 subclasses (silicate, halide, etc.)
  - Rocks: Wikidata Q8063 subclasses (1852 rock types)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger("scientific_research.geodict")

_CACHE_PATH = Path.home() / ".cache" / "geokit" / "geo_dictionary.json"
_CACHE_TTL_DAYS = 30
_BATCH_SIZE = 500
_SPARQL_TIMEOUT = 30

# Bundled fallback terms (used when Wikidata is unreachable)
_FALLBACK_TERMS: list[str] = [
    "garnet", "biotite", "quartz", "plagioclase", "clinopyroxene",
    "orthopyroxene", "olivine", "hornblende", "muscovite", "chlorite",
    "basalt", "granite", "granodiorite", "diorite", "gabbro", "peridotite",
    "rhyolite", "andesite", "dunite", "harzburgite", "lherzolite",
    "eclogite", "amphibolite", "schist", "gneiss", "marble", "quartzite",
    "subduction", "metamorphism", "thermobarometry", "geochemistry",
]


def _sparql(query: str) -> list[str]:
    """Execute SPARQL query, return list of English labels."""
    url = "https://query.wikidata.org/sparql"
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Accept": "application/sparql-results+json",
            "User-Agent": "GeoKit/research (geological dictionary builder)",
        },
    )
    with urllib.request.urlopen(req, timeout=_SPARQL_TIMEOUT) as resp:
        result = json.loads(resp.read())
    labels: list[str] = []
    for binding in result.get("results", {}).get("bindings", []):
        label = binding.get("label", {}).get("value", "")
        if label and len(label) > 2 and len(label) < 40:
            labels.append(label)
    return labels


def _sparql_paginated(base_body: str, batch_size: int = _BATCH_SIZE) -> list[str]:
    """Paginated SPARQL query — fetches all results in batches.

    ``base_body`` is the WHERE clause body (without SELECT or WHERE).
    """
    all_labels: list[str] = []
    offset = 0
    while True:
        query = (
            f"SELECT ?label WHERE {{ {base_body} }} "
            f"LIMIT {batch_size} OFFSET {offset}"
        )
        try:
            batch = _sparql(query)
        except Exception as e:
            log.warning("SPARQL batch failed at offset %d: %s", offset, e)
            break
        if not batch:
            break
        all_labels.extend(batch)
        log.debug("SPARQL batch: offset=%d, got=%d, total=%d", offset, len(batch), len(all_labels))
        if len(batch) < batch_size:
            break
        offset += batch_size
        time.sleep(1)  # be polite to Wikidata
    return all_labels


def _download_geodictionary() -> dict[str, list[str]]:
    """Download geological terms from Wikidata.

    Returns dict with keys: 'minerals', 'mineral_groups', 'rocks', 'all'.
    """
    result: dict[str, list[str]] = {
        "minerals": [],
        "mineral_groups": [],
        "rocks": [],
        "all": [],
    }

    # 1. Mineral species (IMA-approved, Q12089225)
    log.info("Downloading mineral species from Wikidata...")
    try:
        minerals = _sparql_paginated(
            '?item wdt:P31 wd:Q12089225. '
            '?item rdfs:label ?label. '
            'FILTER(LANG(?label) = "en") '
        )
        # Clean: remove entries with parentheses (discredited minerals)
        minerals = [m for m in minerals if "(" not in m and "-" not in m[:2]]
        result["minerals"] = sorted(set(minerals))
        log.info("Minerals: %d species", len(result["minerals"]))
    except Exception as e:
        log.warning("Mineral download failed: %s — using fallback", e)

    # 2. Mineral groups (subclasses of Q7946)
    log.info("Downloading mineral groups from Wikidata...")
    try:
        groups = _sparql(
            'SELECT ?label WHERE { '
            '?item wdt:P279 wd:Q7946. '
            '?item rdfs:label ?label. '
            'FILTER(LANG(?label) = "en") '
            '} LIMIT 200'
        )
        result["mineral_groups"] = sorted(set(groups))
        log.info("Mineral groups: %d", len(result["mineral_groups"]))
    except Exception as e:
        log.warning("Mineral groups download failed: %s", e)

    # 3. Rock types (subclasses of Q8063)
    log.info("Downloading rock types from Wikidata...")
    try:
        rocks = _sparql_paginated(
            '?item wdt:P279+ wd:Q8063. '
            '?item rdfs:label ?label. '
            'FILTER(LANG(?label) = "en") '
        )
        result["rocks"] = sorted(set(rocks))
        log.info("Rocks: %d types", len(result["rocks"]))
    except Exception as e:
        log.warning("Rock download failed: %s", e)

    # Merge all terms
    all_terms = set(result["minerals"] + result["mineral_groups"] + result["rocks"])
    # Add fallback terms if list is too small
    if len(all_terms) < 100:
        log.warning("Downloaded only %d terms — adding fallback list", len(all_terms))
        all_terms.update(_FALLBACK_TERMS)
    result["all"] = sorted(all_terms)
    log.info("Total geological terms: %d", len(result["all"]))

    return result


def _is_cache_stale() -> bool:
    """Check if cache is missing or older than TTL."""
    if not _CACHE_PATH.exists():
        return True
    age_days = (time.time() - _CACHE_PATH.stat().st_mtime) / 86400
    return age_days > _CACHE_TTL_DAYS


_geo_dict: dict[str, list[str]] | None = None


def get_geo_dictionary() -> dict[str, list[str]]:
    """Get the geological dictionary (cached, lazy-loaded).

    Returns dict with keys: 'minerals', 'mineral_groups', 'rocks', 'all'.
    Downloads from Wikidata on first use or when cache expires.
    """
    global _geo_dict
    if _geo_dict is not None:
        return _geo_dict

    # Try cache
    if not _is_cache_stale():
        try:
            _geo_dict = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            log.debug("Geo dictionary loaded from cache: %d terms", len(_geo_dict.get("all", [])))
            return _geo_dict
        except Exception as e:
            log.warning("Cache read failed: %s — re-downloading", e)

    # Download fresh
    _geo_dict = _download_geodictionary()

    # Cache to disk
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_geo_dict, ensure_ascii=False), encoding="utf-8")
        log.info("Geo dictionary cached to %s", _CACHE_PATH)
    except Exception as e:
        log.warning("Cache write failed: %s", e)

    return _geo_dict


def get_all_terms() -> list[str]:
    """Get flat list of all geological terms (for fuzzy matching + BGE)."""
    return get_geo_dictionary().get("all", [])


def get_minerals() -> list[str]:
    """Get list of mineral species."""
    return get_geo_dictionary().get("minerals", [])


def get_rocks() -> list[str]:
    """Get list of rock types."""
    return get_geo_dictionary().get("rocks", [])


# =============================================================================
# Journal ISSN discovery via Crossref
# =============================================================================
_JOURNAL_CACHE_PATH = Path.home() / ".cache" / "geokit" / "geo_journals.json"
_GEO_JOURNAL_QUERIES = [
    "geology", "geochemistry", "petrology", "mineralogy",
    "earth science", "tectonics", "sedimentology", "geophysics",
    "volcanology", "economic geology", "structural geology",
    "metamorphic", "mineral deposit", "hydrogeology",
    "paleontology", "stratigraphy", "seismology",
]
_geo_journals: list[str] | None = None


def _download_geo_journals() -> list[str]:
    """Download geoscience journal ISSNs from Crossref."""
    issns: set[str] = set()
    for query in _GEO_JOURNAL_QUERIES:
        try:
            url = (
                f"https://api.crossref.org/journals"
                f"?query={urllib.parse.quote(query)}&rows=50"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "GeoKit/research (journal discovery)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                import json as _json

                data = _json.loads(resp.read())
            for journal in data.get("message", {}).get("items", []):
                for issn in journal.get("ISSN", []):
                    if issn and len(issn) >= 8:
                        issns.add(issn)
            log.debug("Journal query '%s': %d unique ISSNs so far", query, len(issns))
            time.sleep(0.5)  # be polite to Crossref
        except Exception as e:
            log.warning("Journal query '%s' failed: %s", query, e)

    result = sorted(issns)
    log.info("Downloaded %d geo journal ISSNs from Crossref", len(result))
    return result


def get_geo_journals() -> list[str]:
    """Get list of geoscience journal ISSNs (cached, Crossref-powered).

    Returns list of ISSN strings for journals matching geological queries.
    Cached for 30 days, falls back to empty list if Crossref unreachable.
    """
    global _geo_journals
    if _geo_journals is not None:
        return _geo_journals

    # Check cache
    if _JOURNAL_CACHE_PATH.exists():
        age_days = (time.time() - _JOURNAL_CACHE_PATH.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            try:
                import json as _json
                _geo_journals = _json.loads(_JOURNAL_CACHE_PATH.read_text(encoding="utf-8"))
                log.debug("Geo journals from cache: %d ISSNs", len(_geo_journals))
                return _geo_journals
            except Exception:
                pass

    # Download fresh
    _geo_journals = _download_geo_journals()

    # Cache
    try:
        _JOURNAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        _JOURNAL_CACHE_PATH.write_text(
            _json.dumps(_geo_journals), encoding="utf-8"
        )
    except Exception as e:
        log.warning("Journal cache write failed: %s", e)

    return _geo_journals

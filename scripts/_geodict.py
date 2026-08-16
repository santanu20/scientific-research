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

_BATCH_SIZE = 500
_SPARQL_TIMEOUT = 30

# Bundled fallback terms (used when Wikidata is unreachable)
_FALLBACK_TERMS: list[str] = [
    "garnet",
    "biotite",
    "quartz",
    "plagioclase",
    "clinopyroxene",
    "orthopyroxene",
    "olivine",
    "hornblende",
    "muscovite",
    "chlorite",
    "basalt",
    "granite",
    "granodiorite",
    "diorite",
    "gabbro",
    "peridotite",
    "rhyolite",
    "andesite",
    "dunite",
    "harzburgite",
    "lherzolite",
    "eclogite",
    "amphibolite",
    "schist",
    "gneiss",
    "marble",
    "quartzite",
    "subduction",
    "metamorphism",
    "thermobarometry",
    "geochemistry",
]


def _sparql(query: str) -> list[str]:
    """Execute SPARQL query, return list of English labels."""
    url = "https://query.wikidata.org/sparql"
    data = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    req = urllib.request.Request(
        url,
        data=data,
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
        query = f"SELECT ?label WHERE {{ {base_body} }} LIMIT {batch_size} OFFSET {offset}"
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


# =============================================================================
# Journal ISSN discovery via Crossref
# =============================================================================
_JOURNAL_CACHE_PATH = Path.home() / ".cache" / "geokit" / "geo_journals.json"
_CACHE_TTL_DAYS = 30
_GEO_JOURNAL_QUERIES = [
    "geology",
    "geochemistry",
    "petrology",
    "mineralogy",
    "earth science",
    "tectonics",
    "sedimentology",
    "geophysics",
    "volcanology",
    "economic geology",
    "structural geology",
    "metamorphic",
    "mineral deposit",
    "hydrogeology",
    "paleontology",
    "stratigraphy",
    "seismology",
]
_geo_journals: list[str] | None = None


def _download_geo_journals() -> list[str]:
    """Download geoscience journal ISSNs from Crossref."""
    issns: set[str] = set()
    for query in _GEO_JOURNAL_QUERIES:
        try:
            url = f"https://api.crossref.org/journals?query={urllib.parse.quote(query)}&rows=50"
            req = urllib.request.Request(url, headers={"User-Agent": "GeoKit/research (journal discovery)"})
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

                data = _json.loads(_JOURNAL_CACHE_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    log.debug("Geo journals from cache: %d ISSNs", len(data))
                    return data
                log.warning("Geo journals cache has wrong shape (%s) — re-downloading", type(data).__name__)
            except Exception as e:
                log.warning("Geo journals cache read failed: %s — re-downloading", e)

    # Download fresh
    _geo_journals = _download_geo_journals()

    # Cache
    try:
        _JOURNAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        _JOURNAL_CACHE_PATH.write_text(_json.dumps(_geo_journals), encoding="utf-8")
    except Exception as e:
        log.warning("Journal cache write failed: %s", e)

    return _geo_journals

#!/usr/bin/env python3
"""Knowledge base for scientific-research skill.

A permanent, growing store of papers + lightweight manifest indexes.
Lives in ~/.local/share/scientific-research/ (XDG permanent data).

Architecture:
  Paper-wise files: <hash>/paper.json  — full record (title, abstract, authors, refs, etc.)
  Single manifest: search_manifest.json — query → paper_ids + timestamp (index, not data)
  Single manifest: verify_manifest.json — doi → verification result + timestamp (index)

Paper-wise files are the single source of truth. Manifests are just indexes.
Use --force-refresh to skip manifests and fetch fresh from APIs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("scientific_research.knowledge_base")

# =============================================================================
# Paths
# =============================================================================
KB_BASE = Path(
    os.environ.get(
        "SCIENTIFIC_RESEARCH_KB",
        str(Path.home() / ".local" / "share" / "scientific-research"),
    )
)
KB_BASE.mkdir(parents=True, exist_ok=True)

SEARCH_MANIFEST = KB_BASE / "search_manifest.json"
VERIFY_MANIFEST = KB_BASE / "verify_manifest.json"

SEARCH_TTL = 30 * 86400  # 30 days
VERIFY_TTL = 90 * 86400  # 90 days

_LEGACY_CACHE = Path.home() / ".cache" / "scientific-research"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_manifest(path: Path) -> dict:
    """Load a manifest file (single JSON dict)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(path: Path, data: dict) -> None:
    """Save a manifest file."""
    try:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    except OSError as e:
        log.warning("Manifest write failed: %s", e)


def _migrate_from_legacy():
    """One-time migration from old per-file cache to manifest + paper files."""
    if not _LEGACY_CACHE.exists():
        return
    # Migrate paper files (already done by _sources.py CACHE_DIR change)
    # Migrate search files → single manifest
    legacy_searches = _LEGACY_CACHE / "searches"
    if legacy_searches.exists() and not SEARCH_MANIFEST.exists():
        manifest = {}
        for f in legacy_searches.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                key = _hash(
                    f"{data.get('query', '')}|{data.get('sources', '')}|{data.get('max_per_source', 0)}"
                )
                manifest[key] = {
                    "query": data.get("query", ""),
                    "sources": data.get("sources", ""),
                    "paper_ids": data.get("paper_ids", []),
                    "timestamp": data.get("timestamp", 0),
                }
            except (json.JSONDecodeError, OSError):
                pass
        if manifest:
            _save_manifest(SEARCH_MANIFEST, manifest)
            log.info("Migrated %d searches to manifest", len(manifest))

    # Also check new searches dir (from partial migration)
    new_searches = KB_BASE / "searches"
    if new_searches.exists() and not SEARCH_MANIFEST.exists():
        manifest = {}
        for f in new_searches.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                key = _hash(
                    f"{data.get('query', '')}|{data.get('sources', '')}|{data.get('max_per_source', 0)}"
                )
                manifest[key] = {
                    "query": data.get("query", ""),
                    "sources": data.get("sources", ""),
                    "paper_ids": data.get("paper_ids", []),
                    "timestamp": data.get("timestamp", 0),
                }
            except (json.JSONDecodeError, OSError):
                pass
        if manifest:
            _save_manifest(SEARCH_MANIFEST, manifest)
            log.info("Migrated %d searches from new dir to manifest", len(manifest))

    # Migrate verify files → single manifest
    legacy_verify = _LEGACY_CACHE / "verifications"
    new_verify = KB_BASE / "verifications"
    vmanifest = {}
    for vdir in [legacy_verify, new_verify]:
        if vdir.exists():
            for f in vdir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                    doi = data.get("doi", "")
                    if doi:
                        vmanifest[doi] = {
                            "result": data.get("result", {}),
                            "timestamp": data.get("timestamp", 0),
                        }
                except (json.JSONDecodeError, OSError):
                    pass
    if vmanifest and not VERIFY_MANIFEST.exists():
        _save_manifest(VERIFY_MANIFEST, vmanifest)
        log.info("Migrated %d verifications to manifest", len(vmanifest))


# Auto-migrate on import
_migrate_from_legacy()


# =============================================================================
# Search index (single manifest)
# =============================================================================
def search_cache_get(
    query: str, sources: str, max_per_source: int, force_refresh: bool = False,
    filters_hash: str = "",
) -> tuple[list[str] | None, int]:
    """Check search manifest for prior results.

    Returns (paper_ids, age_in_days).
    - Hit (fresh): ([ids], N days old)
    - Miss/stale/force: (None, -1)
    """
    if force_refresh:
        return (None, -1)
    key = _hash(f"{query}|{sources}|{max_per_source}|{filters_hash}")
    manifest = _load_manifest(SEARCH_MANIFEST)
    entry = manifest.get(key)
    if not entry:
        return (None, -1)
    age = time.time() - entry.get("timestamp", 0)
    age_days = int(age // 86400)
    if age < SEARCH_TTL:
        ids = entry.get("paper_ids", [])
        log.info(
            "Knowledge base HIT: '%s' (%d papers, %d days old)",
            query[:40],
            len(ids),
            age_days,
        )
        return (ids, age_days)
    else:
        log.info("Knowledge base STALE: '%s' (%d days old) — re-fetching", query[:40], age_days)
    return (None, -1)


def search_cache_put(query: str, sources: str, max_per_source: int, paper_ids: list[str]) -> None:
    """Add/update search entry in manifest."""
    key = _hash(f"{query}|{sources}|{max_per_source}|{filters_hash}")
    manifest = _load_manifest(SEARCH_MANIFEST)
    manifest[key] = {
        "query": query,
        "sources": sources,
        "max_per_source": max_per_source,
        "paper_ids": paper_ids,
        "timestamp": time.time(),
    }
    _save_manifest(SEARCH_MANIFEST, manifest)


# =============================================================================
# Verification index (single manifest)
# =============================================================================
def verify_cache_get(doi: str, force_refresh: bool = False) -> dict | None:
    """Check verification manifest for prior DOI verification."""
    if force_refresh:
        return None
    manifest = _load_manifest(VERIFY_MANIFEST)
    entry = manifest.get(doi)
    if not entry:
        return None
    age = time.time() - entry.get("timestamp", 0)
    if age < VERIFY_TTL:
        return entry.get("result")
    return None


def verify_cache_put(doi: str, result: dict) -> None:
    """Add/update verification entry in manifest."""
    manifest = _load_manifest(VERIFY_MANIFEST)
    manifest[doi] = {
        "result": result,
        "timestamp": time.time(),
    }
    _save_manifest(VERIFY_MANIFEST, manifest)


# =============================================================================
# Stats
# =============================================================================
def kb_stats() -> dict:
    """Return knowledge base statistics."""
    # Paper count
    paper_count = 0
    paper_size = 0
    for f in KB_BASE.glob("*/paper.json"):
        paper_count += 1
        paper_size += f.stat().st_size

    # Manifests
    search_manifest = _load_manifest(SEARCH_MANIFEST)
    verify_manifest = _load_manifest(VERIFY_MANIFEST)
    search_size = SEARCH_MANIFEST.stat().st_size if SEARCH_MANIFEST.exists() else 0
    verify_size = VERIFY_MANIFEST.stat().st_size if VERIFY_MANIFEST.exists() else 0

    # LLM cache (temporary)
    llm_dir = Path.home() / ".cache" / "scientific_research" / "llm_extract"
    llm_count = len(list(llm_dir.glob("*.json"))) if llm_dir.exists() else 0
    llm_size = sum(f.stat().st_size for f in llm_dir.glob("*.json")) if llm_dir.exists() else 0

    return {
        "knowledge_base": str(KB_BASE),
        "paper_store": {
            "papers": paper_count,
            "size_mb": round(paper_size / 1e6, 1),
            "permanent": True,
            "description": "Paper-wise files — full records (title, abstract, authors, refs)",
        },
        "search_manifest": {
            "path": str(SEARCH_MANIFEST),
            "entries": len(search_manifest),
            "size_kb": round(search_size / 1024, 1),
            "ttl_days": SEARCH_TTL // 86400,
        },
        "verify_manifest": {
            "path": str(VERIFY_MANIFEST),
            "entries": len(verify_manifest),
            "size_kb": round(verify_size / 1024, 1),
            "ttl_days": VERIFY_TTL // 86400,
        },
        "llm_cache": {
            "path": str(llm_dir),
            "entries": llm_count,
            "size_mb": round(llm_size / 1e6, 1),
            "temporary": True,
        },
        "total_mb": round((paper_size + search_size + verify_size + llm_size) / 1e6, 1),
    }


if __name__ == "__main__":
    print(json.dumps(kb_stats(), indent=2))

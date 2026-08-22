"""Centralized timeout constants for external web-API calls.

Holds the timeouts that are actually wired through TIMEOUTS.* call sites
(_sources.py, verify.py). Each can be overridden via environment variable
for debugging/tuning. New external-API timeouts should be added here rather
than inlined at call sites.

Usage:
    from _timeouts import TIMEOUTS
    httpx.get(url, timeout=TIMEOUTS.crossref_search)
"""

from __future__ import annotations

import os


class _Timeouts:
    """Centralized timeout configuration.

    Values read from environment variables (for runtime tuning) or
    fall back to sensible defaults.
    """

    @property
    def crossref_search(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_CROSSREF", "30"))

    @property
    def crossref_doi(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_DOI_LOOKUP", "15"))

    @property
    def openalex(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_OPENALEX", "10"))

    @property
    def semantic_scholar(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_S2", "10"))

    @property
    def unpaywall(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_UNPAYWALL", "15"))

    @property
    def doaj(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_DOAJ", "15"))


    @property
    def wikipedia(self) -> int:
        """Wikipedia REST summary API (entity kinds/aliases, _context.py)."""
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_WIKIPEDIA", "10"))

TIMEOUTS = _Timeouts()

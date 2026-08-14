"""Centralized timeout constants for the research pipeline.

All timeouts in one place — no scattered magic numbers across files.
Each can be overridden via environment variable for debugging/tuning.

Usage:
    from _timeouts import TIMEOUTS
    urllib.request.urlopen(req, timeout=TIMEOUTS.ollama_health)
"""

from __future__ import annotations

import os


class _Timeouts:
    """Centralized timeout configuration.

    Values read from environment variables (for runtime tuning) or
    fall back to sensible defaults. Every timeout in the pipeline
    should reference these constants, never inline magic numbers.
    """

    @property
    def ollama_health(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_OLLAMA_HEALTH", "3"))

    @property
    def ollama_extract(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_OLLAMA_EXTRACT", "60"))

    @property
    def ollama_synthesis(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_OLLAMA_SYNTHESIS", "180"))

    @property
    def ollama_smoothing(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_OLLAMA_SMOOTH", "60"))

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
    def wikidata(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_WIKIDATA", "30"))

    @property
    def wikipedia(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_WIKIPEDIA", "5"))

    @property
    def unpaywall(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_UNPAYWALL", "15"))

    @property
    def pdf_download(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_PDF_DOWNLOAD", "30"))

    @property
    def llm_intent(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_LLM_INTENT", "30"))

    @property
    def bge_embedding(self) -> int:
        return int(os.environ.get("SCIENTIFIC_RESEARCH_TIMEOUT_BGE", "60"))


TIMEOUTS = _Timeouts()

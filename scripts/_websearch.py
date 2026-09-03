"""Web-search integration for the research pipeline (standalone skill).

Text search delegates to the OMP web-search skill CLI when present
(optional, degrades loudly). Web-pro synthesis and the crawler tiers are
NOT part of the standalone skill — callers must treat None as "skip".
"""

from __future__ import annotations

import logging

log = logging.getLogger("scientific_research.websearch")

SearchError = RuntimeError  # narrowest compatible sentinel


def run_pro_synthesis(query, **kwargs):  # noqa: ANN001, ANN003
    log.warning("web-pro synthesis is not available in the standalone skill")
    return None


def search_text(query, **kwargs):  # noqa: ANN001, ANN003
    from discover import _web_text_engine

    return _web_text_engine(query, **kwargs)


def agentic(*args, **kwargs):  # noqa: ANN002, ANN003
    return 1, {"error": "standalone skill: agentic tier unavailable"}


def crawl(*args, **kwargs):  # noqa: ANN002, ANN003
    return 1, {"error": "standalone skill: crawl tier unavailable"}


def extract(*args, **kwargs):  # noqa: ANN002, ANN003
    return None


def search_news(*args, **kwargs):  # noqa: ANN002, ANN003
    return 1, {"error": "standalone skill: news tier unavailable"}


def run(*args, **kwargs):  # noqa: ANN002, ANN003
    return 1, {"error": "standalone skill: engine unavailable"}

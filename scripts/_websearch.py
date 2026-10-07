"""Web-search integration for the research pipeline (standalone skill).

Text search delegates to the web-search skill CLI when present (optional,
degrades loudly). Web-pro synthesis shells out to the sibling web-search
skill's `pro` command (Perplexity-style agentic research: decompose ->
multi-round search+extract -> cited synthesis) and adapts the payload to
the pipeline's supplement contract {synthesis, confidence, facets,
coverage, sources}. The crawler tiers are NOT part of the standalone
skill — callers must treat None as "skip".
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("scientific_research.websearch")

SearchError = RuntimeError  # narrowest compatible sentinel

# Bounded: this is an advisory supplement, not the main path. pro on CPU
# can take minutes; anything longer means the run is wedged — skip it.
_PRO_TIMEOUT_S = 480.0
_PRO_MAX_ROUNDS = 2


def _websearch_skill_root() -> Path | None:
    """Locate the sibling web-search skill (env override > sibling guess)."""
    env = os.environ.get("WEB_SEARCH_SKILL_DIR")
    if env:
        p = Path(env).expanduser()
        return p if (p / "web_search.py").is_file() else None
    sibling = Path(__file__).resolve().parent.parent.parent / "web-search"
    return sibling if (sibling / "web_search.py").is_file() else None


def _cli_python(root: Path) -> str:
    venv_py = root / ".venv" / "bin" / "python"
    return str(venv_py) if venv_py.is_file() else sys.executable


def run_pro_synthesis(query: str, **kwargs):  # noqa: ANN001, ANN003
    """Perplexity-style web synthesis via the sibling web-search skill CLI.

    Returns the pipeline supplement contract dict, or None when the skill
    is absent/unusable (caller logs + skips — never raises). kwargs is
    accepted for signature compatibility with the GUI-host original.
    """
    root = _websearch_skill_root()
    if root is None:
        log.warning(
            "web-pro synthesis: sibling web-search skill not found "
            "(set WEB_SEARCH_SKILL_DIR to override) — skipping supplement"
        )
        return None
    cmd = [
        _cli_python(root),
        "web_search.py",
        "pro",
        query,
        "--max-rounds",
        str(_PRO_MAX_ROUNDS),
        # Deterministic heuristic supplement: no LLM dependency. The
        # pipeline's own phase-5 synthesis already used Ollama; a second
        # 9B synthesis here would contend for it inside the subprocess
        # timeout (found live 2026-10-07).
        "--no-llm",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=_PRO_TIMEOUT_S,
            cwd=root,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "web-pro synthesis timed out after %.0fs — skipping supplement",
            _PRO_TIMEOUT_S,
        )
        return None
    except OSError as e:
        log.warning("web-pro synthesis failed to launch: %s — skipping", e)
        return None
    if proc.returncode != 0:
        # Last 5 stderr lines: one line hid the real error in the first
        # live run (found 2026-10-07) — the failing call site matters.
        tail = (proc.stderr or "").strip().splitlines()[-5:] or [f"exit {proc.returncode}"]
        log.warning(
            "web-pro synthesis exited %d: %s — skipping",
            proc.returncode,
            " | ".join(line[:200] for line in tail),
        )
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        log.warning("web-pro synthesis output not JSON: %s — skipping", e)
        return None
    synthesis = str(payload.get("synthesis") or "")
    if not synthesis:
        log.warning("web-pro synthesis returned empty synthesis — skipping")
        return None
    # Facet coverage mirrors pro's own semantics: a term counts as covered
    # when >= 2 distinct sources mention it (single-page hits are weak).
    cov_detail = payload.get("coverage_detail") or {}
    sources = payload.get("sources") or []
    # Inline [n] markers in the synthesis are 0-based indices into the
    # sources list — append the resolution list so the brief section is
    # self-contained (URLs, not DOIs — advisory supplement contract).
    if sources:
        src_lines = "\n".join(
            f"[{i}] {s.get('title') or s.get('host') or ''} — {s.get('url')}" for i, s in enumerate(sources)
        )
        synthesis += f"\n\nWeb sources:\n{src_lines}"
    return {
        "synthesis": synthesis,
        "confidence": float(payload.get("confidence") or 0.0),
        "facets": list(cov_detail),
        "coverage": {t: len(hits) >= 2 for t, hits in cov_detail.items()},
        "sources": sources,
        "n_sources": payload.get("n_sources"),
        "citation_grounding": payload.get("citation_grounding"),
    }


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

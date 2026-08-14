"""Run-scoped rate-limit circuit breakers for research API sources.

Replaces the module-level mutable globals that persisted across
``run_pipeline()`` calls, causing failure cascades (a transient S2 429
or Ollama crash in one topic poisoned all subsequent topics).

Design (SOTA — matches PaperQA2 v5 "removed statefulness" + STORM
run-scoped context pattern):

* One :class:`CircuitBreaker` instance per API source.
* All breakers held in a :class:`RateLimitRegistry`.
* The pipeline creates a **fresh** registry per run via
  :func:`set_registry` → automatic reset, no globals to forget.
* Standalone script usage falls back to a module-level default registry.

Source paper: Future-House PaperQA2 v5 redesign ("Removed much of the
statefulness from the ``Docs`` object", Settings injection model,
centralized rate limits via LiteLLM). Stanford STORM uses per-run
callback/context injection. Both abandoned module-level mutable state.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """Per-source rate-limit state: pacing + circuit-open/close tracking.

    Thread-safe via internal lock. A fresh instance per pipeline run
    eliminates cross-run state pollution.

    Parameters
    ----------
    name
        Human-readable source name for log messages.
    threshold
        Consecutive failures before the circuit opens (source skipped).
    min_interval
        Minimum seconds between calls (polite pacing).
    """

    name: str
    threshold: int = 5
    min_interval: float = 0.5
    _failures: int = 0
    _open: bool = False
    _last_call: float = 0.0
    force_skipped: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def available(self) -> bool:
        """True if circuit is closed (source is usable)."""
        return not self._open

    def reset(self) -> None:
        """Reset failure tracking for a new pipeline run.

        Does NOT undo ``force_skip`` — user source preference persists
        across resets (matches original ``reset_s2_circuit`` semantics).
        """
        with self._lock:
            if self.force_skipped:
                return
            self._failures = 0
            self._open = False
            self._last_call = 0.0

    def pace(self) -> None:
        """Enforce minimum interval between consecutive calls."""
        with self._lock:
            wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            self._last_call = time.time()

    def record_success(self) -> None:
        """Reset failure counter and close circuit on successful API call."""
        with self._lock:
            self._failures = 0
            if not self.force_skipped:
                self._open = False

    def record_failure(self) -> None:
        """Track failure; open circuit after ``threshold`` consecutive failures."""
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and not self._open:
                self._open = True
                log.warning(
                    "%s circuit breaker OPENED after %d consecutive failures"
                    " — skipping %s for remainder of run",
                    self.name,
                    self._failures,
                    self.name,
                )

    def force_skip(self) -> None:
        """Skip ALL calls for this source (user deselected it).

        Persists across ``reset()`` — only ``enable()`` reverses it.
        """
        with self._lock:
            self.force_skipped = True
            self._open = True
            self._failures = self.threshold

    def enable(self) -> None:
        """Re-enable IF force-skipped.

        Does NOT reset a circuit opened by actual 429 rate-limiting —
        a rate-limited source will rate-limit again (matches original
        ``enable_s2_for_enrichment`` semantics).
        """
        with self._lock:
            if self.force_skipped:
                self.force_skipped = False
                self._open = False
                self._failures = 0
                log.debug("%s re-enabled (was force-skipped)", self.name)
            else:
                log.debug(
                    "%s circuit %s — not resetting (rate-limited)",
                    self.name,
                    "open" if self._open else "closed",
                )


def _s2_min_interval() -> float:
    """S2 unauthenticated: 100 req / 5 min = 1 req / 3s.
    With API key: 1 req / 0.1s (100x more).
    """
    if os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        return 0.1
    return 3.0


@dataclass
class RateLimitRegistry:
    """Container for all source circuit breakers.

    A fresh instance per pipeline run eliminates the cross-run state
    pollution that caused cascading failures. Created via
    ``RateLimitRegistry()`` at pipeline start, set active via
    :func:`set_registry`.
    """

    crossref: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("Crossref", threshold=3, min_interval=0.5)
    )
    openalex: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("OpenAlex", threshold=5, min_interval=1.0)
    )
    s2: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("S2", threshold=5, min_interval=_s2_min_interval())
    )
    arxiv: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("arXiv", threshold=3, min_interval=1.0)
    )
    ddgs: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("DDGS", threshold=3, min_interval=2.0)
    )
    ollama: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("Ollama", threshold=10, min_interval=0.0)
    )
    verify_s2: CircuitBreaker = field(
        default_factory=lambda: CircuitBreaker("verify_S2", threshold=1, min_interval=0.0)
    )

    def reset_all(self) -> None:
        """Reset all breakers for a fresh pipeline run."""
        for cb in (
            self.crossref,
            self.openalex,
            self.s2,
            self.arxiv,
            self.ddgs,
            self.ollama,
            self.verify_s2,
        ):
            cb.reset()


# ── Module-level default registry ────────────────────────────────────
# For standalone script usage (scripts run outside the pipeline).
# The pipeline swaps this for a fresh instance per run via ``set_registry``.
_registry: RateLimitRegistry = RateLimitRegistry()


def get_registry() -> RateLimitRegistry:
    """Return the active rate-limit registry."""
    return _registry


def set_registry(registry: RateLimitRegistry) -> RateLimitRegistry:
    """Swap the active registry. Returns the previous one."""
    global _registry
    old = _registry
    _registry = registry
    return old


__all__ = [
    "CircuitBreaker",
    "RateLimitRegistry",
    "get_registry",
    "set_registry",
]

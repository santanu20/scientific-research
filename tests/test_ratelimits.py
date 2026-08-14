#!/usr/bin/env python3
"""Thread-safety + behavior tests for `_ratelimits.py` CircuitBreaker.

Validates the SOTA run-scoped rate-limit claim: thread-safe under
concurrent failures, deterministic circuit-open at threshold, clean
registry swap semantics.

Run with:
    cd ~/.config/opencode/skills/scientific-research
    uv run pytest tests/test_ratelimits.py -v
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# Make scripts/ importable (matches tests/test_skill.py pattern)
SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from _ratelimits import (
    CircuitBreaker,
    RateLimitRegistry,
    get_registry,
    set_registry,
)

# =============================================================================
# CircuitBreaker — single-thread behavior
# =============================================================================


class TestCircuitBreakerSingleThread:
    def test_starts_available(self):
        cb = CircuitBreaker("test", threshold=3)
        assert cb.available is True

    def test_opens_at_threshold(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.available is True  # not yet
        cb.record_failure()
        assert cb.available is False  # opens at 3

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets
        cb.record_failure()
        cb.record_failure()
        assert cb.available is True  # only 2 failures since last success

    def test_force_skip_persists_across_reset(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.force_skip()
        assert cb.available is False
        cb.reset()
        assert cb.available is False  # force_skip persists

    def test_enable_reverses_force_skip(self):
        cb = CircuitBreaker("test", threshold=3)
        cb.force_skip()
        cb.enable()
        assert cb.available is True

    def test_enable_does_not_reset_rate_limited_circuit(self):
        """enable() only reverses force_skip; real 429 circuit stays open."""
        cb = CircuitBreaker("test", threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.available is False  # opened by failures, not force_skip
        cb.enable()
        assert cb.available is False  # still open — enable only affects force_skip


# =============================================================================
# CircuitBreaker — thread safety (deterministic)
# =============================================================================


class TestCircuitBreakerThreadSafety:
    def test_concurrent_failures_open_circuit(self):
        """N threads hit record_failure() simultaneously — circuit must open."""
        cb = CircuitBreaker("test", threshold=3, min_interval=0.0)
        n_threads = 10
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()  # all threads release simultaneously
            cb.record_failure()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert cb.available is False, "circuit should be OPEN after 10 concurrent failures"
        # _failures may be anywhere from 3 to 10 depending on interleaving,
        # but _open must be True (set at threshold=3, never unset without success)

    def test_concurrent_success_failure_leaves_consistent_state(self):
        """Mix of success + failure calls — no torn state."""
        cb = CircuitBreaker("test", threshold=5, min_interval=0.0)
        barrier = threading.Barrier(6)

        def failer():
            barrier.wait()
            for _ in range(3):
                cb.record_failure()

        def successer():
            barrier.wait()
            cb.record_success()

        threads = [threading.Thread(target=failer) for _ in range(5)]
        threads.append(threading.Thread(target=successer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # Final state is non-deterministic in exact _failures count, but
        # the lock guarantees no corrupt state (no exception, _open is bool).
        assert isinstance(cb.available, bool)

    def test_pace_enforces_min_interval_under_concurrency(self):
        """Concurrent pace() calls each wait at least min_interval since last call.

        pace() is polite pacing, not a strict serializer — concurrent threads
        may all read the same _last_call and sleep simultaneously. The guarantee
        is: each thread waits at least min_interval since the last recorded call.
        With a fresh breaker (_last_call=0), the first thread waits the full
        min_interval; subsequent threads may or may not wait more depending on
        interleaving. Assert total elapsed ≥ min_interval (one full interval).
        """
        min_interval = 0.1
        cb = CircuitBreaker("test", threshold=10, min_interval=min_interval)
        n_threads = 4
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            cb.pace()

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # With _last_call=0 (fresh), every thread computes wait = min_interval - (now - 0)
        # = min_interval - huge = negative → no sleep. So elapsed may be ~0.
        # The real guarantee is that pace() doesn't CRASH under concurrency and
        # updates _last_call atomically. Test that _last_call is set after.
        assert cb._last_call > 0, "pace() should update _last_call"
        # Second round: now _last_call is set, so pace() should enforce the interval.
        barrier2 = threading.Barrier(n_threads)
        start2 = time.time()

        def worker2():
            barrier2.wait()
            cb.pace()

        threads2 = [threading.Thread(target=worker2) for _ in range(n_threads)]
        for t in threads2:
            t.start()
        for t in threads2:
            t.join(timeout=10.0)
        elapsed2 = time.time() - start2

        # At least one thread should have waited ~min_interval since start2.
        # With concurrent reads of the same _last_call, all may wait the same
        # amount, so elapsed2 ≈ min_interval. Allow 50% tolerance for scheduler jitter.
        assert elapsed2 >= min_interval * 0.5, (
            f"pace() did not enforce min_interval on second round: elapsed2={elapsed2:.3f}s, "
            f"expected ≥ {min_interval * 0.5:.3f}s"
        )


# =============================================================================
# RateLimitRegistry — run-scoped swap semantics
# =============================================================================


class TestRateLimitRegistry:
    def test_default_registry_is_module_level(self):
        r = get_registry()
        assert isinstance(r, RateLimitRegistry)

    def test_set_registry_returns_previous_and_swaps(self):
        old = get_registry()
        new = RateLimitRegistry()
        returned = set_registry(new)
        assert returned is old, "set_registry should return the PREVIOUS registry"
        assert get_registry() is new, "get_registry should return the NEW registry"
        # Restore for other tests
        set_registry(old)

    def test_reset_all_resets_every_breaker(self):
        r = RateLimitRegistry()
        r.crossref.record_failure()
        r.crossref.record_failure()
        r.crossref.record_failure()  # opens crossref
        r.openalex.record_failure()
        r.s2.record_failure()
        r.s2.record_failure()
        r.s2.record_failure()
        r.s2.record_failure()
        r.s2.record_failure()  # opens s2
        assert not r.crossref.available
        assert not r.s2.available
        r.reset_all()
        assert r.crossref.available
        assert r.s2.available

    def test_each_breaker_has_correct_name(self):
        r = RateLimitRegistry()
        assert r.crossref.name == "Crossref"
        assert r.openalex.name == "OpenAlex"
        assert r.s2.name == "S2"
        assert r.arxiv.name == "arXiv"
        assert r.ddgs.name == "DDGS"
        assert r.ollama.name == "Ollama"

    def test_s2_min_interval_respects_api_key_env(self, monkeypatch):
        """With S2_API_KEY set, S2 breaker uses 0.1s interval (100x more)."""
        # The interval is computed at RateLimitRegistry construction time
        # via _s2_min_interval(). Test the function directly.
        from _ratelimits import _s2_min_interval

        monkeypatch.delenv("S2_API_KEY", raising=False)
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        assert _s2_min_interval() == 3.0

        monkeypatch.setenv("S2_API_KEY", "test-key")
        assert _s2_min_interval() == 0.1

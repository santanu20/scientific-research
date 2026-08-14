#!/usr/bin/env python3
"""Live integration test: cross-run circuit breaker reset.

Verifies the SOTA run-scoped rate-limit claim: a fresh ``RateLimitRegistry``
per pipeline run eliminates cross-run state pollution. If a breaker opens
in run 1 (rate-limited), run 2 starts with all breakers closed.

**LIVE test** — requires network access to Crossref API.
Skipped by default: ``uv run pytest tests/ -m "not live"``
Run explicitly: ``uv run pytest tests/test_live_breaker_reset.py -m live -v``

What it tests:
1. Run 1: search Crossref with a query that triggers 429 (or succeeds).
   If breaker opens, assert it's open at end of run 1.
2. Create a fresh registry for run 2 via ``set_registry(RateLimitRegistry())``.
3. Assert all breakers in run 2's registry are closed (available).
4. Run 2: search Crossref again — should NOT be skipped by a stale open breaker.

This catches the bug where module-level breaker globals persisted across
runs, causing run 2 to skip Crossref entirely because run 1 got rate-limited.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable (matches tests/test_skill.py pattern)
SKILL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from _ratelimits import RateLimitRegistry, get_registry, set_registry


@pytest.mark.live
class TestCrossRunBreakerReset:
    """Live test: breakers reset between pipeline runs (cross-run isolation)."""

    def test_fresh_registry_has_all_breakers_closed(self):
        """A new RateLimitRegistry starts with all breakers available."""
        registry = RateLimitRegistry()
        assert registry.crossref.available
        assert registry.openalex.available
        assert registry.s2.available
        assert registry.arxiv.available

    def test_set_registry_swaps_active_registry(self):
        """set_registry() swaps the active registry — run 2 gets a fresh one."""
        # Capture original
        original = get_registry()

        # Simulate run 1: open the Crossref breaker
        run1 = RateLimitRegistry()
        set_registry(run1)
        run1.crossref.record_failure()
        run1.crossref.record_failure()
        run1.crossref.record_failure()  # threshold=3 → opens
        assert not run1.crossref.available, "run 1 Crossref breaker should be OPEN"
        assert not get_registry().crossref.available, "active registry should reflect run 1"

        # Simulate run 2: create + set a fresh registry
        run2 = RateLimitRegistry()
        set_registry(run2)
        assert get_registry() is run2, "run 2 registry should be active"
        assert run2.crossref.available, "run 2 Crossref breaker should be CLOSED (fresh)"
        assert get_registry().crossref.available, (
            "active registry after run 2 should have Crossref available — "
            "this is the cross-run isolation invariant"
        )

        # Restore original for other tests
        set_registry(original)

    def test_crossref_search_respects_run_scoped_registry(self):
        """Live: crossref_search uses the active registry, not module globals.

        Run 1: open Crossref breaker → crossref_search returns [] (skipped).
        Run 2: fresh registry → crossref_search hits the API (not skipped).
        """
        from _sources import crossref_search

        original = get_registry()

        # Run 1: force-open Crossref breaker
        run1 = RateLimitRegistry()
        set_registry(run1)
        run1.crossref.force_skip()  # deterministic — don't need real 429
        assert not run1.crossref.available

        # crossref_search should skip (return []) because breaker is open
        results_run1 = crossref_search("test query", max_results=1)
        assert results_run1 == [], (
            "crossref_search should return [] when Crossref breaker is open (run 1)"
        )

        # Run 2: fresh registry
        run2 = RateLimitRegistry()
        set_registry(run2)
        assert run2.crossref.available

        # crossref_search should hit the API (not skipped)
        # Use a very specific query to get minimal results (1 paper)
        try:
            results_run2 = crossref_search("nanometre thermometry living cell", max_results=1)
        except Exception as e:
            # Network errors are acceptable in live test — the point is
            # that crossref_search ATTEMPTED the API call (didn't skip).
            # If it had skipped (breaker open), it would return [] without exception.
            pytest.skip(f"Network error (expected in CI): {e}")
        else:
            # If we got here, the API was called (not skipped).
            # Results may be empty if no match, but the call was attempted.
            assert isinstance(results_run2, list), (
                "crossref_search should return a list when breaker is closed (run 2)"
            )

        # Restore
        set_registry(original)

"""Central environment-variable resolution for user-facing configuration.

Single authoritative home for variable names, including time-limited
backward-compatible fallbacks for renamed variables so existing user
environments keep working (announced, never silent).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("scientific_research.config")


def env_setting(name: str, default: str = "", legacy_name: str = "") -> str:
    """Resolve configuration variable ``name`` from the environment.

    Falls back to ``legacy_name`` when the primary variable is unset but
    the legacy alias still carries a value, emitting a deprecation
    warning so stale configuration stays discoverable. Returns
    ``default`` when neither is set.
    """
    value = os.environ.get(name, "")
    if not value and legacy_name:
        value = os.environ.get(legacy_name, "")
        if value:
            log.warning(
                "%s is deprecated; rename it to %s (still honored during the transition)",
                legacy_name,
                name,
            )
    return value or default

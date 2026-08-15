"""Shared skill-venv bootstrap for all entry scripts.

Stdlib-only — importable BEFORE the venv exists. Consolidates the 13×
duplicated inline bootstrap blocks (2026-08-14) into one site.

Behavior (identical to the former inline blocks):
  1. create skill venv on first run (uv, python 3.13) unless
     SCIENTIFIC_RESEARCH_NO_BOOTSTRAP=1
  2. re-exec into the venv when running under a different interpreter
     (skipped when SCIENTIFIC_RESEARCH_NO_SKILL_VENV=1)
  3. verify required imports; missing → auto-install once via uv
     (stale-venv self-repair), re-check, then fail loud if still missing

Entry-script usage (must stay at module top, before heavy imports):

    if __name__ == "__main__":
        import _bootstrap

        _bootstrap.ensure_env()
"""

from __future__ import annotations

import os
import subprocess
import sys

SKILL_VENV = os.path.expanduser("~/.config/opencode/skills/scientific-research/.venv")
SKILL_VENV_PY = os.path.join(SKILL_VENV, "bin", "python")

BASE_IMPORTS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "sklearn",
    "httpx",
    "pypdf",
)
BASE_INSTALLS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "scikit-learn",
    "httpx",
    "pypdf",
    "matplotlib",  # PRISMA flow diagram rendering (discover.py)
    "pytest",  # test-suite runner (kept in venv for `uv run pytest` parity)
    "ruff",  # lint gate
)


def _install(installs: tuple[str, ...]) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", SKILL_VENV_PY, *installs],
        check=True,
        capture_output=True,
    )


def _missing_imports(req: tuple[str, ...]) -> list[str]:
    missing = []
    for mod in req:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    return missing


def ensure_env(
    extra_imports: tuple[str, ...] = (),
    extra_installs: tuple[str, ...] = (),
) -> None:
    """Bootstrap the skill venv. Call only from ``if __name__ == "__main__"``."""
    req_imports = BASE_IMPORTS + tuple(extra_imports)
    req_installs = BASE_INSTALLS + tuple(extra_installs)
    if os.environ.get("SCIENTIFIC_RESEARCH_NO_SKILL_VENV"):
        return
    # 1. first run: create venv + install everything
    if not os.path.exists(SKILL_VENV_PY) and not os.environ.get(
        "SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"
    ):
        sys.stderr.write(
            "Bootstrapping scientific-research skill venv (one-time setup)...\n"
        )
        try:
            subprocess.run(
                ["uv", "venv", SKILL_VENV, "--python", "3.13"],
                check=True,
                capture_output=True,
            )
            _install(req_installs)
            sys.stderr.write("scientific-research skill venv ready.\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            sys.stderr.write(
                f"Failed to auto-bootstrap: {ex}\n"
                f"Manual: uv venv {SKILL_VENV} --python 3.13 && "
                f"uv pip install --python {SKILL_VENV_PY} {' '.join(req_installs)}\n"
            )
            sys.exit(2)
    # 2. re-exec into the venv when under a different interpreter
    if os.path.exists(SKILL_VENV_PY) and os.path.normpath(
        sys.prefix
    ) != os.path.normpath(SKILL_VENV):
        os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        os.execv(
            SKILL_VENV_PY, [SKILL_VENV_PY, os.path.abspath(sys.argv[0])] + sys.argv[1:]
        )
    # 3. verify deps; stale venv → auto-install once, re-check, fail loud
    missing = _missing_imports(req_imports)
    if missing and not os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        try:
            sys.stderr.write(f"Installing missing deps: {', '.join(missing)}\n")
            _install(req_installs)
            missing = _missing_imports(req_imports)
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            sys.stderr.write(f"Auto-install failed: {ex}\n")
    if missing:
        sys.stderr.write(
            f"FATAL: missing required deps: {', '.join(missing)}\n"
            f"Install: uv pip install --python {SKILL_VENV_PY} {' '.join(req_installs)}\n"
        )
        sys.exit(2)

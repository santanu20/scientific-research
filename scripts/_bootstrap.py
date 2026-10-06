"""Shared research-venv bootstrap for all entry scripts.

Stdlib-only — importable BEFORE the venv exists. Consolidates the 13×
duplicated inline bootstrap blocks (2026-08-14) into one site.

Isolation contract (2026-09-01): the host app and the agent skill folders are
separate projects — no shared paths, no borrowed venvs. This venv is
owned by the skill and lives under its own XDG cache.

Behavior:
  0. when the CURRENT interpreter already imports everything (e.g. the host
     uv venv), return immediately — never leave a capable environment
  1. create the research venv on first run (uv, python 3.14) unless
     SCIENTIFIC_RESEARCH_NO_BOOTSTRAP=1
  2. re-exec into the venv when running under a different interpreter
     (skipped when SCIENTIFIC_RESEARCH_NO_SKILL_VENV=1 — legacy var name
     kept so existing toggles keep working)
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
from pathlib import Path

# Self-relative: the venv belongs to THIS skill tree, wherever it lives
# (2026-10-06: previous hardcoded home path went stale after the tree moved,
# causing silent re-exec into a resurrected venv at the dead location).
_SKILL_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_VENV = str(_SKILL_ROOT / ".venv")
RESEARCH_VENV_PY = os.path.join(RESEARCH_VENV, "bin", "python")

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
    "orjson",
    "fastembed",
    "networkx",
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
    "orjson",  # pipeline.py JSON I/O (missing here broke re-exec'd runs)
    "fastembed",  # BGE embeddings (_embeddings.py, _ollama_extract.py)
    "networkx",  # citation graph (_citation_graph.py)
    "matplotlib",  # PRISMA flow diagram rendering (discover.py)
    "pytest",  # test-suite runner (kept in venv for `uv run pytest` parity)
    "ruff==0.16.3",  # lint gate — PINNED (A10): unpinned tools drift between sessions
    "pyright==1.1.411",  # type gate — PINNED (A10)
)


def _install(installs: tuple[str, ...]) -> None:
    subprocess.run(
        ["uv", "pip", "install", "--python", RESEARCH_VENV_PY, *installs],
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
    """Bootstrap the research venv. Call only from ``if __name__ == "__main__"``."""
    req_imports = BASE_IMPORTS + tuple(extra_imports)
    req_installs = BASE_INSTALLS + tuple(extra_installs)
    if os.environ.get("SCIENTIFIC_RESEARCH_NO_SKILL_VENV"):
        return
    # 0. current interpreter already provides everything (host uv venv)
    if not _missing_imports(req_imports):
        return
    # 1. first run: create venv + install everything
    if not os.path.exists(RESEARCH_VENV_PY) and not os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        sys.stderr.write("Bootstrapping research venv (one-time setup)...\n")
        try:
            subprocess.run(
                ["uv", "venv", RESEARCH_VENV, "--python", "3.14"],
                check=True,
                capture_output=True,
            )
            _install(req_installs)
            sys.stderr.write("research venv ready.\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            sys.stderr.write(
                f"Failed to auto-bootstrap: {ex}\n"
                f"Manual: uv venv {RESEARCH_VENV} --python 3.14 && "
                f"uv pip install --python {RESEARCH_VENV_PY} {' '.join(req_installs)}\n"
            )
            sys.exit(2)
    # 2. re-exec into the venv when under a different interpreter
    if os.path.exists(RESEARCH_VENV_PY) and os.path.normpath(sys.prefix) != os.path.normpath(RESEARCH_VENV):
        os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        os.execv(RESEARCH_VENV_PY, [RESEARCH_VENV_PY, os.path.abspath(sys.argv[0]), *sys.argv[1:]])
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
            f"Install: uv pip install --python {RESEARCH_VENV_PY} {' '.join(req_installs)}\n"
        )
        sys.exit(2)

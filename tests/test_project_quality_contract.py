"""Project metadata contract tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_supported_python_and_unique_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["requires-python"] == ">=3.12"
    dependency_names = []
    for dependency in project["dependencies"]:
        name = dependency.split(">=")[0].split("==")[0].split("[")[0].lower()
        dependency_names.append(name)
    assert len(dependency_names) == len(set(dependency_names)), "duplicate dependency declarations"


def test_ci_installs_project_and_runs_declared_quality_gates() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "pip install -e ." in workflow
    assert "ruff check scripts tests" in workflow
    assert 'pytest tests -m "not live" -q' in workflow
    assert "permissions:" in workflow

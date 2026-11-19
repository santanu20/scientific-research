"""Run provenance manifest — FAIR / PRISMA-S reproducibility audit trail.

Every pipeline stage appends one record to <results_dir>/run_provenance.json:
input/output file hashes (SHA-256), parameters, dependency versions, and
timestamps. Purpose: any figure/claim in a brief can be traced to the exact
code state and inputs that produced it (enterprise audit requirement).

Usage:
    from _provenance import record_stage
    record_stage("discovery", params={...}, inputs={...}, outputs={...},
                 results_dir=Path("research_outputs"))
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib import metadata as _ilmd
from pathlib import Path
from typing import Any

PROVENANCE_FILENAME = "run_provenance.json"


class ProvenanceManifestError(RuntimeError):
    """Raised when an existing provenance manifest cannot be trusted."""


# deps whose versions matter for reproducibility (skip silently if absent)
_TRACKED_DEPS = (
    "numpy",
    "scipy",
    "scikit-learn",
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "fastembed",
)


def _file_info(path: Path | None) -> dict[str, Any] | None:
    """SHA-256 + size + mtime for a file; None if path is None or missing."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return {"path": str(path), "missing": True}
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(stat.st_mtime)),
    }


def _env_snapshot() -> dict[str, Any]:
    deps: dict[str, str] = {}
    for name in _TRACKED_DEPS:
        try:
            deps[name] = _ilmd.version(name)
        except _ilmd.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "deps": deps,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read an existing manifest without ever replacing untrusted contents."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ProvenanceManifestError(
            f"Existing provenance manifest is unreadable; preserving it at {path}: {exc}"
        ) from exc
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    if isinstance(data, dict):
        return [data]
    raise ProvenanceManifestError(
        f"Existing provenance manifest has an invalid root/schema at {path}; "
        "expected an object or a list of objects"
    )


def record_stage(
    stage: str,
    params: dict,
    inputs: dict[str, Path | None],
    outputs: dict[str, Path | None],
    results_dir: Path,
    extra: dict | None = None,
) -> dict:
    """Append one stage record to run_provenance.json.

    Invalid existing manifests fail closed: they are never silently reset,
    because that would destroy the only record of earlier pipeline stages.
    Writes are atomic; the original manifest remains intact on failure.
    """
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string")

    results_dir = Path(results_dir)
    manifest_path = results_dir / PROVENANCE_FILENAME
    manifest = _read_manifest(manifest_path)
    record = {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": params,
        "inputs": {k: _file_info(v) for k, v in inputs.items()},
        "outputs": {k: _file_info(v) for k, v in outputs.items()},
        "env": _env_snapshot(),
    }
    if extra:
        record["extra"] = extra
    manifest.append(record)

    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_name(f".{manifest_path.name}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(manifest_path)
    finally:
        tmp.unlink(missing_ok=True)
    return record


def load_manifest(results_dir: Path) -> list[dict]:
    """Load provenance, preserving a strict distinction between absent and corrupt."""
    path = Path(results_dir) / PROVENANCE_FILENAME
    return _read_manifest(path)

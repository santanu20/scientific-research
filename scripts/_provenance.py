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

PROVENANCE_FILENAME = "run_provenance.json"

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


def _file_info(path: Path | None) -> dict | None:
    """SHA-256 + size + mtime for a file; None if path is None or missing."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return {"path": str(path), "missing": True}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(path),
        "sha256": h.hexdigest(),
        "size_bytes": path.stat().st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(path.stat().st_mtime)),
    }


def _env_snapshot() -> dict:
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


def record_stage(
    stage: str,
    params: dict,
    inputs: dict[str, Path | None],
    outputs: dict[str, Path | None],
    results_dir: Path,
    extra: dict | None = None,
) -> dict:
    """Append one stage record to run_provenance.json. Returns the record.

    Inputs/outputs: {logical_name: path}. Missing files are recorded as
    {"missing": true} — provenance must show gaps, not hide them.
    """
    results_dir = Path(results_dir)
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
    manifest_path = results_dir / PROVENANCE_FILENAME
    manifest: list[dict] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, list):
                manifest = [manifest]
        except (json.JSONDecodeError, OSError):
            manifest = []  # corrupt manifest → start fresh rather than fail the run
    manifest.append(record)
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)
    return record


def load_manifest(results_dir: Path) -> list[dict]:
    """Load the provenance manifest; empty list if absent/corrupt."""
    p = Path(results_dir) / PROVENANCE_FILENAME
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    except (json.JSONDecodeError, OSError):
        return []

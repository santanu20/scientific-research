"""Pins for the KB-OCR sidecar cache (2026-09-03): the local-library grind.

A default research run re-parsed the whole local PDF library every run
because build_local_index WROTE a sidecar .md per PDF but never read it
back. The sidecar read-back pins live here (standalone: PyMuPDF text).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def test_unit__local_index__sidecar_cache_prevents_reparse(monkeypatch, tmp_path) -> None:
    """A fresh sidecar .md (mtime >= pdf) is read back — the PDF is NOT
    re-parsed. This is the 15-GPU-minute -> milliseconds fix."""
    import _localindex as ft

    calls: list[Path] = []
    monkeypatch.setattr(ft, "_parse_pdf_to_text", lambda p, **k: calls.append(p) or "")

    pdf = tmp_path / "craig1973.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    sidecar = tmp_path / "craig1973.md"
    sidecar.write_text(
        "<!-- Source: PDF craig1973.pdf -->\n\n" + "pyrite pentlandite text. " * 30,
        encoding="utf-8",
    )
    # sidecar newer than pdf
    import os

    os.utime(pdf, (1000, 1000))

    idx = ft.build_local_index(tmp_path)
    assert calls == [], "sidecar cache must prevent re-parsing"
    pdf_entries = [e for e in idx if e["source"] == "pdf"]
    assert len(pdf_entries) == 1
    assert pdf_entries[0]["path"] == sidecar


def test_unit__local_index__stale_sidecar_reparses(monkeypatch, tmp_path) -> None:
    """PDF newer than sidecar (library updated) → re-parse, rewrite."""
    import os

    import _localindex as ft

    calls: list[Path] = []
    monkeypatch.setattr(
        ft, "_parse_pdf_to_text", lambda p, **k: calls.append(p) or "fresh text " * 50
    )

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    sidecar = tmp_path / "paper.md"
    sidecar.write_text("stale short", encoding="utf-8")
    os.utime(sidecar, (1000, 1000))  # sidecar OLDER

    idx = ft.build_local_index(tmp_path)
    assert calls == [pdf]
    assert any(e["source"] == "pdf" for e in idx)
    assert sidecar.read_text(encoding="utf-8").startswith("<!--")


def test_unit__local_index__sidecar_not_double_counted(monkeypatch, tmp_path) -> None:
    """The md loop must skip PDF sidecars (one PDF = one entry, not an
    'md' entry + a 'pdf' entry pointing at the same file)."""
    import _localindex as ft

    monkeypatch.setattr(ft, "_parse_pdf_to_text", lambda p, **k: "text " * 60)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    idx = ft.build_local_index(tmp_path)
    assert len(idx) == 1
    assert idx[0]["source"] == "pdf"



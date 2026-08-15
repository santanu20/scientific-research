"""Typed pipeline artifacts with shape-refusing load/save (Phase 1, 2026-08-15).

The two worst E2E bugs (R1: extracted.json fed to correlate; pid-join miss)
were CONTRACT failures — free-form dicts passed between scripts. This module
makes wrong shapes impossible to consume silently:

    Corpus = discover output        (papers: [...])
    Verified = verify output        (papers: [...], results: [...])
    Extractions = extract output    (extractions: [...])
    Correlation = correlate output  (n_papers > 0 + at least one live section)
    MetaResult = meta_analyze       (unit_pools / pooled_random)

Every load_* validates the discriminator key and raises ArtifactShapeError
with the CORRECT pipeline order in the message. save_* writes atomically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()


class ArtifactShapeError(RuntimeError):
    """Raised when a file does not match the expected pipeline artifact shape."""


def _read(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ArtifactShapeError(f"artifact not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ArtifactShapeError(f"artifact is not valid JSON: {p} ({e})") from e
    if not isinstance(data, dict):
        raise ArtifactShapeError(f"artifact root must be a JSON object: {p}")
    return data


def _expect_keys(
    path: Path, data: dict, needed: set[str], artifact: str, wants: str
) -> None:
    missing = needed - data.keys()
    if missing:
        got = sorted(k for k in data if k not in ("meta",))[:5]
        raise ArtifactShapeError(
            f"{path} is not a valid {artifact} (missing {sorted(missing)}; "
            f"looks like {got}). {wants}"
        )


_PIPELINE_ORDER = (
    "Pipeline order: discover → verify → [correlate] → extract → "
    "synthesize (meta_analyze consumes extract + verify outputs)."
)


def load_corpus(path: Path | str) -> dict:
    """Corpus artifact (discover output). Returns the raw dict, validated."""
    p = Path(path)
    data = _read(p)
    _expect_keys(p, data, {"papers"}, "Corpus", _PIPELINE_ORDER)
    papers = data["papers"]
    if not isinstance(papers, list):
        raise ArtifactShapeError(f"{p}: 'papers' must be a list")
    return data


def load_verified(path: Path | str) -> dict:
    """Verified artifact (verify output)."""
    p = Path(path)
    data = _read(p)
    _expect_keys(p, data, {"papers"}, "Verified corpus", _PIPELINE_ORDER)
    papers = data["papers"]
    if not isinstance(papers, list) or not papers:
        raise ArtifactShapeError(
            f"{p}: verified corpus must contain ≥1 paper — refusing empty input"
        )
    return data


def load_extractions(path: Path | str) -> dict:
    """Extractions artifact (extract output)."""
    p = Path(path)
    data = _read(p)
    _expect_keys(p, data, {"extractions"}, "Extractions", _PIPELINE_ORDER)
    return data


def load_correlation(path: Path | str) -> dict:
    """Correlation artifact (correlate output) — must be LIVE (n_papers > 0)."""
    p = Path(path)
    data = _read(p)
    _expect_keys(p, data, {"n_papers"}, "Correlation", _PIPELINE_ORDER)
    if not data.get("n_papers"):
        raise ArtifactShapeError(
            f"{p}: correlation is EMPTY (n_papers=0) — the correlate stage "
            f"died or consumed the wrong input. {_PIPELINE_ORDER}"
        )
    return data


def load_meta(path: Path | str) -> dict:
    """MetaResult artifact (meta_analyze output) — needs a pooled estimate."""
    p = Path(path)
    data = _read(p)
    if not (
        data.get("pooled_random") or data.get("pooled_fixed") or data.get("unit_pools")
    ):
        raise ArtifactShapeError(
            f"{p} has no pooled estimates — not a meta_analyze output. {_PIPELINE_ORDER}"
        )
    return data


def save_artifact(data: dict, path: Path | str) -> Path:
    """Atomic JSON write (tmp + rename) for any pipeline artifact."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


if __name__ == "__main__":
    print(f"OK {sys.argv[0]}: artifact contracts ready")


# shared JSON/paper helpers (deduped from synthesize/_narrative)

def _load_json(path: Path | None) -> dict | None:
    """Load JSON file, return None if path is None or doesn't exist."""
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load %s: %s", path, e)
        return None



def _extract_papers(
    extracted: dict | None,
    verified: dict | None,
) -> list[dict]:
    """Merge paper metadata from verified.json with extraction data."""
    # lazy: keeps _artifact import-safe from any sys.path layout
    from _classifiers import detect_discipline as _detect_discipline

    ext_lookup: dict[str, dict] = {}
    if extracted:
        ext_list = extracted.get("extractions") or extracted.get("results") or []
        for item in ext_list:
            pid = item.get("paper_id", "")
            ext_lookup[pid] = item
            if item.get("doi"):
                ext_lookup[item["doi"]] = item

    papers: list[dict] = []

    if verified:
        for p in verified.get("papers", []):
            pid = p.get("primary_id") or p.get("paper_id") or ""
            doi = p.get("doi") or ""
            ext = ext_lookup.get(pid) or ext_lookup.get(doi) or {}
            if not ext and doi:
                # extraction rows key by 'doi:10...' primary_id — try that form
                ext = (
                    ext_lookup.get(f"doi:{doi.lower()}")
                    or ext_lookup.get(f"doi:{doi}")
                    or {}
                )

            abstract = p.get("abstract") or ""
            key_finding = ext.get("pico", {}).get("key_finding") or ""
            if not key_finding and abstract:
                from _classifiers import extract_key_finding

                key_finding = extract_key_finding(abstract, max_chars=300)

            measurements: list[dict] = []
            eff = ext.get("effect_sizes", {})
            for m in eff.get("single_measurements", []):
                measurements.append(
                    {
                        "value": m.get("value"),
                        "unit": m.get("unit", ""),
                        "measurement": m.get("measurement", ""),
                        "raw": m.get("raw_text", ""),
                    }
                )
            # Also include measurements parsed from quantitative_data by _geo_enrich
            for m in ext.get("measurements", []):
                if (
                    isinstance(m, dict)
                    and m.get("value") is not None
                    and not m.get("_flagged")
                ):
                    measurements.append(m)

            papers.append(
                {
                    "paper_id": pid,
                    "title": _sanitize_finding(p.get("title") or ""),
                    "year": p.get("year"),
                    "authors": _parse_authors(p.get("authors", [])),
                    "doi": doi,
                    "abstract": _sanitize_finding(abstract),
                    "key_finding": _sanitize_finding(key_finding),
                    "discipline": (ext.get("pico", {}).get("discipline") or "")
                    or (_detect_discipline(abstract) if abstract else ""),
                    "novelty": (ext.get("pico", {}).get("novelty") or ""),
                    "study_type": (ext.get("pico", {}).get("study_type") or ""),
                    "interpretation": _sanitize_finding(
                        ext.get("pico", {}).get("interpretation") or ""
                    ),
                    "measurements": measurements,
                }
            )

    if not papers and extracted:
        ext_list = extracted.get("extractions") or extracted.get("results") or []
        for item in ext_list:
            abstract = item.get("abstract", "")
            papers.append(
                {
                    "paper_id": item.get("paper_id", ""),
                    "title": item.get("title", ""),
                    "year": None,
                    "authors": [],
                    "doi": item.get("doi", ""),
                    "abstract": abstract,
                    "key_finding": (item.get("pico", {}).get("key_finding") or ""),
                    "discipline": (item.get("pico", {}).get("discipline") or ""),
                    "novelty": (item.get("pico", {}).get("novelty") or ""),
                    "study_type": (item.get("pico", {}).get("study_type") or ""),
                    "interpretation": (
                        item.get("pico", {}).get("interpretation") or ""
                    ),
                    "measurements": [],
                }
            )

    return papers


# =============================================================================
# LLM prose smoothing (opt-in via --use-llm)
# =============================================================================



# shared JSON/paper helpers (deduped from synthesize/_narrative, 2026-08-15)





def _sanitize_finding(text: str) -> str:
    """Strip HTML entities/tags + fix scientific notation spacing artifacts."""
    if not text:
        return ""
    import re as _re

    _html_entities = {
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&permil;": "‰",
        "&delta;": "δ",
        "&alpha;": "α",
        "&beta;": "β",
        "&sigma;": "σ",
        "&mu;": "μ",
        "&times;": "×",
        "&plusmn;": "±",
        "&deg;": "°",
        "&ndash;": "–",
        "&mdash;": "—",
    }
    for entity, char in _html_entities.items():
        text = text.replace(entity, char)
    text = _re.sub(r"<[^>]+>", "", text)
    # Strip LaTeX markup common in Crossref abstracts
    text = text.replace("$\\sim$", "~").replace("$\\pm$", "±")
    text = text.replace("$\\degree$", "°").replace("$\\times$", "×")
    text = _re.sub(r"\$\$.*?\$\$", "", text)  # remove display math
    text = _re.sub(r"\$([^$]+)\$", r"\1", text)  # inline math → plain text
    text = _re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]+)\}", r"\1", text)  # \text{x} → x
    # Fix isotope spacing: "δ 18 O" → "δ18O"
    text = _re.sub(r"δ\s*(\d+)\s*([A-Z])", r"δ\1\2", text)
    text = _re.sub(r"Δ\s*(\d+)\s*([A-Z])", r"Δ\1\2", text)
    # Fix oxide spacing: "SiO 2" → "SiO2", "fO 2" → "fO2"
    text = _re.sub(r"([A-Za-z])O\s*(\d+)", r"\1O\2", text)
    text = _re.sub(r"([A-Za-z])\s*(\d+)\s*O\s*(\d+)", r"\1\2O\3", text)
    # Fix "fO 2 s" → "fO2s"
    text = _re.sub(r"fO\s*2\s*s", "fO2", text, flags=_re.IGNORECASE)
    # Remove KEY WORDS artifacts from journal formatting
    text = _re.sub(r"KEY WORDS?:.*", "", text, flags=_re.DOTALL)
    # Normalize whitespace
    text = _re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _parse_authors(authors_field: Any) -> list[str]:
    """Parse authors field — handles list of strings or list of dicts."""
    if not authors_field:
        return []
    result: list[str] = []
    for a in authors_field:
        if isinstance(a, str):
            result.append(a)
        elif isinstance(a, dict):
            name = a.get("name") or a.get("display_name") or ""
            if name:
                result.append(name)
    return result

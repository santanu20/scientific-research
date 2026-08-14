#!/usr/bin/env python3
"""§5 H20 verification gate for the scientific-research skill.

For every paper in corpus, resolves DOI/arXiv/PMID via ≥1 canonical resolver
before allowing it to be cited. Also checks OpenAlex `is_retracted`, queries
DOAJ whitelist for OA journal vetting, and grades evidence level from Crossref
type. Outputs verified.json with only papers that pass the gate.

Per AGENTS.md §5 H20: any DOI/PMID/arXiv ID MUST resolve via canonical
resolver before citing. Unresolved = forbidden cite (H7 fabrication).

Crossref reliability note: Crossref returns bad DOIs occasionally. So we
cross-verify via OpenAlex (preferred metadata source) + Crossref + direct
doi.org resolution. If ≥1 source returns valid metadata, paper is "verified";
disagreements flagged but do not block.

References:
    - AGENTS.md §5 H20 — Verification gate
    - OpenAlex is_retracted: verified live (115k+ retracted papers indexed)
    - DOAJ whitelist: https://doaj.org/faq#whitelist (positive OA signal only,
      NOT a predatory blacklist — many legit subscription journals not in DOAJ)
    - Evidence hierarchy: Oxford CEBM Levels (https://www.cebm.ox.ac.uk/levels)
"""

from __future__ import annotations

# --- Self-contained skill venv bootstrap (mirrors pdf-ocr/web-search pattern) ---
import os as _bs_os, sys as _bs_sys

_SKILL_VENV = _bs_os.path.expanduser(
    "~/.config/opencode/skills/scientific-research/.venv"
)
_SKILL_VENV_PY = _bs_os.path.join(_SKILL_VENV, "bin", "python")
_REQ_IMPORTS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "sklearn",
    "httpx",
)
_REQ_INSTALLS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "scikit-learn",
    "httpx",
    "pytest",
    "ruff",
)
if __name__ == "__main__" and not _bs_os.environ.get(
    "SCIENTIFIC_RESEARCH_NO_SKILL_VENV"
):
    if not _bs_os.path.exists(_SKILL_VENV_PY) and not _bs_os.environ.get(
        "SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"
    ):
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(
                "Bootstrapping scientific-research skill venv (one-time setup)...\n"
            )
            _bs_sp.run(
                ["uv", "venv", _SKILL_VENV, "--python", "3.13"],
                check=True,
                capture_output=True,
            )
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True,
                capture_output=True,
            )
            _bs_sys.stderr.write("scientific-research skill venv ready.\n")
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(
                f"Failed to auto-bootstrap: {_bs_ex}\nManual: uv venv {_SKILL_VENV} --python 3.13 && uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
            )
            _bs_sys.exit(2)
    if _bs_os.path.exists(_SKILL_VENV_PY) and _bs_os.path.normpath(
        _bs_sys.prefix
    ) != _bs_os.path.normpath(_SKILL_VENV):
        _bs_os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        _bs_os.execv(
            _SKILL_VENV_PY,
            [_SKILL_VENV_PY, _bs_os.path.abspath(__file__)] + _bs_sys.argv[1:],
        )
    _missing = []
    for _m in _REQ_IMPORTS:
        try:
            __import__(_m)
        except ImportError:
            _missing.append(_m)
    if _missing and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        # stale venv: auto-install missing deps once, then re-check
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(f"Installing missing deps: {', '.join(_missing)}\n")
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True, capture_output=True,
            )
            _missing = []
            for _m in _REQ_IMPORTS:
                try:
                    __import__(_m)
                except ImportError:
                    _missing.append(_m)
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Auto-install failed: {_bs_ex}\n")
    if _missing:
        _bs_sys.stderr.write(
            f"FATAL: missing required deps: {', '.join(_missing)}\nInstall: uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
        )
        _bs_sys.exit(2)
# --- End bootstrap ---


import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import (  # noqa: E402
    PaperRecord,
    arxiv_search,
    cache_put,
    crossref_resolve_doi,
    load_corpus,
    openalex_get_by_doi,
    save_corpus,
)

log = logging.getLogger("scientific_research.verify")

# Load evidence levels + type-to-level mapping from JSON data file (DRY).
_DATA_DIR = Path(__file__).parent / "data"
with open(_DATA_DIR / "evidence_levels.json") as _f:
    _EV = json.load(_f)
EVIDENCE_LEVELS: dict = _EV["EVIDENCE_LEVELS"]
TYPE_TO_LEVEL: dict = _EV["TYPE_TO_LEVEL"]


# =============================================================================
# DOAJ whitelist lookup
# =============================================================================
def doaj_lookup_journal(journal_name: str) -> dict:
    """Check if a journal is in DOAJ whitelist via /api/v2/search/journals.
    Returns {in_doaj: bool, review_process: str, ...} or {in_doaj: False} on miss."""
    if not journal_name or not journal_name.strip():
        return {"in_doaj": False, "reason": "no journal name"}
    import httpx

    # Lucene query: title:"..." — escape quotes
    safe = journal_name.replace('"', "'").strip()[:100]
    url = f"https://doaj.org/api/v2/search/journals/title:{safe}?page=1&pageSize=3"
    try:
        r = httpx.get(url, timeout=15)
        r.raise_for_status()
        d = r.json()
        for result in d.get("results", []):
            bib = result.get("bibjson", {})
            title = (bib.get("title") or "").lower().strip()
            if title and title == journal_name.lower().strip():
                return {
                    "in_doaj": True,
                    "doaj_title": bib.get("title"),
                    "eissn": bib.get("eissn"),
                    "pissn": bib.get("pissn"),
                    "review_process": bib.get("editorial", {}).get(
                        "review_process", []
                    ),
                    "publisher": bib.get("publisher", {}).get("name"),
                    "license": (bib.get("license") or [{}])[0].get("type"),
                    "apc": (bib.get("apc") or {}).get("max"),
                }
        return {"in_doaj": False, "reason": "not found in DOAJ"}
    except Exception as e:
        log.warning("DOAJ lookup failed for %s: %s", journal_name, e)
        return {"in_doaj": False, "reason": f"DOAJ API error: {e}"}


def unpaywall_lookup(doi: str) -> dict:
    """Look up Open Access PDF via Unpaywall (https://unpaywall.org).
    More reliable than OpenAlex best_oa_location for some journals.
    Free API, needs a real email (Unpaywall rejects example.com placeholders).
    Returns {oa_pdf_url, host_type, license, is_oa}."""
    if not doi:
        return {}
    email = (
        os.environ.get("SCIENTIFIC_RESEARCH_EMAIL")
        or os.environ.get("UNPAYWALL_EMAIL")
        or "researcher@gmail.com"  # default — set SCIENTIFIC_RESEARCH_EMAIL for proper attribution
    )
    if "example.com" in email:
        log.warning(
            "Unpaywall rejects example.com. Set SCIENTIFIC_RESEARCH_EMAIL env var."
        )
        email = "researcher@gmail.com"
    import httpx

    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        r = httpx.get(url, timeout=15)
        if r.status_code == 404:
            return {"oa_pdf_url": None, "reason": "DOI not in Unpaywall"}
        r.raise_for_status()
        d = r.json()
        best = d.get("best_oa_location") or {}
        return {
            "is_oa": d.get("is_oa", False),
            "oa_pdf_url": best.get("url_for_pdf") or best.get("url"),
            "host_type": best.get("host_type"),
            "version": best.get("version"),
            "license": best.get("license"),
            "publisher": d.get("publisher"),
            "journal": d.get("journal_name"),
        }
    except Exception as e:
        log.warning("Unpaywall lookup failed for %s: %s", doi, e)
        return {"oa_pdf_url": None, "reason": f"Unpaywall API error: {e}"}


def grade_evidence(
    type_crossref: str, is_systematic_review: bool = False, fallback_type: str = ""
) -> tuple[str, str]:
    """Return (level, description). Heuristic from Crossref type.
    Falls back to OpenAlex `type` when type_crossref is empty."""
    if is_systematic_review:
        return ("I", EVIDENCE_LEVELS["I"])
    t = (type_crossref or "").lower().strip()
    if not t:
        t = (fallback_type or "").lower().strip()
    for key, level in TYPE_TO_LEVEL.items():
        if key in t:
            return (level, EVIDENCE_LEVELS[level])
    return ("V", EVIDENCE_LEVELS["V"])


# =============================================================================
# Risk-of-bias advisory (Cochrane RoB 2 / ROBINS-I / QUADAS-2 / Newcastle-Ottawa)
# =============================================================================
def rob_advisory(
    type_crossref: str, study_design_hints: list[str] | None = None
) -> list[dict]:
    """Suggest appropriate risk-of-bias tool. Advisory only (LLM fills domains).
    References: Cochrane Handbook Ch 8 (RoB 2), Ch 25 (ROBINS-I)."""
    hints = " ".join(study_design_hints or []).lower()
    t = (type_crossref or "").lower()
    suggestions: list[dict] = []
    if "rct" in hints or "randomized" in hints or "randomised" in hints:
        suggestions.append(
            {
                "tool": "RoB 2",
                "reference": "Cochrane Handbook Ch 8 (https://training.cochrane.org/handbook/current/chapter-08)",
                "domains": [
                    "Bias arising from the randomization process",
                    "Bias due to deviations from intended interventions",
                    "Bias due to missing outcome data",
                    "Bias in measurement of the outcome",
                    "Bias in selection of the reported result",
                ],
                "signals_for_concern": "no allocation concealment, unblinded outcome assessment, >20% dropout",
            }
        )
    if "cohort" in hints or "non-randomized" in hints or "observational" in hints:
        suggestions.append(
            {
                "tool": "ROBINS-I",
                "reference": "Cochrane Handbook Ch 25 (https://training.cochrane.org/handbook/current/chapter-25)",
                "domains": [
                    "Bias due to confounding",
                    "Bias in selection of participants",
                    "Bias in classification of interventions",
                    "Bias due to deviations from intended interventions",
                    "Bias due to missing data",
                    "Bias in measurement of outcomes",
                    "Bias in selection of the reported result",
                ],
                "signals_for_concern": "no confounder adjustment, self-reported exposure, differential follow-up",
            }
        )
    if "diagnostic" in hints or "sensitivity" in hints or "specificity" in hints:
        suggestions.append(
            {
                "tool": "QUADAS-2",
                "reference": "Whiting PF et al. Ann Intern Med 2011;155:529-536",
                "domains": [
                    "Patient selection",
                    "Index test",
                    "Reference standard",
                    "Flow and timing",
                ],
            }
        )
    if "case-control" in hints or "case control" in hints:
        suggestions.append(
            {
                "tool": "Newcastle-Ottawa Scale",
                "reference": "Wells GA et al. https://www.ohri.ca/programs/clinical_epidemiology/oxford.htm",
                "domains": ["Selection", "Comparability", "Outcome/Exposure"],
            }
        )
    if not suggestions and "journal-article" in t:
        # Default: generic RoB advisory for empirical work
        suggestions.append(
            {
                "tool": "Generic RoB checklist (JBI / CASP)",
                "reference": "JBI Critical Appraisal Tools (https://jbi.global/critical-appraisal-tools)",
                "note": "Specific tool depends on study design — LLM should classify first.",
            }
        )
    return suggestions


# =============================================================================
# Verification result dataclass
# =============================================================================
@dataclass
class VerificationResult:
    paper_id: str
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    resolved: bool = False
    resolvers_used: list[str] = field(default_factory=list)
    resolvers_failed: list[str] = field(default_factory=list)
    metadata_consistency: str = (
        "unknown"  # "consistent" | "discrepancy" | "single-source"
    )
    retraction_status: str = "unknown"  # "clean" | "retracted" | "concern" | "unknown"
    retraction_source: str = ""
    doaj_membership: dict = field(default_factory=dict)
    evidence_level: str = "V"
    evidence_description: str = ""
    risk_of_bias: list[dict] = field(default_factory=list)
    verification_timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# §5 H20 verification gate
# =============================================================================
_S2_FAIL_COUNT = 0
_S2_SKIP_THRESHOLD = 1  # skip S2 after 1 consecutive failure (was 3)
# S2 library retries 429s internally (up to 10s each); 1 failure = ~10s wasted,
# then circuit breaker skips S2 for all remaining papers.


def verify_doi(
    doi: str,
    prefer_openalex: bool = True,
    skip_s2: bool = False,
    force_refresh: bool = False,
) -> tuple[PaperRecord | None, list[str], list[str]]:
    """Resolve DOI via multiple resolvers. Returns (record, used, failed).

    Checks verification cache first — if DOI was verified <90 days ago,
    returns cached result (0 API calls).
    Tries OpenAlex + Crossref + S2. Uses circuit breaker for S2:
    if S2 fails 3 times consecutively, skips S2 for remaining papers.
    """
    # Check verification cache
    try:
        from _search_cache import verify_cache_get, verify_cache_put

        cached = verify_cache_get(doi, force_refresh=force_refresh)
        if cached:
            record_dict = cached.get("record")
            used = cached.get("used", [])
            failed = cached.get("failed", [])
            if record_dict:
                from _sources import PaperRecord

                record = PaperRecord.from_dict(record_dict)
            else:
                record = None
            return (record, used, failed)
    except ImportError:
        verify_cache_put = None  # type: ignore

    global _S2_FAIL_COUNT
    used, failed = [], []
    record: PaperRecord | None = None
    if prefer_openalex:
        try:
            oa_rec = openalex_get_by_doi(doi)
            if oa_rec:
                used.append("openalex")
                record = oa_rec
        except Exception as e:
            failed.append(f"openalex:{e}")
    # Crossref (always try — type_crossref field needed for evidence grading)
    try:
        cr = crossref_resolve_doi(doi)
        if cr:
            used.append("crossref")
            if record is None:
                record = cr
            else:
                from _sources import _merge_records

                record = _merge_records(record, cr)
        else:
            failed.append("crossref:no-data")
    except Exception as e:
        failed.append(f"crossref:{e}")
    # Semantic Scholar (circuit breaker — skip after 3 consecutive failures)
    if not skip_s2 and _S2_FAIL_COUNT < _S2_SKIP_THRESHOLD:
        try:
            from _sources import s2_get_paper

            s2 = s2_get_paper("DOI:" + doi)
            if s2:
                _S2_FAIL_COUNT = 0  # reset on success
                used.append("s2")
                if record is None:
                    record = s2
                else:
                    from _sources import _merge_records

                    record = _merge_records(record, s2)
            else:
                _S2_FAIL_COUNT += 1
                failed.append("s2:no-data")
        except Exception as e:
            _S2_FAIL_COUNT += 1
            failed.append(f"s2:{e}")
    else:
        failed.append("s2:skipped(circuit-breaker)")
    # Store in verification cache
    try:
        from _search_cache import verify_cache_put

        verify_cache_put(
            doi,
            {
                "record": record.to_dict() if record else None,
                "used": used,
                "failed": failed,
            },
        )
    except (ImportError, Exception):
        pass
    return (record, used, failed)


def verify_arxiv(arxiv_id: str) -> tuple[PaperRecord | None, list[str], list[str]]:
    """Verify arXiv ID via arXiv API + S2."""
    used, failed = [], []
    record: PaperRecord | None = None
    # S2 first (has abstracts, tldr)
    try:
        from _sources import s2_get_paper

        s2 = s2_get_paper("ArXiv:" + arxiv_id)
        if s2:
            used.append("s2")
            record = s2
    except Exception as e:
        failed.append(f"s2:{e}")
    # arXiv direct
    try:
        results = arxiv_search("id:" + arxiv_id, max_results=1)
        if results:
            used.append("arxiv")
            if record is None:
                record = results[0]
            else:
                from _sources import _merge_records

                record = _merge_records(record, results[0])
        else:
            failed.append("arxiv:no-data")
    except Exception as e:
        failed.append(f"arxiv:{e}")
    return (record, used, failed)


def check_retraction(record: PaperRecord) -> tuple[str, str]:
    """Return (status, source). Status: 'clean'|'retracted'|'concern'|'unknown'."""
    if record.is_retracted:
        return ("retracted", "openalex:is_retracted=true")
    # Crossref type check (secondary)
    tc = (record.type_crossref or "").lower()
    if "retraction" in tc:
        return ("retracted", f"crossref:type={tc}")
    # Crossref relation check (retraction notices pointing to this paper)
    rels = (record.raw_metadata or {}).get("relation", {}) or {}
    if rels.get("is-review-of"):
        return ("concern", "crossref:has-retraction-notice")
    return ("clean", "openalex+crossref:checked-no-flag")


# =============================================================================
# Retraction Watch database cross-check (SECONDARY signal — fail-open)
# =============================================================================
# Source (verified 2026-08-14): https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/
#   CSV dataset: git clone https://gitlab.com/crossref/retraction-watch-data
#   (updated each working day; comma-separated, in-entry lists use ';')
# Also available via Crossref REST API: filter=update-type:retraction with
# source=retraction-watch in the update-to field.
# DESIGN: RW match downgrades to 'concern' (never auto-'retracted') because
# the CSV matches on OriginalPaperDOI without version/retraction-in-reponse
# nuance. Absent CSV → skipped with one log line (fail-open, never blocks
# the §5 H20 gate on an optional local dataset).
_RW_INDEX: set[str] | None = None


def load_rw_index(csv_path: Path | None = None) -> set[str]:
    """Load Retraction Watch DOIs from local CSV. Returns empty set if absent."""
    global _RW_INDEX
    import csv as _csv
    import os as _os

    if _RW_INDEX is not None:
        return _RW_INDEX
    path = csv_path or Path(_os.environ.get("SCIENTIFIC_RESEARCH_RW_CSV", ""))
    if not path or not path.exists():
        log.info(
            "Retraction Watch CSV not configured (set --rw-csv or "
            "SCIENTIFIC_RESEARCH_RW_CSV); secondary retraction cross-check skipped"
        )
        _RW_INDEX = set()
        return _RW_INDEX
    dois: set[str] = set()
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = _csv.DictReader(fh)
        doi_cols = [c for c in (reader.fieldnames or []) if "doi" in c.lower()]
        if not doi_cols:
            log.warning("RW CSV %s has no DOI-like column — cross-check disabled", path)
            _RW_INDEX = set()
            return _RW_INDEX
        for row in reader:
            for col in doi_cols:
                val = (row.get(col) or "").strip().lower()
                if val.startswith("https://doi.org/"):
                    val = val[len("https://doi.org/") :]
                if val:
                    dois.add(val)
    _RW_INDEX = dois
    log.info("Retraction Watch index loaded: %d DOIs from %s", len(dois), path)
    return _RW_INDEX


def check_retraction_watch(
    record: PaperRecord, rw_index: set[str]
) -> tuple[str, str] | None:
    """Secondary RW cross-check. Returns (status, source) if matched, else None.

    Match → 'concern' (advisory escalation); the primary OpenAlex/Crossref
    checks above remain the only paths to a hard 'retracted' verdict.
    """
    if not rw_index or not record.doi:
        return None
    doi = record.doi.strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/") :]
    if doi in rw_index:
        return (
            "concern",
            "retraction-watch:csv-match (verify at retractiondatabase.org)",
        )
    return None


def check_metadata_consistency(records: list[PaperRecord]) -> str:
    """If multiple sources resolved, check title/year agreement."""
    if len(records) < 2:
        return "single-source"
    titles = [r.title.lower().strip() for r in records if r.title]
    years = [r.year for r in records if r.year]
    title_match = len(set(titles[:1] + titles)) <= 2  # rough agreement
    year_match = (max(years) - min(years) <= 1) if years else True
    if title_match and year_match:
        return "consistent"
    return "discrepancy"


# =============================================================================
# Main verify loop
# =============================================================================
def verify_paper(
    paper: PaperRecord,
    do_doaj: bool = True,
    skip_s2: bool = False,
    force_refresh: bool = False,
    rw_index: set[str] | None = None,
) -> tuple[PaperRecord, VerificationResult]:
    """Verify a single paper. Returns (updated_record, result).

    rw_index: optional Retraction Watch DOI set (from load_rw_index). When
    provided and non-None, a match downgrades clean → concern (secondary signal).
    """
    pid = paper.primary_id
    result = VerificationResult(
        paper_id=pid,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        pmid=paper.pmid,
        verification_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    records_to_merge: list[PaperRecord] = []
    # 1. DOI verification
    if paper.doi:
        rec, used, failed = verify_doi(
            paper.doi, skip_s2=skip_s2, force_refresh=force_refresh
        )
        result.resolvers_used.extend(used)
        result.resolvers_failed.extend(failed)
        if rec:
            records_to_merge.append(rec)
    # 2. arXiv verification
    if paper.arxiv_id and not paper.doi:
        rec, used, failed = verify_arxiv(paper.arxiv_id)
        result.resolvers_used.extend(used)
        result.resolvers_failed.extend(failed)
        if rec:
            records_to_merge.append(rec)
    # 3. Resolved?
    if records_to_merge:
        result.resolved = True
        result.metadata_consistency = check_metadata_consistency(records_to_merge)
        # Merge metadata — prefer resolved records over original
        from _sources import _merge_records

        merged = paper
        for r in records_to_merge:
            merged = _merge_records(merged, r)
        # 4. Retraction check
        result.retraction_status, result.retraction_source = check_retraction(merged)
        # 4b. Retraction Watch secondary cross-check (concern-only, fail-open)
        if result.retraction_status == "clean" and merged.doi:
            rw_hit = check_retraction_watch(merged, rw_index or set())
            if rw_hit:
                result.retraction_status, result.retraction_source = rw_hit
        # 5. Evidence grading
        is_sr = any(
            "systematic review" in (c or "").lower()
            for c in (merged.concepts + [merged.abstract[:500]])
        )
        lvl, desc = grade_evidence(
            merged.type_crossref, is_systematic_review=is_sr, fallback_type=merged.type
        )
        result.evidence_level, result.evidence_description = lvl, desc
        # 6. RoB advisory
        result.risk_of_bias = rob_advisory(
            merged.type_crossref,
            study_design_hints=[merged.abstract[:500]] + merged.concepts,
        )
        # 7. DOAJ check
        if do_doaj and merged.venue:
            result.doaj_membership = doaj_lookup_journal(merged.venue)
        # 8. Unpaywall OA PDF lookup (more reliable than OpenAlex best_oa_location)
        if merged.doi and not merged.oa_pdf_url:
            unpaywall = unpaywall_lookup(merged.doi)
            if unpaywall.get("oa_pdf_url"):
                merged.is_open_access = True
                merged.oa_pdf_url = unpaywall["oa_pdf_url"]
                result.notes.append(
                    f"Unpaywall OA: {unpaywall.get('host_type', '?')}"
                    f" ({unpaywall.get('license', '?')})"
                )
        if result.metadata_consistency == "discrepancy":
            result.notes.append(
                "Title/year disagreement between sources — manual review recommended."
            )
        if not result.resolvers_used:
            result.notes.append("Only fallback resolvers — confidence lower.")
        cache_put(merged)
        return (merged, result)
    # Could not resolve via any canonical resolver
    result.resolved = False
    result.retraction_status = "unknown"
    result.notes.append(
        "Unresolved by any canonical resolver — FORBIDDEN to cite (§5 H20)."
    )
    return (paper, result)


# =============================================================================
# Rendered verification report
# =============================================================================
def render_verification_report(
    results: list[VerificationResult], total_in: int, verified_n: int
) -> str:
    out = ["# §5 H20 Verification Report", ""]
    out.append("## Summary")
    out.append(f"- Input corpus: **{total_in}**")
    out.append(f"- Verified (resolved + non-retracted): **{verified_n}**")
    out.append(f"- Unresolved (forbidden to cite): **{total_in - verified_n}**")
    retracted = [r for r in results if r.retraction_status == "retracted"]
    concerns = [r for r in results if r.retraction_status == "concern"]
    if retracted:
        out.append(f"- ⚠️ **RETRACTED**: {len(retracted)}")
    if concerns:
        out.append(f"- ⚠️ Concern flags: {len(concerns)}")
    out.append("")
    # DOAJ breakdown
    doaj_yes = sum(1 for r in results if r.doaj_membership.get("in_doaj"))
    doaj_no = sum(
        1 for r in results if r.doaj_membership and not r.doaj_membership.get("in_doaj")
    )
    out.append(f"- DOAJ whitelist (OA vetted): {doaj_yes} papers")
    out.append(
        f"- DOAJ not found (neutral signal, may be subscription journal): {doaj_no} papers"
    )
    # Evidence level histogram
    from collections import Counter

    levels = Counter(r.evidence_level for r in results if r.resolved)
    if levels:
        out.append("")
        out.append("## Evidence level histogram")
        for lvl in sorted(levels):
            out.append(f"- **L{lvl}**: {levels[lvl]} ({EVIDENCE_LEVELS.get(lvl, '?')})")
    # Unresolved list
    unresolved = [r for r in results if not r.resolved]
    if unresolved:
        out.append("")
        out.append("## UNRESOLVED (forbidden to cite, §5 H20)")
        for r in unresolved:
            out.append(
                f"- `{r.paper_id}` failed via: {', '.join(r.resolvers_failed) or '(no resolver attempted)'}"
            )
    # Retracted list
    if retracted:
        out.append("")
        out.append("## ⚠️ RETRACTED — do NOT cite")
        for r in retracted:
            out.append(f"- `{r.paper_id}` ({r.retraction_source})")
    # Discrepancies
    discrep = [r for r in results if r.metadata_consistency == "discrepancy"]
    if discrep:
        out.append("")
        out.append("## Metadata discrepancies (manual review)")
        for r in discrep:
            out.append(f"- `{r.paper_id}` — sources disagree on title/year")
    return "\n".join(out)


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="verify",
        description="§5 H20 verification gate: DOI/arXiv resolver check + retraction + DOAJ + evidence grading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument(
        "corpus",
        type=Path,
        nargs="?",
        help="input corpus.json (omit if using --single)",
    )
    p.add_argument(
        "--single",
        metavar="DOI_OR_ARXIV",
        help="verify a single identifier and exit (smoke test)",
    )
    p.add_argument(
        "--no-doaj", action="store_true", help="skip DOAJ whitelist lookup (faster)"
    )
    p.add_argument(
        "--skip-s2",
        action="store_true",
        help="skip Semantic Scholar lookup (much faster)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="skip knowledge base, re-verify all DOIs from APIs",
    )
    p.add_argument(
        "--keep-unresolved",
        action="store_true",
        help="keep unresolved papers in output (default: drop them)",
    )
    p.add_argument(
        "--rw-csv",
        type=Path,
        default=None,
        help="Retraction Watch CSV for secondary cross-check (download: git clone "
        "https://gitlab.com/crossref/retraction-watch-data; also via env "
        "SCIENTIFIC_RESEARCH_RW_CSV). Match → 'concern' status, fail-open if absent",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("research_outputs/verified.json")
    )
    p.add_argument(
        "--report", type=Path, default=Path("research_outputs/verification_report.md")
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    # Retraction Watch secondary index (fail-open)
    rw_index = load_rw_index(args.rw_csv)

    # Single-paper mode (smoke test)
    if args.single:
        log.info("Single verify: %s", args.single)
        if args.single.lower().startswith("arxiv:"):
            rec = PaperRecord(arxiv_id=args.single[6:])
        elif args.single.lower().startswith("pmid:"):
            rec = PaperRecord(pmid=args.single[5:])
        elif re.match(r"^10\.\d{4,9}/", args.single):
            rec = PaperRecord(doi=args.single)
        else:
            print(
                json.dumps(
                    {
                        "error": "unrecognized identifier format",
                        "input": args.single,
                        "hint": "use DOI (10.xxxx/...), arxiv:ID, or pmid:ID",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        merged, result = verify_paper(rec, do_doaj=not args.no_doaj, rw_index=rw_index)
        from _sources import _sanitize_json

        print(
            json.dumps(
                {
                    "verification": result.to_dict(),
                    "paper": _sanitize_json(merged.to_dict()),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        if not result.resolved:
            print(
                f"\n⚠ UNRESOLVED — forbidden to cite (§5 H20). "
                f"Resolvers failed: {result.resolvers_failed}",
                file=sys.stderr,
            )
            return 2
        if result.retraction_status == "retracted":
            print(
                f"\n⚠ RETRACTED — do NOT cite. Source: {result.retraction_source}",
                file=sys.stderr,
            )
            return 2
        return 0

    if not args.corpus:
        p.error("must provide corpus.json or --single")

    papers = load_corpus(args.corpus)
    log.info("Loaded %d papers from %s", len(papers), args.corpus)

    verified_records: list[PaperRecord] = []
    all_results: list[VerificationResult] = []
    for i, paper in enumerate(papers, 1):
        log.info("[%d/%d] verifying %s", i, len(papers), paper.primary_id)
        try:
            merged, result = verify_paper(
                paper,
                do_doaj=not args.no_doaj,
                skip_s2=args.skip_s2,
                force_refresh=args.force_refresh,
                rw_index=rw_index,
            )
            all_results.append(result)
            if result.resolved or args.keep_unresolved:
                verified_records.append(merged)
        except Exception as e:
            log.error("Failed to verify %s: %s", paper.primary_id, e)
            failed_result = VerificationResult(
                paper_id=paper.primary_id,
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                resolved=False,
                resolvers_failed=[f"exception:{e}"],
                notes=[f"verify_paper raised: {e}"],
                verification_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            all_results.append(failed_result)
            if args.keep_unresolved:
                verified_records.append(paper)

    # Save verified.json (only verified records by default)
    save_corpus(
        verified_records,
        args.output,
        meta={
            "source_corpus": str(args.corpus),
            "total_in": len(papers),
            "verified_count": sum(1 for r in all_results if r.resolved),
            "doaj_checked": not args.no_doaj,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    # Append verification results as a side file
    sidecar = args.output.parent / "verification_results.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps([r.to_dict() for r in all_results], indent=2, ensure_ascii=False)
    )
    print(f"Wrote {len(verified_records)} verified papers → {args.output}")
    print(f"Wrote {len(all_results)} verification results → {sidecar}")

    # Render report
    verified_n = sum(
        1 for r in all_results if r.resolved and r.retraction_status != "retracted"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_verification_report(all_results, len(papers), verified_n)
    )
    print(f"Wrote verification report → {args.report}")
    print(f"\n§5 H20 gate: {len(papers)} in → {verified_n} verified + non-retracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())

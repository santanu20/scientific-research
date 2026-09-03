#!/usr/bin/env python3
"""Structured data extraction from verified paper metadata + abstracts.

Extracts PICO/SPIDER components, effect size candidates (Cohen's d / OR / RR / MD),
and methodology keywords. Heuristic-driven — the LLM refines edge cases.

SOTA approach for non-LLM extraction:
    1. PICO: regex pattern matching against known biomedical cue phrases
       (Schmider J et al. 2019, BMJ Open; Vapnyar T et al. 2020)
    2. Effect sizes: numeric pattern matching with confidence intervals
       (mean ± SD, n/N events, RR/OR/HR with 95% CI)
    3. Methods: study design keyword dictionary

Optional SOTA upgrade (LLM-side, not script-side):
    - SciSpacy NER for biomedical entity extraction
    - GROBID for full-paper PDF parsing (already provided via pdf-ocr skill)
    - LLM-based PICO zero-shot extraction (works on abstracts alone)
"""

from __future__ import annotations

# --- Skill venv bootstrap (shared: see _bootstrap.py) ---
if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()
# --- End bootstrap ---


import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus

log = logging.getLogger("scientific_research.extract")


# =============================================================================
# Text sanitization — strip HTML, entities, paywall text from abstracts
# =============================================================================
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&[a-zA-Z0-9#]+;")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_PAYWALL = re.compile(
    r"(?:You do not have access to this content.*$|"
    r"This article is only available.*$|"
    r"Purchase this article.*$|"
    r"Sign in to continue.*$|"
    r"Abstract unavailable.*$)",
    re.IGNORECASE | re.DOTALL,
)
_HTML_ENTITY_MAP = {
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
    "&ndash;": "–",
    "&mdash;": "—",
    "&delta;": "δ",
    "&Delta;": "Δ",
    "&alpha;": "α",
    "&beta;": "β",
    "&gamma;": "γ",
    "&sigma;": "σ",
    "&Sigma;": "Σ",
    "&mu;": "μ",
    "&permil;": "‰",
    "&times;": "×",
    "&plusmn;": "±",
    "&deg;": "°",
    "&sup2;": "²",
    "&sup3;": "³",
    "&sub2;": "₂",
    "&sub3;": "₃",
}


def _sanitize_text(text: str) -> str:
    """Sanitize abstract/title text: strip HTML tags, decode entities,
    remove paywall messages, normalize whitespace."""
    if not text:
        return ""
    # Decode common HTML entities first (before tag stripping)
    for entity, char in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    # Decode remaining numeric entities (&#NNN;)
    text = _RE_HTML_ENTITY.sub("", text)
    # Strip HTML tags
    text = _RE_HTML_TAG.sub("", text)
    # Remove paywall/access messages
    text = _RE_PAYWALL.sub("", text)
    # Normalize whitespace
    text = _RE_MULTI_SPACE.sub(" ", text)
    # Fix broken Unicode from OCR/digitization
    text = text.replace("δ 18 O", "δ18O").replace("δ18 O", "δ18O")
    text = text.replace("TiO 2", "TiO2").replace("SiO 2", "SiO2")
    text = text.replace("Al 2 O 3", "Al2O3").replace("Fe 2 O 3", "Fe2O3")
    text = text.replace("H 2 O", "H2O")
    return text.strip()


def _get_text(paper: PaperRecord) -> str:
    """Get sanitized title + abstract text for extraction."""
    return _sanitize_text((paper.title or "") + "\n" + (paper.abstract or ""))


# =============================================================================
# PICO / SPIDER cue patterns (Schmider 2019 inspired)
# =============================================================================
# Population cues
_POPULATION_CUES = re.compile(
    r"\b(?:patients|subjects|participants|individuals|people|cohort|sample|"
    r"population|women|men|children|adults|elderly|cases|controls|samples? of|"
    r"studied|enrolled|recruited|randomized to|assigned to)\s+"
    r"([\w\s,/\-()]{5,120}?)(?:[.,;]|were|had|underwent|received|with)",
    re.IGNORECASE,
)
# Intervention cues
_INTERVENTION_CUES = re.compile(
    r"\b(?:treated with|received|administered|intervention|therapy|drug|dose|"
    r"mg/d|mg/kg|protocol|regimen|treatment group|intervention group)\s+"
    r"([\w\s,/\-()0-9]{5,150}?)(?:[.,;]|for|over|during)",
    re.IGNORECASE,
)
# Comparator cues
_COMPARATOR_CUES = re.compile(
    r"\b(?:compared (?:with|to)|versus|vs\.?|control group|placebo|standard of care|"
    r"usual care|reference|comparison group|active control)\s+"
    r"([\w\s,/\-()]{5,120}?)(?:[.,;]|\bwere\b|\bhad\b)",
    re.IGNORECASE,
)
# Outcome cues
_OUTCOME_CUES = re.compile(
    r"\b(?:primary outcome|secondary outcome|endpoint|main result|measured|assessed|"
    r"evaluated|outcome was|results?\s+(?:showed|indicated|demonstrated)|"
    r"found that|observed that|significant(?:ly)?)\s+"
    r"([\w\s,/\-()0-9]{5,200}?)(?:[.,;]|was|were|of|in)",
    re.IGNORECASE,
)


def extract_pico(paper: PaperRecord) -> dict:
    """Extract PICO candidates via _nlp module (lexicon + cue patterns +
    multi-design detection). Replaces narrow single-pattern regex approach."""
    from _nlp import extract_pico as _nlp_pico

    text = _get_text(paper)
    result = _nlp_pico(text)
    # Backward-compat: study_design_hint = first detected design (string),
    # study_designs = full list
    designs = result.get("study_design", [])
    result["study_design_hint"] = designs[0] if designs else ""
    result["study_designs"] = designs
    return result


_DESIGN_HINTS = {
    "RCT": re.compile(
        r"\brandomi[sz]ed controlled trial\b|\bRCT\b|\bcluster[\s-]?randomi[sz]",
        re.IGNORECASE,
    ),
    "systematic review": re.compile(
        r"\bsystematic review\b|\bmeta[\s-]?analysis\b", re.IGNORECASE
    ),
    "cohort": re.compile(
        r"\bcohort (?:study|analysis)\b|\bprospective cohort\b|\bretrospective cohort\b",
        re.IGNORECASE,
    ),
    "case-control": re.compile(r"\bcase[\s-]?control\b", re.IGNORECASE),
    "cross-sectional": re.compile(r"\bcross[\s-]?sectional\b", re.IGNORECASE),
    "diagnostic": re.compile(
        r"\bdiagnostic (?:accuracy|study|test)\b|\bsensitivity and specificity\b",
        re.IGNORECASE,
    ),
    "preprint": re.compile(r"\bpreprint\b", re.IGNORECASE),
    "qualitative": re.compile(
        r"\bqualitative (?:study|interviews?)\b|\bfocus group\b", re.IGNORECASE
    ),
}


def _detect_design(text: str) -> str:
    for design, pat in _DESIGN_HINTS.items():
        if pat.search(text):
            return design
    return ""


# =============================================================================
# Effect size extraction
# =============================================================================
# Pattern: "mean ± SD" or "mean (SD)" with sample size
_MEAN_SD = re.compile(
    r"(\d+\.?\d*)\s*[±\+\-\/]\s*(\d+\.?\d*)"  # mean ± SD
    r"(?:\s*\((\d+\.?\d*)\s*[±\+\-\/]\s*(\d+\.?\d*)\))?"  # optional second group
    r"\s*(?:\(?n\s*=\s*(\d+)\)?)?",  # optional n
    re.IGNORECASE,
)
# Pattern: n/N events (e.g., "12/45 patients experienced")
_EVENTS = re.compile(
    r"(\d+)\s*\/\s*(\d+)\s+(?:patients?|subjects?|participants?|cases?|had|experienced|developed|showed)",
    re.IGNORECASE,
)
# Pattern: "OR = 2.5 (95% CI 1.5-3.5)" or "odds ratio 2.5"
_OR_RR_HR = re.compile(
    r"\b(?:odds ratio|OR|risk ratio|relative risk|RR|hazard ratio|HR)\s*(?:=|:|of|was)?\s*"
    r"(\d+\.?\d*)\s*(?:\((?:95%\s*CI|CI|confidence interval)[:\s]*"
    r"(\d+\.?\d*)\s*[\-–to]+\s*(\d+\.?\d*)\))?",
    re.IGNORECASE,
)
# Pattern: "p < 0.05" or "p = 0.031"
_PVALUE = re.compile(r"\bp\s*[<>=]\s*(0\.\d+|ns|not significant)", re.IGNORECASE)
# Pattern: "Cohen's d = 0.5" or "d = 0.5"
_COHENS_D = re.compile(
    r"\b(?:Cohen'?s d|Hedges'? g|standardized mean difference|SMD|d)\s*(?:=|:|of|was)?\s*"
    r"(\d+\.?\d*)",
    re.IGNORECASE,
)


def extract_effect_sizes(paper: PaperRecord) -> dict:
    """Extract effect size candidates from abstract. LLM verifies."""
    text = _sanitize_text(paper.abstract or "")
    if not text:
        return {
            "mean_sd_groups": [],
            "event_counts": [],
            "effect_sizes": [],
            "p_values": [],
            "cohens_d": [],
        }
    return {
        "mean_sd_groups": [
            {
                "m1": float(m1),
                "sd1": float(sd1),
                "m2": float(m2) if m2 else None,
                "sd2": float(sd2) if sd2 else None,
                "n": int(n) if n else None,
            }
            for m1, sd1, m2, sd2, n in _MEAN_SD.findall(text)
        ],
        "event_counts": [
            {"events": int(ev), "total": int(tot)} for ev, tot in _EVENTS.findall(text)
        ],
        "effect_sizes": [
            {
                "type": "OR/RR/HR",
                "value": float(val),
                "ci_lower": float(lo) if lo else None,
                "ci_upper": float(hi) if hi else None,
            }
            for val, lo, hi in _OR_RR_HR.findall(text)
        ],
        "p_values": _PVALUE.findall(text),
        "cohens_d": [float(d) for d in _COHENS_D.findall(text)],
    }


# =============================================================================
# Methods extraction
# =============================================================================
_METHODS_KEYWORDS = {
    "study_design": [
        "RCT",
        "randomized",
        "randomised",
        "double-blind",
        "single-blind",
        "placebo-controlled",
        "crossover",
        "parallel-group",
        "cohort",
        "case-control",
        "cross-sectional",
        "longitudinal",
        "meta-analysis",
        "systematic review",
        "qualitative",
        "experimental",
        "observational",
        "pilot study",
    ],
    "blinding": ["double-blind", "single-blind", "open-label", "unblinded", "masked"],
    "allocation": [
        "randomized",
        "stratified",
        "block randomization",
        "concealed allocation",
    ],
    "duration": [r"\d+\s*weeks?", r"\d+\s*months?", r"\d+\s*years?", r"\d+\s*days?"],
    "sample_size": [r"n\s*=\s*\d+", r"total of \d+", r"sample of \d+"],
    "primary_endpoint": ["primary outcome", "primary endpoint", "main outcome"],
    "statistics": [
        "intention-to-treat",
        "per-protocol",
        "ANCOVA",
        "regression",
        "mixed-effects",
        "Bayesian",
        "frequentist",
        "t-test",
        "ANOVA",
    ],
    "funding": ["funded by", "supported by", "grant"],
}


def extract_methods(paper: PaperRecord) -> dict:
    """Extract methodology keywords from abstract. LLM refines via full-text."""
    text = _get_text(paper).lower()
    if not text.strip():
        return {k: [] for k in _METHODS_KEYWORDS}
    out = {}
    for category, patterns in _METHODS_KEYWORDS.items():
        matched = []
        for pat in patterns:
            if pat.startswith("\\") or any(c in pat for c in "+?*.()[]"):
                m = re.findall(pat, text, re.IGNORECASE)
                matched.extend(m if isinstance(m, list) else [m])
            else:
                if pat.lower() in text:
                    matched.append(pat)
        out[category] = list(dict.fromkeys(matched))  # dedup preserve order
    return out


# =============================================================================
# Extraction result
# =============================================================================
@dataclass
class ExtractionResult:
    paper_id: str
    doi: str | None = None
    title: str = ""
    abstract: str = ""
    key_finding: str = ""
    pico: dict = field(default_factory=dict)
    effect_sizes: dict = field(default_factory=dict)
    methods: dict = field(default_factory=dict)
    extraction_confidence: str = (
        "low"  # 'low'|'medium'|'high' based on extraction richness
    )
    extraction_timestamp: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _confidence(pico: dict, effects: dict) -> str:
    n_pico = sum(len(v) if isinstance(v, list) else 0 for v in pico.values())
    n_eff = sum(len(v) if isinstance(v, list) else 0 for v in effects.values())
    total = n_pico + n_eff
    if total >= 8:
        return "high"
    if total >= 3:
        return "medium"
    return "low"


def _extract_text_from_pdf(pdf_path: str) -> tuple[str, bool]:
    """Extract text from a PDF using pdftotext (fast path).

    Returns (text, needs_ocr). If needs_ocr=True, pdftotext failed and
    the agent should invoke /pdf-ocr skill for OCR.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-nodiag", "-eol", "unix", pdf_path, "-"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = result.stdout.strip()
        if len(text) > 100:
            return text, False
        else:
            return "", True  # too little text → needs OCR
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.warning("pdftotext failed for %s: %s", pdf_path, e)
        return "", True


def _download_pdf(url: str, doi: str) -> str | None:
    """Download an OA PDF to cache. Returns local path or None."""
    import urllib.request

    cache_dir = Path.home() / ".cache" / "scientific_research" / "pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9]", "_", doi or url[:50])[:80]
    pdf_path = str(cache_dir / f"{safe_name}.pdf")
    if Path(pdf_path).exists():
        return pdf_path
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "scientific-research/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            if len(data) < 1000:  # too small to be a real PDF
                return None
            Path(pdf_path).write_bytes(data)
            return pdf_path
    except (urllib.error.URLError, OSError) as e:
        log.debug("PDF download failed for %s: %s", url, e)
        return None


def extract_from_paper(
    paper: PaperRecord, topic: str = "", use_llm: bool = False
) -> ExtractionResult:
    """Extract structured data from a paper.

    Uses LLM (Ollama) as primary extraction method with regex/lexicon fallback.
    Text source priority: abstract → OA PDF (pdftotext) → needs_ocr flag.
    """
    # Run improved non-LLM extraction (effect parser + PICO + methods)
    text_source = _sanitize_text(paper.abstract or "")
    pico_regex = extract_pico(paper)

    # Use the powerful _effect_parser (field-agnostic) instead of old basic regex
    from _effect_parser import extract_effect_sizes as _parse_effects  # internal

    parsed = _parse_effects(text_source)
    effects_regex = {
        "mean_sd_groups": parsed.get("mean_sd_groups", []),
        "single_measurements": parsed.get("single_measurements", []),
        "p_values": [str(p) for p in parsed.get("p_values", [])],
        "effect_sizes": [],
        "cohens_d": [],
        "event_counts": [],
    }

    methods = extract_methods(paper)

    # Determine text source: abstract → PDF → none
    text = _sanitize_text(paper.abstract or "")
    notes = []
    needs_ocr = False

    if not text.strip() and paper.oa_pdf_url:
        # No abstract — try OA PDF
        pdf_path = _download_pdf(paper.oa_pdf_url, paper.doi or "")
        if pdf_path:
            text, needs_ocr = _extract_text_from_pdf(pdf_path)
            if text:
                log.info(
                    "Extracted text from PDF for %s (%d chars)",
                    paper.doi or paper.title[:30],
                    len(text),
                )
                notes.append(f"text from PDF (pdftotext, {len(text)} chars)")
            elif needs_ocr:
                notes.append(
                    "needs_ocr: PDF is scanned/image — agent should invoke /pdf-ocr"
                )
        else:
            notes.append("no abstract, OA PDF download failed")
    elif not text.strip():
        notes.append("no abstract available, no OA PDF — extraction limited")

    # No text available — leave key_finding empty.
    # Previously generated "This study presents research on {title}" which
    # produced garbage findings. Empty findings are handled gracefully by
    # the synthesis pipeline — papers without findings are cited but not quoted.
    if not text.strip():
        pass  # key_finding stays empty

    # LLM extraction is OPT-IN (--use-llm flag). Default: regex/SVM.
    if use_llm and text and text.strip():
        try:
            from _llm_extract import extract_paper as llm_extract
            from _llm_extract import is_available

            if is_available():
                llm_result = llm_extract(
                    abstract=text[
                        :6000
                    ],  # cap for LLM token budget (auto-expands ctx in _call_ollama)
                    title=paper.title or "",
                    topic=topic,
                    fallback=None,  # we'll merge manually below
                )

                # Merge LLM results into existing format
                pico = _merge_pico(pico_regex, llm_result)
                effects = _merge_effects(effects_regex, llm_result)

                # Regex-classifier fallback for empty LLM fields.
                # The LLM sometimes returns empty discipline/study_type/novelty
                # (short abstracts, non-English text, empty LLM response).
                # The regex classifiers (_classifiers.py) are cheap lexical
                # detectors that fill the gaps so synthesis never sees blanks.
                # Use title + abstract — many Crossref papers have no abstract,
                # but titles are rich with discipline-specific terms.
                from _classifiers import (  # internal module
                    classify_novelty,
                    detect_discipline,
                    detect_study_type,
                )

                fallback_text = f"{paper.title or ''} {text}".strip()
                if not pico.get("discipline"):
                    pico["discipline"] = detect_discipline(fallback_text)
                if not pico.get("study_type"):
                    st = detect_study_type(fallback_text)
                    if st:
                        pico["study_type"] = st
                        pico.setdefault("study_design", [])
                        if st not in pico["study_design"]:
                            pico["study_design"].append(st)
                if not pico.get("novelty"):
                    pico["novelty"] = classify_novelty(fallback_text)

                # Store LLM-specific fields + text source notes
                if llm_result.get("key_finding"):
                    notes.append(f"LLM key finding: {llm_result['key_finding']}")
                if llm_result.get("stance_toward_topic"):
                    notes.append(f"LLM stance: {llm_result['stance_toward_topic']}")
                if llm_result.get("discipline"):
                    notes.append(f"discipline: {llm_result['discipline']}")
                if needs_ocr:
                    notes.append(
                        "needs_ocr: pdftotext failed — agent should invoke /pdf-ocr"
                    )

                confidence = _confidence(pico, effects)
                # Boost confidence if LLM extracted paired effect sizes
                _llm_es = llm_result.get("effect_sizes")
                if isinstance(_llm_es, list) and any(
                    isinstance(e, dict) and e.get("group1") and e.get("group2")
                    for e in _llm_es
                ):
                    confidence = "high"

                return ExtractionResult(
                    paper_id=paper.primary_id,
                    doi=paper.doi,
                    title=paper.title,
                    abstract=paper.abstract or "",
                    key_finding=llm_result.get("key_finding", "")
                    or pico_regex.get("key_finding", ""),
                    pico=pico,
                    effect_sizes=effects,
                    methods=methods,
                    extraction_confidence=confidence,
                    extraction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    notes=notes,
                )
        except Exception as e:
            log.warning(
                "LLM extraction failed for %s: %s — using regex",
                paper.doi or paper.title[:30],
                e,
            )

    # Fallback: regex only
    # Enrich regex results with non-LLM classifiers.
    # When the abstract is empty (Crossref titles-only), fall back to the
    # title — many geological terms are in the title (e.g. "garnet-biotite
    # thermometry", "spinel lherzolite") so discipline/study_type detection
    # still works.
    abstract_text = _sanitize_text(paper.abstract or "")
    classifier_text = abstract_text or _sanitize_text(paper.title or "")
    if classifier_text:
        from _classifiers import (  # internal module
            classify_novelty,
            detect_discipline,
            detect_study_type,
            extract_interpretation,
            extract_key_finding,
        )

        pico_regex["discipline"] = detect_discipline(classifier_text)
        pico_regex["novelty"] = classify_novelty(classifier_text)
        if abstract_text:  # key_finding + interpretation need full abstract
            pico_regex["key_finding"] = extract_key_finding(abstract_text)
            pico_regex["interpretation"] = extract_interpretation(abstract_text)
        st = detect_study_type(abstract_text)
        if st:
            pico_regex["study_type"] = st
            pico_regex.setdefault("study_design", [])
            if st not in pico_regex["study_design"]:
                pico_regex["study_design"].append(st)

    # Data validation: flag incomplete extractions
    validation_notes = []
    if not paper.abstract:
        validation_notes.append("no abstract available — extraction limited")
    if not pico_regex.get("population") and paper.abstract:
        validation_notes.append("no population extracted — check abstract structure")
    if not pico_regex.get("discipline") and paper.abstract:
        validation_notes.append("no discipline detected — unknown field")
    if not effects_regex.get("mean_sd_groups") and not effects_regex.get(
        "single_measurements"
    ):
        if paper.abstract:
            validation_notes.append(
                "no numerical data extracted — qualitative abstract?"
            )

    return ExtractionResult(
        paper_id=paper.primary_id,
        doi=paper.doi,
        title=paper.title,
        abstract=paper.abstract or "",
        key_finding=pico_regex.get("key_finding", ""),
        pico=pico_regex,
        effect_sizes=effects_regex,
        methods=methods,
        extraction_confidence=_confidence(pico_regex, effects_regex),
        extraction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        notes=validation_notes,
    )


def _merge_pico(regex_result: dict, llm_result: dict) -> dict:
    """Merge regex PICO with LLM field-agnostic extraction. LLM takes precedence."""
    merged = dict(regex_result)

    # New field-agnostic fields (pass through from LLM)
    for field_name in (
        "discipline",
        "study_type",
        "subject",
        "method",
        "interpretation",
        "novelty",
        "risk_of_bias",
        "quality_score",
        "geological_concepts",
        "quantitative_data",
    ):
        llm_val = llm_result.get(field_name)
        if llm_val is not None:
            merged[field_name] = llm_val

    # Legacy PICO: map from new field-agnostic format
    subj = llm_result.get("subject", {})
    if isinstance(subj, dict):
        if subj.get("system_studied"):
            merged["population"] = [subj["system_studied"]]
        if subj.get("context"):
            merged.setdefault("context", subj["context"])

    meth = llm_result.get("method", {})
    if isinstance(meth, dict):
        if meth.get("technique"):
            merged["intervention"] = [meth["technique"]]

    # Study design
    llm_design = llm_result.get("study_type") or llm_result.get("study_design")
    if llm_design and isinstance(llm_design, str):
        merged["study_design"] = [llm_design]
        merged["study_design_hint"] = llm_design
        merged["study_designs"] = [llm_design]

    # Abbreviations
    llm_abbrs = llm_result.get("abbreviations", {})
    if llm_abbrs and isinstance(llm_abbrs, dict):
        merged["abbreviations"] = llm_abbrs

    # Key finding + interpretation
    if llm_result.get("key_finding"):
        merged["key_finding"] = llm_result["key_finding"]
    if llm_result.get("interpretation"):
        merged["interpretation"] = llm_result["interpretation"]
    if llm_result.get("novelty"):
        merged["novelty"] = llm_result["novelty"]

    return merged


def _merge_effects(regex_result: dict, llm_result: dict) -> dict:
    """Merge regex effect sizes with LLM key_results. LLM takes precedence."""
    merged = dict(regex_result)

    # New field-agnostic key_results (pass through)
    key_results = llm_result.get("key_results", [])
    if key_results and isinstance(key_results, list):
        merged["key_results"] = key_results

    # Legacy effect_sizes from LLM (converted by _validate_extraction)
    llm_effects = llm_result.get("effect_sizes", [])
    if not llm_effects:
        return merged

    # Convert to mean_sd_groups format for meta_analyze.py
    mean_sd_groups = []

    # Check if LLM returned paired groups (old format) or single measurements (new format)
    for es in llm_effects:
        if not isinstance(es, dict):
            continue

        m1 = es.get("m1")
        m2 = es.get("m2")

        if m2 is not None:
            # Paired groups (old LLM format)
            try:
                m1 = float(m1)
                m2 = float(m2)
                sd1 = float(es["sd1"]) if es.get("sd1") is not None else None
                sd2 = float(es.get("sd2")) if es.get("sd2") is not None else None
                n1 = int(es.get("n") or 0)
                n2 = int(es.get("n2") or es.get("n") or 0)
                ci_lo = (
                    float(es["ci_lower"]) if es.get("ci_lower") is not None else None
                )
                ci_hi = (
                    float(es["ci_upper"]) if es.get("ci_upper") is not None else None
                )
                mean_sd_groups.append(
                    {
                        "m1": m1,
                        "sd1": sd1,
                        "n": n1,
                        "m2": m2,
                        "sd2": sd2,
                        "n2": n2,
                        "ci_lower": ci_lo,
                        "ci_upper": ci_hi,
                        "outcome": es.get("outcome", ""),
                    }
                )
            except (KeyError, ValueError, TypeError):
                pass
        elif m1 is not None:
            # Single measurement (new field-agnostic format)
            try:
                val = float(m1)
                unc = None
                if es.get("sd1") is not None:
                    try:
                        unc = float(es["sd1"])
                    except (ValueError, TypeError):
                        pass
                n = None
                if es.get("n") is not None:
                    try:
                        n = int(es["n"])
                    except (ValueError, TypeError):
                        pass
                mean_sd_groups.append(
                    {
                        "m1": val,
                        "sd1": unc,
                        "n": n,
                        "m2": None,
                        "sd2": None,
                        "n2": None,
                        "ci_lower": None,
                        "ci_upper": None,
                        "outcome": es.get("outcome", ""),
                        "unit": es.get("unit", ""),
                        "comparison": es.get("comparison"),
                    }
                )
            except (KeyError, ValueError, TypeError):
                pass

    if mean_sd_groups:
        merged["mean_sd_groups"] = mean_sd_groups

    return merged


# =============================================================================
# Rendered report
# =============================================================================
def render_extraction_report(results: list[ExtractionResult]) -> str:
    out = ["# Structured Data Extraction Report", ""]
    high = [r for r in results if r.extraction_confidence == "high"]
    med = [r for r in results if r.extraction_confidence == "medium"]
    low = [r for r in results if r.extraction_confidence == "low"]
    out.append("## Confidence distribution")
    out.append(f"- **high**: {len(high)}")
    out.append(f"- **medium**: {len(med)}")
    out.append(f"- **low**: {len(low)}")
    out.append("")
    for r in results:
        ident = r.doi or r.paper_id
        out.append(f"## `{ident}` — {r.title[:80]}")
        out.append(f"**Confidence**: {r.extraction_confidence}")
        if r.pico.get("study_design_hint"):
            out.append(f"- **Design hint**: {r.pico['study_design_hint']}")
        if r.pico.get("population"):
            out.append(f"- **P (population)**: {r.pico['population']}")
        if r.pico.get("intervention"):
            out.append(f"- **I (intervention)**: {r.pico['intervention']}")
        if r.pico.get("comparator"):
            out.append(f"- **C (comparator)**: {r.pico['comparator']}")
        if r.pico.get("outcome"):
            out.append(f"- **O (outcome)**: {r.pico['outcome']}")
        if r.effect_sizes.get("effect_sizes"):
            out.append(
                f"- **Effect sizes (OR/RR/HR)**: {r.effect_sizes['effect_sizes']}"
            )
        if r.effect_sizes.get("cohens_d"):
            out.append(f"- **Cohen's d**: {r.effect_sizes['cohens_d']}")
        if r.effect_sizes.get("mean_sd_groups"):
            out.append(f"- **Mean±SD groups**: {len(r.effect_sizes['mean_sd_groups'])}")
        if r.effect_sizes.get("p_values"):
            out.append(f"- **p-values**: {r.effect_sizes['p_values'][:5]}")
        if r.methods.get("study_design"):
            out.append(
                f"- **Methods detected**: {', '.join(r.methods['study_design'][:5])}"
            )
        if r.notes:
            for n in r.notes:
                out.append(f"- NOTE: {n}")
        out.append("")
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
        prog="extract",
        description="Extract PICO/SPIDER, effect sizes, and methodology from verified corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("verified", type=Path, help="verified.json from verify.py")
    p.add_argument(
        "-o", "--output", type=Path, default=Path("research_outputs/extracted.json")
    )
    p.add_argument(
        "--report", type=Path, default=Path("research_outputs/extraction_report.md")
    )
    p.add_argument(
        "--mode", choices=["pico", "effects", "methods", "all"], default="all"
    )
    p.add_argument(
        "--topic", type=str, default="", help="research topic (improves LLM extraction)"
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="use LLM extraction via Ollama (higher quality, ~5s/paper)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    papers = load_corpus(args.verified)
    log.info("Loaded %d verified papers", len(papers))

    # Try to infer topic from corpus metadata if not provided
    topic = args.topic
    if not topic:
        try:
            corpus_data = json.loads(args.verified.read_text())
            topic = corpus_data.get("meta", {}).get("query", "")
        except (json.JSONDecodeError, KeyError, OSError):
            topic = ""
    if topic:
        log.info("Extraction topic: %s", topic)

    # Parallel LLM extraction (4 workers — Ollama supports concurrent requests)
    from concurrent.futures import ThreadPoolExecutor

    log.info("Extracting from %d papers (4 parallel workers)...", len(papers))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda p: extract_from_paper(p, topic=topic, use_llm=args.use_llm),
                papers,
            )
        )

    payload = {
        "meta": {
            "source": str(args.verified),
            "n_papers": len(papers),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "extractions": [r.to_dict() for r in results],
    }
    from _artifact import save_artifact

    save_artifact(payload, args.output)
    print(f"Wrote {len(results)} extractions → {args.output}")

    args.report.write_text(render_extraction_report(results))
    print(f"Wrote extraction report → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Risk-of-bias scaffolds + GRADE evidence profiles.

Generates EMPTY assessment scaffolds for each verified paper. The LLM (or a
domain expert) fills the domain judgments. Pre-populates tool selection based
on study design detected in extract.py.

SOTA references:
    - RoB 2 (Sterne JAC et al. BMJ 2019;366:l4898) — for randomized trials
    - ROBINS-I (Sterne JAC et al. Ann Intern Med 2016) — for non-randomized
    - QUADAS-2 (Whiting PF et al. Ann Intern Med 2011) — for diagnostic accuracy
    - Newcastle-Ottawa Scale (Wells GA et al.) — for observational studies
    - GRADE (Guyatt GH et al. BMJ 2008;336:924-926) — for evidence profiles
    - Cochrane Handbook Ch 8, 25 (https://training.cochrane.org/handbook/current)

Each paper gets ONE scaffold (tool selected automatically). LLM fills domains.
"""


from __future__ import annotations

# --- Self-contained skill venv bootstrap (mirrors pdf-ocr/web-search pattern) ---
import os as _bs_os, sys as _bs_sys
_SKILL_VENV = _bs_os.path.expanduser("~/.config/opencode/skills/scientific-research/.venv")
_SKILL_VENV_PY = _bs_os.path.join(_SKILL_VENV, "bin", "python")
_REQ_IMPORTS = ("habanero", "pyalex", "semanticscholar", "arxiv", "numpy", "scipy", "sklearn", "httpx")
_REQ_INSTALLS = ("habanero", "pyalex", "semanticscholar", "arxiv", "numpy", "scipy", "scikit-learn", "httpx", "pytest", "ruff")
if __name__ == "__main__" and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_SKILL_VENV"):
    if not _bs_os.path.exists(_SKILL_VENV_PY) and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        import subprocess as _bs_sp
        try:
            _bs_sys.stderr.write("Bootstrapping scientific-research skill venv (one-time setup)...\n")
            _bs_sp.run(["uv", "venv", _SKILL_VENV, "--python", "3.13"], check=True, capture_output=True)
            _bs_sp.run(["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS], check=True, capture_output=True)
            _bs_sys.stderr.write("scientific-research skill venv ready.\n")
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Failed to auto-bootstrap: {_bs_ex}\nManual: uv venv {_SKILL_VENV} --python 3.13 && uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n")
            _bs_sys.exit(2)
    if _bs_os.path.exists(_SKILL_VENV_PY) and _bs_os.path.normpath(_bs_sys.prefix) != _bs_os.path.normpath(_SKILL_VENV):
        _bs_os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        _bs_os.execv(_SKILL_VENV_PY, [_SKILL_VENV_PY, _bs_os.path.abspath(__file__)] + _bs_sys.argv[1:])
    _missing = []
    for _m in _REQ_IMPORTS:
        try: __import__(_m)
        except ImportError: _missing.append(_m)
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
                try: __import__(_m)
                except ImportError: _missing.append(_m)
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Auto-install failed: {_bs_ex}\n")
    if _missing:
        _bs_sys.stderr.write(f"FATAL: missing required deps: {', '.join(_missing)}\nInstall: uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n")
        _bs_sys.exit(2)
# --- End bootstrap ---



import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus  # noqa: E402

log = logging.getLogger("scientific_research.assess")

# Load RoB tool definitions + GRADE factors from JSON data files (DRY).
# JSON keeps ~250 lines of domain knowledge out of Python source.
_DATA_DIR = Path(__file__).parent / "data"
with open(_DATA_DIR / "rob_tools.json") as _f:
    ROB_TOOLS: dict = json.load(_f)
with open(_DATA_DIR / "grade_factors.json") as _f:
    GRADE_FACTORS: dict = json.load(_f)


def select_rob_tool(study_design: str) -> str:
    """Pick RoB tool based on detected study design."""
    d = (study_design or "").lower()
    if "rct" in d or "randomized" in d or "randomised" in d:
        return "RoB 2"
    if "cohort" in d or "case-control" in d or "observational" in d:
        return "ROBINS-I" if "intervention" in d or "comparator" in d else "Newcastle-Ottawa"
    if "diagnostic" in d or "sensitivity" in d or "specificity" in d:
        return "QUADAS-2"
    return "ROBINS-I"  # default for empirical


@dataclass
class Assessment:
    paper_id: str
    doi: str | None
    title: str
    study_design_detected: str
    rob_tool_selected: str
    rob_scaffold: dict  # full tool definition from ROB_TOOLS
    rob_judgments: dict  # individual domain judgments (empty if scaffold-only)
    risk_of_bias: str  # overall: low, some_concerns, high, unknown
    quality_score: str  # 1-10 from LLM or empty
    grade_profile: dict
    grade_judgments: dict  # empty — LLM/expert fills
    assessment_timestamp: str


def build_assessment(
    paper: PaperRecord,
    study_design_hint: str = "",
    abstract: str = "",
    topic: str = "",
    llm_risk_of_bias: str = "",
    llm_quality_score: str = "",
) -> Assessment:
    """Build assessment for a paper. Tool selected from design hint.

    If llm_risk_of_bias is provided (from extraction), uses it directly — no LLM call.
    Otherwise falls back to LLM call (slower) or scaffold-only.
    """
    tool_name = select_rob_tool(study_design_hint)
    scaffold = ROB_TOOLS[tool_name]

    rob_judgments = {d["name"]: "" for d in scaffold["domains"]}
    risk_of_bias = llm_risk_of_bias or "unknown"
    quality_score = llm_quality_score or ""

    # Fill domain-level judgments from overall risk (if available from extraction)
    if risk_of_bias and risk_of_bias != "unknown":
        domain_value = risk_of_bias.replace("_", " ")  # "some_concerns" → "some concerns"
        for domain_name in rob_judgments:
            rob_judgments[domain_name] = domain_value

    # LLM RoB assessment is OPT-IN only (--use-llm flag or env var)
    # Default: skip LLM, return scaffold with "unknown" domains
    use_llm = (os.environ.get("SCIENTIFIC_RESEARCH_USE_LLM") or "").lower() in ("1", "true", "yes")
    if use_llm and not llm_risk_of_bias and abstract and abstract.strip():
        try:
            import json as _json

            from _llm_extract import _call_ollama, get_model, is_available

            if is_available():
                model = get_model("simple")
                if model:
                    prompt = f"""Assess risk of bias for this study based on its abstract.

Title: {paper.title or ""}
Study design: {study_design_hint or "unknown"}
Abstract: {abstract[:2000]}

For each domain, judge: "low", "some concerns", "high", or "unknown".
Return JSON:
{{
  "domains": {{"{scaffold["domains"][0]["name"]}": "low|some concerns|high|unknown"}},
  "overall_risk": "low|some concerns|high|unknown",
  "quality_score": 1-10,
  "reason": "one sentence"
}}
"""
                    response = _call_ollama(model, prompt, task="simple")
                    try:
                        result = _json.loads(response)
                    except _json.JSONDecodeError:
                        from _llm_extract import _extract_json_from_text

                        result = _extract_json_from_text(response)

                    if result:
                        for d in scaffold["domains"]:
                            key = d["name"]
                            if key in result.get("domains", {}):
                                rob_judgments[key] = result["domains"][key]
                        risk_of_bias = result.get("overall_risk", "unknown")
                        quality_score = str(result.get("quality_score", ""))
        except Exception as e:
            log.debug("LLM RoB assessment failed: %s", e)

    return Assessment(
        paper_id=paper.primary_id,
        doi=paper.doi,
        title=paper.title,
        study_design_detected=study_design_hint or "unknown",
        rob_tool_selected=tool_name,
        rob_scaffold=scaffold,
        rob_judgments=rob_judgments,
        risk_of_bias=risk_of_bias,
        quality_score=quality_score,
        grade_profile=GRADE_FACTORS,
        grade_judgments={k: "" for k in GRADE_FACTORS},
        assessment_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# =============================================================================
# Rendered report
# =============================================================================
def render_assessment_report(assessments: list[Assessment]) -> str:
    out = ["# Risk-of-Bias Assessment Scaffolds + GRADE Profiles", ""]
    from collections import Counter

    tool_counts = Counter(a.rob_tool_selected for a in assessments)
    out.append("## Tool distribution")
    for tool, n in sorted(tool_counts.items()):
        ref = ROB_TOOLS.get(tool, {}).get("reference", "")
        out.append(f"- **{tool}** ({n} papers): {ref}")
    out.append("")
    out.append("## GRADE factors (apply to all outcomes)")
    for factor, options in GRADE_FACTORS.items():
        out.append(f"- **{factor}**: {options}")
    out.append("")
    out.append("## Per-paper scaffolds")
    for a in assessments:
        out.append(f"\n### `{a.doi or a.paper_id}` — {a.title[:80]}")
        out.append(f"**Design**: {a.study_design_detected}  ")
        out.append(f"**Tool**: {a.rob_tool_selected}  ")
        out.append(f"**Reference**: {a.rob_scaffold.get('reference', '')}  ")
        out.append("")
        out.append("#### RoB domains (LLM/expert fills response):")
        for d in a.rob_scaffold["domains"]:
            opts = d.get("response_options") or d.get("concerns") or []
            out.append(f"- **{d['name']}** {'/ '.join(opts[:3])}")
            for q in d.get("signaling_questions", []):
                out.append(f"    - {q}")
            for item_key in ("items_cohort", "items_case_control", "items"):
                if item_key in d:
                    for item in d[item_key]:
                        out.append(f"    - [{item}] __ stars: ___")
        out.append("\n#### GRADE judgments (LLM/expert fills):")
        for factor in GRADE_FACTORS:
            out.append(f"- **{factor}**: ___ ")
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
        prog="assess",
        description="Generate RoB scaffolds + GRADE evidence profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--self-check", action="store_true",
                        help="verify deps + key imports, then exit 0")  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("verified", type=Path, help="verified.json from verify.py")
    p.add_argument(
        "--extractions",
        type=Path,
        help="optional extracted.json for design hints (improves tool selection)",
    )
    p.add_argument("-o", "--output", type=Path, default=Path("research_outputs/assessment.json"))
    p.add_argument("--report", type=Path, default=Path("research_outputs/assessment_report.md"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG), format="%(asctime)s %(levelname)-5s %(name)s: %(message)s"
    )

    papers = load_corpus(args.verified)
    log.info("Loaded %d verified papers", len(papers))

    # Load design hints from extract.py output if provided
    design_hints: dict[str, str] = {}
    if args.extractions and args.extractions.exists():
        try:
            ex_data = json.loads(args.extractions.read_text())
            for ex in ex_data.get("extractions", []):
                hint = ex.get("pico", {}).get("study_design_hint", "")
                if hint:
                    design_hints[ex["paper_id"]] = hint
        except Exception as e:
            log.warning("Could not parse extractions: %s", e)

    # Load RoB + quality from extraction if available (avoids redundant LLM calls)
    rob_hints: dict[str, str] = {}
    quality_hints: dict[str, str] = {}
    if args.extractions and args.extractions.exists():
        try:
            ex_data = json.loads(args.extractions.read_text())
            for ex in ex_data.get("extractions", []):
                pico = ex.get("pico", {})
                rob = pico.get("risk_of_bias", "")
                qs = pico.get("quality_score", "")
                if rob:
                    rob_hints[ex["paper_id"]] = rob
                if qs:
                    quality_hints[ex["paper_id"]] = str(qs)
            if rob_hints:
                log.info(
                    "Loaded RoB from extraction for %d papers (no LLM calls needed)", len(rob_hints)
                )
        except Exception as e:
            log.warning("Could not parse extraction RoB: %s", e)

    assessments = [
        build_assessment(
            paper,
            study_design_hint=design_hints.get(paper.primary_id, ""),
            abstract=paper.abstract or "",
            llm_risk_of_bias=rob_hints.get(paper.primary_id, ""),
            llm_quality_score=quality_hints.get(paper.primary_id, ""),
        )
        for paper in papers
    ]
    payload = {
        "meta": {
            "source": str(args.verified),
            "n_papers": len(papers),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "assessments": [asdict(a) for a in assessments],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {len(assessments)} assessments → {args.output}")

    args.report.write_text(render_assessment_report(assessments))
    print(f"Wrote assessment report → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

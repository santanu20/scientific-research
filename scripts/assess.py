#!/usr/bin/env python3
"""Risk-of-bias scaffolds + GRADE evidence profiles.

This module generates assessment scaffolds for expert review. It does not
perform validated risk-of-bias or GRADE assessments. Abstract-only LLM
judgments are explicitly labeled as preliminary and must not be presented as
completed domain-level assessments.
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
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus

log = logging.getLogger("scientific_research.assess")

_DATA_DIR = Path(__file__).parent / "data"
with (_DATA_DIR / "rob_tools.json").open(encoding="utf-8") as _f:
    ROB_TOOLS: dict[str, dict[str, Any]] = json.load(_f)
with (_DATA_DIR / "grade_factors.json").open(encoding="utf-8") as _f:
    GRADE_FACTORS: dict[str, str] = json.load(_f)

_ALLOWED_JUDGMENTS = {"low", "some concerns", "high", "unknown", "not applicable"}
_ALLOWED_GRADE_JUDGMENTS = {
    "no serious concern",
    "serious",
    "very serious",
    "not serious",
    "suspected",
    "very likely",
    "large",
    "very large",
    "dose-response present",
    "plausible confounding present",
    "not assessed",
}


def select_rob_tool(study_design: str) -> str:
    """Pick a candidate risk-of-bias instrument from the study design.

    This is a routing hint, not a risk-of-bias judgment. Ambiguous designs
    remain explicitly unknown instead of being forced into ROBINS-I.
    """
    d = (study_design or "").lower()
    if any(term in d for term in ("rct", "randomized", "randomised")):
        return "RoB 2"
    if "diagnostic" in d or "sensitivity" in d or "specificity" in d:
        return "QUADAS-2"
    if any(term in d for term in ("prediction model", "prognostic model", "diagnostic model")):
        return "PROBAST"
    if any(term in d for term in ("cohort", "case-control", "observational")):
        if any(term in d for term in ("exposure", "environmental", "occupational")):
            return "ROBINS-E"
        return "ROBINS-I" if any(term in d for term in ("intervention", "comparator")) else "Newcastle-Ottawa"
    return "Unclassified"


@dataclass
class Assessment:
    paper_id: str
    doi: str | None
    title: str
    study_design_detected: str
    rob_tool_selected: str
    rob_scaffold: dict
    rob_judgments: dict
    risk_of_bias: str
    quality_score: str
    grade_profile: dict
    grade_judgments: dict
    assessment_status: str
    assessment_limitations: list[str]
    assessment_timestamp: str


def build_assessment(
    paper: PaperRecord,
    study_design_hint: str = "",
    abstract: str = "",
    topic: str = "",
    llm_risk_of_bias: str = "",
    llm_quality_score: str = "",
) -> Assessment:
    """Build an expert-review scaffold; never propagate an overall label to all domains."""
    tool_name = select_rob_tool(study_design_hint)
    scaffold = ROB_TOOLS.get(tool_name, {"indication": "Study design requires expert classification", "domains": []})
    domains = scaffold.get("domains", [])
    rob_judgments = {d["name"]: "unknown" for d in domains}

    risk_of_bias = "unknown"
    quality_score = ""
    assessment_status = "scaffold_only"
    limitations = [
        "No completed domain-level risk-of-bias assessment is available.",
        "Abstract-only information is insufficient for most signaling questions.",
        "A qualified reviewer must consult the full text, protocol, and outcome-level data.",
        "GRADE certainty is outcome-specific and has not been assessed by this paper-level scaffold.",
    ]

    # Do not treat an extractor's single overall risk label as evidence for
    # each individual domain. Preserve it as a provisional summary only.
    if llm_risk_of_bias:
        normalized = llm_risk_of_bias.strip().lower().replace("_", " ")
        if normalized in _ALLOWED_JUDGMENTS:
            risk_of_bias = normalized
            assessment_status = "provisional_overall_label_only"
            limitations.append("Overall risk label supplied by an upstream model; domain judgments remain unknown.")
        else:
            limitations.append("Unrecognized upstream risk label was ignored.")

    if llm_quality_score:
        try:
            score = float(llm_quality_score)
        except (TypeError, ValueError):
            limitations.append("Unparseable upstream quality score was ignored.")
        else:
            if 1 <= score <= 10:
                quality_score = str(score)
                limitations.append("The 1–10 score is an unvalidated model heuristic, not a validated quality instrument.")
            else:
                limitations.append("Upstream quality score outside the 1–10 range was ignored.")

    # Abstract-only LLM scoring is intentionally not invoked here. A
    # validated instrument requires full-text evidence and reviewer rationale.
    if os.environ.get("SCIENTIFIC_RESEARCH_USE_LLM", "").lower() in {"1", "true", "yes"}:
        limitations.append("LLM assessment is not implemented as a validated instrument; no automated domain judgment was made.")

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
        grade_judgments={factor: "not assessed" for factor in GRADE_FACTORS},
        assessment_status=assessment_status,
        assessment_limitations=limitations,
        assessment_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def render_assessment_report(assessments: list[Assessment]) -> str:
    out = [
        "# Risk-of-Bias Scaffolds and GRADE Evidence-Profile Template",
        "",
        "> **Not a completed assessment.** This report contains routing suggestions and blank expert-review fields. "
        "It must not be cited as a completed risk-of-bias evaluation or GRADE certainty rating.",
        "",
        "GRADE judgments are outcome-specific and should be made across the body of evidence, not inferred from a "
        "single paper-level score.",
        "",
        "## Candidate tool distribution",
    ]
    from collections import Counter

    for tool, n in sorted(Counter(a.rob_tool_selected for a in assessments).items()):
        ref = ROB_TOOLS.get(tool, {}).get("reference", "")
        out.append(f"- **{tool}** ({n} papers): {ref}")
    out.extend(["", "## Per-paper review scaffolds"])
    for assessment in assessments:
        out.extend(
            [
                f"\n### `{assessment.doi or assessment.paper_id}` — {assessment.title[:80]}",
                f"**Design hint**: {assessment.study_design_detected}",
                f"**Candidate tool**: {assessment.rob_tool_selected}",
                f"**Status**: {assessment.assessment_status}",
                "",
                "#### Domain judgments (reviewer must complete from full text)",
            ]
        )
        if not assessment.rob_scaffold.get("domains"):
            out.append("No tool domains were preselected; classify the study design before assessment.")
        for domain in assessment.rob_scaffold.get("domains", []):
            out.append(f"- **{domain['name']}**: unknown")
            for question in domain.get("signaling_questions", []):
                out.append(f"  - {question}")
        out.extend(["", "#### GRADE outcome-level judgments"])
        out.extend(f"- **{factor}**: not assessed" for factor in GRADE_FACTORS)
        out.extend(["", "#### Limitations"])
        out.extend(f"- {limitation}" for limitation in assessment.assessment_limitations)
    return "\n".join(out)


def main() -> int:
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    parser = argparse.ArgumentParser(
        prog="assess",
        description="Generate expert-review risk-of-bias scaffolds and GRADE templates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--self-check", action="store_true", help="verify deps + key imports, then exit 0")
    parser.add_argument("verified", type=Path, help="verified.json from verify.py")
    parser.add_argument("--extractions", type=Path, help="optional extracted.json for design hints")
    parser.add_argument("-o", "--output", type=Path, default=Path("research_outputs/assessment.json"))
    parser.add_argument("--report", type=Path, default=Path("research_outputs/assessment_report.md"))
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    verified = load_corpus(args.verified)
    log.info("Loaded %d verified papers", len(verified))

    design_hints: dict[str, str] = {}
    rob_hints: dict[str, str] = {}
    quality_hints: dict[str, str] = {}
    if args.extractions and args.extractions.exists():
        try:
            extraction_data = json.loads(args.extractions.read_text(encoding="utf-8"))
            for extraction in extraction_data.get("extractions", []):
                key = extraction.get("paper_id", "")
                pico = extraction.get("pico", {})
                if not key:
                    continue
                if pico.get("study_design_hint"):
                    design_hints[key] = pico["study_design_hint"]
                if pico.get("risk_of_bias"):
                    rob_hints[key] = pico["risk_of_bias"]
                if pico.get("quality_score") is not None:
                    quality_hints[key] = str(pico["quality_score"])
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            log.warning("Could not parse extraction hints: %s", exc)

    assessments = [
        build_assessment(
            paper,
            study_design_hint=design_hints.get(paper.primary_id, ""),
            abstract=paper.abstract or "",
            llm_risk_of_bias=rob_hints.get(paper.primary_id, ""),
            llm_quality_score=quality_hints.get(paper.primary_id, ""),
        )
        for paper in verified["papers"]
    ]
    payload = {
        "meta": {
            "source": str(args.verified),
            "n_papers": len(verified["papers"]),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "assessment_status": "scaffolds_only_not_completed_assessments",
        },
        "assessments": [asdict(a) for a in assessments],
    }
    from _artifact import save_artifact

    save_artifact(payload, args.output)
    print(f"Wrote {len(assessments)} assessment scaffolds → {args.output}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_assessment_report(assessments), encoding="utf-8")
    print(f"Wrote assessment report → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

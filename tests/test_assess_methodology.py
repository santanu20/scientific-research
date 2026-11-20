"""Methodological safety tests for risk-of-bias and GRADE scaffolds."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from assess import build_assessment, render_assessment_report, select_rob_tool


def paper():
    return SimpleNamespace(primary_id="doi:10.1234/example", doi="10.1234/example", title="Example trial", abstract="Short abstract")


def test_unknown_design_is_not_forced_into_robins_i():
    assert select_rob_tool("") == "Unclassified"
    assert select_rob_tool("cross-sectional survey") == "Unclassified"


def test_prediction_model_routes_to_probast():
    assert select_rob_tool("prognostic prediction model") == "PROBAST"


def test_abstract_or_overall_risk_does_not_fill_each_domain():
    assessment = build_assessment(paper(), study_design_hint="randomized controlled trial", abstract="Abstract", llm_risk_of_bias="high")
    assert assessment.risk_of_bias == "high"
    assert assessment.assessment_status == "provisional_overall_label_only"
    assert set(assessment.rob_judgments.values()) == {"unknown"}


def test_grade_judgments_remain_unassessed_until_outcome_level_review():
    assessment = build_assessment(paper(), study_design_hint="randomized trial")
    assert set(assessment.grade_judgments.values()) == {"not assessed"}
    assert "outcome-specific" in " ".join(assessment.assessment_limitations)


def test_rendered_report_cannot_be_mistaken_for_completed_assessment():
    assessment = build_assessment(paper(), study_design_hint="observational cohort")
    report = render_assessment_report([assessment])
    assert "Not a completed assessment" in report
    assert "reviewer must complete from full text" in report.lower()
    assert "not assessed" in report.lower()

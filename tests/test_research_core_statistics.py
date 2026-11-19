from __future__ import annotations

import pytest

from research_core import (
    AnalysisDataset,
    AnalysisModel,
    AnalysisPlan,
    ClaimEvidenceLink,
    ClaimKind,
    ConfidenceIntervalMethod,
    EffectEstimate,
    EffectMeasure,
    EvidenceRelation,
    EvidenceSpan,
    ResearchBundle,
    ResearchClaim,
    ResearchProtocol,
    SourceRecord,
    StatisticalContractError,
    TauSquaredEstimator,
)


DIGEST = "c" * 64


def make_plan(**overrides) -> AnalysisPlan:
    values = {
        "analysis_id": "analysis-1",
        "effect_measure": EffectMeasure.MEAN_DIFFERENCE,
        "outcome_id": "outcome-blood-pressure",
        "population_id": "adults-with-hypertension",
        "timepoint": "12-weeks",
        "study_design_family": "randomized-controlled-trial",
        "pooling_model": AnalysisModel.RANDOM_EFFECTS,
        "unit": "mmHg",
        "intervention_id": "drug-a",
        "comparator_id": "placebo",
        "tau_squared_estimator": TauSquaredEstimator.REML,
    }
    values.update(overrides)
    return AnalysisPlan(**values)


def make_estimate(effect_id: str, study_id: str, source_id: str, evidence_id: str, **overrides) -> EffectEstimate:
    values = {
        "effect_id": effect_id,
        "study_id": study_id,
        "source_id": source_id,
        "evidence_ids": (evidence_id,),
        "effect_measure": EffectMeasure.MEAN_DIFFERENCE,
        "outcome_id": "outcome-blood-pressure",
        "population_id": "adults-with-hypertension",
        "timepoint": "12-weeks",
        "study_design_family": "randomized-controlled-trial",
        "estimate": -5.2,
        "standard_error": 1.1,
        "unit": "mmHg",
        "intervention_id": "drug-a",
        "comparator_id": "placebo",
        "n_intervention": 120,
        "n_comparator": 118,
        "ci_lower": -7.4,
        "ci_upper": -3.0,
    }
    values.update(overrides)
    return EffectEstimate(**values)


def test_compatible_effects_form_a_valid_analysis_dataset() -> None:
    dataset = AnalysisDataset(
        make_plan(),
        (
            make_estimate("e1", "study-1", "source-1", "span-1"),
            make_estimate("e2", "study-2", "source-2", "span-2", estimate=-4.1, ci_lower=-6.1, ci_upper=-2.1),
        ),
    )
    assert len(dataset.estimates) == 2
    assert dataset.estimates[0].variance == pytest.approx(1.21)


def test_incompatible_units_cannot_be_pooled_as_mean_differences() -> None:
    with pytest.raises(StatisticalContractError, match="incompatible"):
        AnalysisDataset(
            make_plan(),
            (
                make_estimate("e1", "study-1", "source-1", "span-1"),
                make_estimate("e2", "study-2", "source-2", "span-2", unit="kPa"),
            ),
        )


@pytest.mark.parametrize(
    "dimension",
    [
        {"outcome_id": "weight"},
        {"population_id": "children"},
        {"timepoint": "24-weeks"},
        {"study_design_family": "observational-cohort"},
    ],
)
def test_incompatible_estimand_dimensions_are_rejected(dimension) -> None:
    with pytest.raises(StatisticalContractError, match="incompatible"):
        AnalysisDataset(
            make_plan(),
            (
                make_estimate("e1", "study-1", "source-1", "span-1"),
                make_estimate("e2", "study-2", "source-2", "span-2", **dimension),
            ),
        )


def test_duplicate_study_cannot_be_silently_double_counted() -> None:
    with pytest.raises(StatisticalContractError, match="double-count"):
        AnalysisDataset(
            make_plan(),
            (
                make_estimate("effect-1", "same-study", "source-1", "span-1"),
                make_estimate("effect-2", "same-study", "source-1", "span-2"),
            ),
        )


def test_mean_difference_requires_explicit_unit() -> None:
    with pytest.raises(StatisticalContractError, match="explicit unit"):
        make_plan(unit=None)


def test_standardized_effect_sizes_are_dimensionless() -> None:
    with pytest.raises(StatisticalContractError, match="unit must be omitted"):
        make_plan(effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE, unit="mmHg")
    plan = make_plan(effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE, unit=None)
    item = make_estimate(
        "smd-1", "study-1", "source-1", "span-1",
        effect_measure=EffectMeasure.STANDARDIZED_MEAN_DIFFERENCE,
        unit=None,
        estimate=-0.45,
        ci_lower=-0.7,
        ci_upper=-0.2,
    )
    assert item.compatibility_key == plan.compatibility_key


def test_random_effects_and_knapp_hartung_are_explicit() -> None:
    with pytest.raises(StatisticalContractError, match="tau_squared_estimator"):
        make_plan(tau_squared_estimator=None)
    plan = make_plan(confidence_interval_method=ConfidenceIntervalMethod.KNAPP_HARTUNG)
    assert plan.confidence_interval_method is ConfidenceIntervalMethod.KNAPP_HARTUNG


def test_fixed_effect_plan_rejects_random_effects_estimator() -> None:
    with pytest.raises(StatisticalContractError, match="must not specify"):
        make_plan(pooling_model=AnalysisModel.FIXED_EFFECT, tau_squared_estimator=TauSquaredEstimator.REML)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_standard_errors_are_rejected(value) -> None:
    with pytest.raises(StatisticalContractError, match="standard_error"):
        make_estimate("bad", "study", "source", "span", standard_error=value)


def test_confidence_interval_must_contain_effect_and_be_ordered() -> None:
    with pytest.raises(StatisticalContractError, match="contain the estimate"):
        make_estimate("bad", "study", "source", "span", ci_lower=-9, ci_upper=-6)
    with pytest.raises(StatisticalContractError, match="strictly ordered"):
        make_estimate("bad", "study", "source", "span", ci_lower=0, ci_upper=-1)


def test_effect_estimate_requires_source_evidence_ids() -> None:
    with pytest.raises(StatisticalContractError, match="at least one identifier"):
        make_estimate("bad", "study", "source", "span", evidence_ids=())


def test_estimates_must_be_grounded_in_their_own_source() -> None:
    protocol = ResearchProtocol(
        protocol_id="protocol",
        research_question="Does drug A improve blood pressure?",
        created_at_utc="2026-11-19T10:00:00Z",
        inclusion_criteria=("Adults",),
        exclusion_criteria=(),
        planned_sources=("crossref",),
    )
    sources = (
        SourceRecord("source-1", "test", "Study 1", "2026-11-19T10:01:00Z", provider_record_id="record-1", content_sha256=DIGEST),
        SourceRecord("source-2", "test", "Study 2", "2026-11-19T10:02:00Z", provider_record_id="record-2", content_sha256=DIGEST),
    )
    spans = (
        EvidenceSpan("span-1", "source-1", "Outcome reduced by 5 mmHg.", DIGEST, page_number=3),
        EvidenceSpan("span-2", "source-2", "Outcome reduced by 4 mmHg.", DIGEST, page_number=4),
    )
    bundle = ResearchBundle(protocol, (), sources, spans, (), ())
    valid = AnalysisDataset(
        make_plan(),
        (
            make_estimate("e1", "study-1", "source-1", "span-1"),
            make_estimate("e2", "study-2", "source-2", "span-2", estimate=-4.1, ci_lower=-6.1, ci_upper=-2.1),
        ),
    )
    valid.validate_against_bundle(bundle)

    invalid = AnalysisDataset(
        make_plan(),
        (
            make_estimate("e1", "study-1", "source-1", "span-2"),
            make_estimate("e2", "study-2", "source-2", "span-2", estimate=-4.1, ci_lower=-6.1, ci_upper=-2.1),
        ),
    )
    with pytest.raises(StatisticalContractError, match="different source"):
        invalid.validate_against_bundle(bundle)

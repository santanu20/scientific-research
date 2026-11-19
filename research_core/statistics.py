"""Typed statistical synthesis inputs and compatibility validation.

This module is a scientific input gate. It does not pool effects; a numerical
backend should consume only a validated AnalysisDataset and must preserve its
own method-specific diagnostics and sensitivity results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .models import ResearchBundle


class StatisticalContractError(ValueError):
    """Raised when an effect estimate or analysis group is not defensible."""


class EffectMeasure(StrEnum):
    MEAN_DIFFERENCE = "mean_difference"
    STANDARDIZED_MEAN_DIFFERENCE = "standardized_mean_difference"
    LOG_ODDS_RATIO = "log_odds_ratio"
    LOG_RISK_RATIO = "log_risk_ratio"
    LOG_HAZARD_RATIO = "log_hazard_ratio"
    LOG_INCIDENCE_RATE_RATIO = "log_incidence_rate_ratio"
    FISHER_Z_CORRELATION = "fisher_z_correlation"


class AnalysisModel(StrEnum):
    FIXED_EFFECT = "fixed_effect"
    RANDOM_EFFECTS = "random_effects"


class TauSquaredEstimator(StrEnum):
    REML = "reml"
    PAULE_MANDEL = "paule_mandel"
    DER_SIMONIAN_LAIRD = "der_simonian_laird"


class ConfidenceIntervalMethod(StrEnum):
    NORMAL = "normal"
    KNAPP_HARTUNG = "knapp_hartung"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StatisticalContractError(f"{name} must be a non-empty string")
    return value.strip()


def _enum(value: Any, enum_type: type[StrEnum], name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise StatisticalContractError(f"{name} must be one of: {allowed}") from exc


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatisticalContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise StatisticalContractError(f"{name} must be a finite number")
    return result


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StatisticalContractError(f"{name} must be a positive integer")
    return value


def _id_tuple(values: Iterable[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise StatisticalContractError(f"{name} must be an array")
    result = tuple(_text(value, f"{name}[{index}]") for index, value in enumerate(values))
    if not result:
        raise StatisticalContractError(f"{name} must contain at least one identifier")
    if len(result) != len(set(result)):
        raise StatisticalContractError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class AnalysisPlan:
    """Pre-specifies the estimand and pooling choices before estimates are added."""

    analysis_id: str
    effect_measure: EffectMeasure
    outcome_id: str
    population_id: str
    timepoint: str
    study_design_family: str
    pooling_model: AnalysisModel
    unit: str | None = None
    intervention_id: str | None = None
    comparator_id: str | None = None
    tau_squared_estimator: TauSquaredEstimator | None = None
    confidence_interval_method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.NORMAL
    minimum_studies: int = 2

    def __post_init__(self) -> None:
        for name in ("analysis_id", "outcome_id", "population_id", "timepoint", "study_design_family"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "effect_measure", _enum(self.effect_measure, EffectMeasure, "effect_measure"))
        object.__setattr__(self, "pooling_model", _enum(self.pooling_model, AnalysisModel, "pooling_model"))
        object.__setattr__(
            self,
            "confidence_interval_method",
            _enum(self.confidence_interval_method, ConfidenceIntervalMethod, "confidence_interval_method"),
        )
        for name in ("unit", "intervention_id", "comparator_id"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _text(value, name))
        if self.tau_squared_estimator is not None:
            object.__setattr__(
                self,
                "tau_squared_estimator",
                _enum(self.tau_squared_estimator, TauSquaredEstimator, "tau_squared_estimator"),
            )
        if isinstance(self.minimum_studies, bool) or not isinstance(self.minimum_studies, int) or self.minimum_studies < 2:
            raise StatisticalContractError("minimum_studies must be an integer greater than or equal to 2")

        if self.effect_measure is EffectMeasure.MEAN_DIFFERENCE and not self.unit:
            raise StatisticalContractError("mean difference analyses require an explicit unit")
        if self.effect_measure is not EffectMeasure.MEAN_DIFFERENCE and self.unit is not None:
            raise StatisticalContractError("unit must be omitted for standardized or dimensionless effect measures")
        if self.effect_measure is not EffectMeasure.FISHER_Z_CORRELATION:
            if not self.intervention_id or not self.comparator_id:
                raise StatisticalContractError("comparative effect measures require intervention_id and comparator_id")
            if self.intervention_id == self.comparator_id:
                raise StatisticalContractError("intervention and comparator must differ")
        elif self.intervention_id or self.comparator_id:
            raise StatisticalContractError("correlation analyses must not specify intervention/comparator arms")

        if self.pooling_model is AnalysisModel.RANDOM_EFFECTS and self.tau_squared_estimator is None:
            raise StatisticalContractError("random-effects analyses require a tau_squared_estimator")
        if self.pooling_model is AnalysisModel.FIXED_EFFECT and self.tau_squared_estimator is not None:
            raise StatisticalContractError("fixed-effect analyses must not specify a tau_squared_estimator")
        if (
            self.confidence_interval_method is ConfidenceIntervalMethod.KNAPP_HARTUNG
            and self.pooling_model is not AnalysisModel.RANDOM_EFFECTS
        ):
            raise StatisticalContractError("Knapp-Hartung inference requires a random-effects model")

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.effect_measure,
            self.outcome_id,
            self.population_id,
            self.timepoint,
            self.study_design_family,
            self.unit,
            self.intervention_id,
            self.comparator_id,
        )


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    """A study-level estimate on the scale declared by EffectMeasure.

    LOG_* measures use the natural-log scale; Fisher-z correlation uses the
    Fisher z scale. Standard errors and confidence limits must use that same
    analysis scale. Estimates must cite verbatim evidence spans from their
    originating source.
    """

    effect_id: str
    study_id: str
    source_id: str
    evidence_ids: tuple[str, ...]
    effect_measure: EffectMeasure
    outcome_id: str
    population_id: str
    timepoint: str
    study_design_family: str
    estimate: float
    standard_error: float
    unit: str | None = None
    intervention_id: str | None = None
    comparator_id: str | None = None
    n_intervention: int | None = None
    n_comparator: int | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "effect_id", "study_id", "source_id", "outcome_id",
            "population_id", "timepoint", "study_design_family",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "evidence_ids", _id_tuple(self.evidence_ids, "evidence_ids"))
        object.__setattr__(self, "effect_measure", _enum(self.effect_measure, EffectMeasure, "effect_measure"))
        for name in ("unit", "intervention_id", "comparator_id"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _text(value, name))
        object.__setattr__(self, "estimate", _finite(self.estimate, "estimate"))
        object.__setattr__(self, "standard_error", _finite(self.standard_error, "standard_error"))
        if self.standard_error <= 0:
            raise StatisticalContractError("standard_error must be greater than zero")

        for name in ("n_intervention", "n_comparator"):
            object.__setattr__(self, name, _optional_positive_int(getattr(self, name), name))
        if (self.n_intervention is None) != (self.n_comparator is None):
            raise StatisticalContractError("n_intervention and n_comparator must be supplied together")

        if self.effect_measure is EffectMeasure.MEAN_DIFFERENCE and not self.unit:
            raise StatisticalContractError("mean difference estimates require an explicit unit")
        if self.effect_measure is not EffectMeasure.MEAN_DIFFERENCE and self.unit is not None:
            raise StatisticalContractError("unit must be omitted for standardized or dimensionless effect measures")
        if self.effect_measure is not EffectMeasure.FISHER_Z_CORRELATION:
            if not self.intervention_id or not self.comparator_id:
                raise StatisticalContractError("comparative effect estimates require intervention_id and comparator_id")
            if self.intervention_id == self.comparator_id:
                raise StatisticalContractError("intervention and comparator must differ")
        elif self.intervention_id or self.comparator_id:
            raise StatisticalContractError("correlation estimates must not specify intervention/comparator arms")

        if (self.ci_lower is None) != (self.ci_upper is None):
            raise StatisticalContractError("ci_lower and ci_upper must be supplied together")
        if self.ci_lower is not None:
            lower = _finite(self.ci_lower, "ci_lower")
            upper = _finite(self.ci_upper, "ci_upper")
            if lower >= upper:
                raise StatisticalContractError("confidence interval must be strictly ordered")
            if not lower <= self.estimate <= upper:
                raise StatisticalContractError("confidence interval must contain the estimate")
            object.__setattr__(self, "ci_lower", lower)
            object.__setattr__(self, "ci_upper", upper)
        level = _finite(self.ci_level, "ci_level")
        if not 0.5 < level < 1.0:
            raise StatisticalContractError("ci_level must be greater than 0.5 and less than 1")
        object.__setattr__(self, "ci_level", level)

    @property
    def variance(self) -> float:
        return self.standard_error**2

    @property
    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            self.effect_measure,
            self.outcome_id,
            self.population_id,
            self.timepoint,
            self.study_design_family,
            self.unit,
            self.intervention_id,
            self.comparator_id,
        )


@dataclass(frozen=True, slots=True)
class AnalysisDataset:
    """Compatible study estimates for one univariate pooling stratum.

    One estimate per independent study is enforced. Multiple correlated
    outcomes/arms from the same study require a multivariate or multilevel
    model and must not be forced into this univariate container.
    """

    plan: AnalysisPlan
    estimates: tuple[EffectEstimate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, AnalysisPlan):
            raise StatisticalContractError("plan must be an AnalysisPlan")
        if not isinstance(self.estimates, (tuple, list)):
            raise StatisticalContractError("estimates must be an array")
        estimates = tuple(self.estimates)
        if any(not isinstance(item, EffectEstimate) for item in estimates):
            raise StatisticalContractError("estimates must contain EffectEstimate records")
        if len(estimates) < self.plan.minimum_studies:
            raise StatisticalContractError(f"analysis requires at least {self.plan.minimum_studies} study estimates")
        if len({item.effect_id for item in estimates}) != len(estimates):
            raise StatisticalContractError("duplicate effect_id values")
        if len({item.study_id for item in estimates}) != len(estimates):
            raise StatisticalContractError("duplicate study_id values would double-count a study in univariate pooling")
        for item in estimates:
            if item.compatibility_key != self.plan.compatibility_key:
                raise StatisticalContractError(f"effect {item.effect_id!r} is incompatible with the pre-specified analysis plan")
        object.__setattr__(self, "estimates", estimates)

    def validate_against_bundle(self, bundle: ResearchBundle) -> None:
        """Require every estimate to cite spans from its own exact source record."""
        if not isinstance(bundle, ResearchBundle):
            raise StatisticalContractError("bundle must be a ResearchBundle")
        spans = {span.evidence_id: span for span in bundle.evidence_spans}
        for estimate in self.estimates:
            for evidence_id in estimate.evidence_ids:
                span = spans.get(evidence_id)
                if span is None:
                    raise StatisticalContractError(f"effect {estimate.effect_id!r} references unknown evidence span {evidence_id!r}")
                if span.source_id != estimate.source_id:
                    raise StatisticalContractError(f"effect {estimate.effect_id!r} evidence span belongs to a different source")


__all__ = [
    "AnalysisDataset",
    "AnalysisModel",
    "AnalysisPlan",
    "ConfidenceIntervalMethod",
    "EffectEstimate",
    "EffectMeasure",
    "StatisticalContractError",
    "TauSquaredEstimator",
]

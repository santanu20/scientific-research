"""Independent screening and adjudication contracts.

These models distinguish machine suggestions from human decisions and encode
independent-review requirements without changing the legacy screening CLI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class ScreeningContractError(ValueError):
    """Raised when screening or adjudication data violates its contract."""


class ScreeningStage(StrEnum):
    TITLE_ABSTRACT = "title_abstract"
    FULL_TEXT = "full_text"


class ScreeningDecisionCode(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


class DecisionOrigin(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    RULE = "rule"


class AdjudicationStatus(StrEnum):
    CONSENSUS = "consensus"
    ADJUDICATED = "adjudicated"
    UNRESOLVED = "unresolved"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScreeningContractError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _required_text(value, name)


def _aware_timestamp(value: Any, name: str) -> str:
    value = _required_text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScreeningContractError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ScreeningContractError(f"{name} must include a timezone offset")
    return value


def _coerce_enum(value: Any, enum_type: type[StrEnum], name: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ScreeningContractError(f"invalid {name}: {value!r}") from exc


def _string_tuple(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ScreeningContractError(f"{name} must be an array of strings")
    result = tuple(_required_text(value, f"{name}[{index}]") for index, value in enumerate(values))
    if len(result) != len(set(result)):
        raise ScreeningContractError(f"{name} must not contain duplicates")
    return result


def _digest(value: Any, name: str) -> str:
    value = _required_text(value, name).lower()
    if not _SHA256_RE.fullmatch(value):
        raise ScreeningContractError(f"{name} must be a 64-character SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ScreeningDecision:
    """One decision by one actor on one source at one screening stage.

    HUMAN actors are identifiable reviewers. MODEL decisions require model,
    version, and prompt hash. RULE decisions require a rule identifier.
    """

    decision_id: str
    source_id: str
    stage: ScreeningStage
    decision: ScreeningDecisionCode
    actor_id: str
    origin: DecisionOrigin
    decided_at_utc: str
    rationale: str
    reason_code: str | None = None
    criterion_ids: tuple[str, ...] = ()
    source_sha256: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prompt_sha256: str | None = None
    rule_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("decision_id", "source_id", "actor_id", "rationale"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "stage", _coerce_enum(self.stage, ScreeningStage, "stage"))
        object.__setattr__(self, "decision", _coerce_enum(self.decision, ScreeningDecisionCode, "decision"))
        object.__setattr__(self, "origin", _coerce_enum(self.origin, DecisionOrigin, "origin"))
        object.__setattr__(self, "decided_at_utc", _aware_timestamp(self.decided_at_utc, "decided_at_utc"))
        object.__setattr__(self, "reason_code", _optional_text(self.reason_code, "reason_code"))
        object.__setattr__(self, "criterion_ids", _string_tuple(self.criterion_ids, "criterion_ids"))
        for name in ("model_name", "model_version", "rule_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        if self.source_sha256 is not None:
            object.__setattr__(self, "source_sha256", _digest(self.source_sha256, "source_sha256"))
        if self.prompt_sha256 is not None:
            object.__setattr__(self, "prompt_sha256", _digest(self.prompt_sha256, "prompt_sha256"))

        if self.decision is ScreeningDecisionCode.EXCLUDE and not self.reason_code:
            raise ScreeningContractError("exclude decisions require a reason_code")
        if self.stage is ScreeningStage.FULL_TEXT and not self.source_sha256:
            raise ScreeningContractError("full-text decisions require the reviewed source_sha256")
        if self.origin is DecisionOrigin.HUMAN:
            if any((self.model_name, self.model_version, self.prompt_sha256, self.rule_id)):
                raise ScreeningContractError("human decisions cannot carry model or rule execution metadata")
        elif self.origin is DecisionOrigin.MODEL:
            if not all((self.model_name, self.model_version, self.prompt_sha256)):
                raise ScreeningContractError("model decisions require model_name, model_version, and prompt_sha256")
            if self.rule_id:
                raise ScreeningContractError("model decisions cannot carry rule_id")
        elif self.origin is DecisionOrigin.RULE:
            if not self.rule_id:
                raise ScreeningContractError("rule decisions require rule_id")
            if any((self.model_name, self.model_version, self.prompt_sha256)):
                raise ScreeningContractError("rule decisions cannot carry model execution metadata")


@dataclass(frozen=True, slots=True)
class ScreeningAdjudication:
    """Resolution record over at least two independent human decisions."""

    adjudication_id: str
    source_id: str
    stage: ScreeningStage
    decision_ids: tuple[str, ...]
    status: AdjudicationStatus
    rationale: str
    adjudicated_at_utc: str
    final_decision: ScreeningDecisionCode | None = None
    adjudicator_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "source_id", "rationale"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name))
        object.__setattr__(self, "stage", _coerce_enum(self.stage, ScreeningStage, "stage"))
        object.__setattr__(self, "decision_ids", _string_tuple(self.decision_ids, "decision_ids"))
        object.__setattr__(self, "status", _coerce_enum(self.status, AdjudicationStatus, "status"))
        object.__setattr__(self, "adjudicated_at_utc", _aware_timestamp(self.adjudicated_at_utc, "adjudicated_at_utc"))
        if self.final_decision is not None:
            object.__setattr__(self, "final_decision", _coerce_enum(self.final_decision, ScreeningDecisionCode, "final_decision"))
        object.__setattr__(self, "adjudicator_id", _optional_text(self.adjudicator_id, "adjudicator_id"))
        if len(self.decision_ids) < 2:
            raise ScreeningContractError("adjudication requires at least two independent decision IDs")
        if self.status is AdjudicationStatus.CONSENSUS:
            if self.final_decision is None or self.adjudicator_id is not None:
                raise ScreeningContractError("consensus requires final_decision and no adjudicator_id")
        elif self.status is AdjudicationStatus.ADJUDICATED:
            if self.final_decision is None or self.adjudicator_id is None:
                raise ScreeningContractError("adjudicated outcomes require final_decision and adjudicator_id")
        elif self.status is AdjudicationStatus.UNRESOLVED:
            if self.final_decision is not None or self.adjudicator_id is not None:
                raise ScreeningContractError("unresolved outcomes cannot have a final decision or adjudicator")


@dataclass(frozen=True, slots=True)
class ScreeningReviewSet:
    """A validated group of decisions and their adjudication history."""

    decisions: tuple[ScreeningDecision, ...]
    adjudications: tuple[ScreeningAdjudication, ...]

    def __post_init__(self) -> None:
        decisions = tuple(self.decisions)
        adjudications = tuple(self.adjudications)
        if any(not isinstance(item, ScreeningDecision) for item in decisions):
            raise ScreeningContractError("decisions must contain ScreeningDecision records")
        if any(not isinstance(item, ScreeningAdjudication) for item in adjudications):
            raise ScreeningContractError("adjudications must contain ScreeningAdjudication records")
        if len({item.decision_id for item in decisions}) != len(decisions):
            raise ScreeningContractError("duplicate screening decision IDs")
        if len({item.adjudication_id for item in adjudications}) != len(adjudications):
            raise ScreeningContractError("duplicate adjudication IDs")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "adjudications", adjudications)

        by_id = {item.decision_id: item for item in decisions}
        adjudicated_sources: set[tuple[str, ScreeningStage]] = set()
        for adjudication in adjudications:
            key = (adjudication.source_id, adjudication.stage)
            if key in adjudicated_sources:
                raise ScreeningContractError("multiple adjudications for the same source and stage")
            adjudicated_sources.add(key)
            selected: list[ScreeningDecision] = []
            for decision_id in adjudication.decision_ids:
                decision = by_id.get(decision_id)
                if decision is None:
                    raise ScreeningContractError(f"adjudication references missing decision {decision_id!r}")
                if decision.source_id != adjudication.source_id or decision.stage is not adjudication.stage:
                    raise ScreeningContractError("adjudication decisions must concern the same source and stage")
                if decision.origin is not DecisionOrigin.HUMAN:
                    raise ScreeningContractError("independent adjudication requires human reviewer decisions")
                selected.append(decision)

            reviewer_ids = {item.actor_id for item in selected}
            if len(reviewer_ids) < 2:
                raise ScreeningContractError("independent review requires at least two distinct human reviewers")
            outcomes = {item.decision for item in selected}
            disagreement = len(outcomes) > 1

            if adjudication.status is AdjudicationStatus.CONSENSUS:
                if disagreement or adjudication.final_decision not in outcomes:
                    raise ScreeningContractError("consensus requires matching reviewer decisions")
            elif adjudication.status is AdjudicationStatus.ADJUDICATED:
                if not disagreement:
                    raise ScreeningContractError("adjudication is only valid when reviewers disagree")
                if adjudication.adjudicator_id in reviewer_ids:
                    raise ScreeningContractError("adjudicator must be independent of the original reviewers")
            elif adjudication.status is AdjudicationStatus.UNRESOLVED:
                if not disagreement:
                    raise ScreeningContractError("unresolved status requires reviewer disagreement")

    @property
    def unresolved(self) -> tuple[ScreeningAdjudication, ...]:
        return tuple(item for item in self.adjudications if item.status is AdjudicationStatus.UNRESOLVED)

    def decisions_for(self, source_id: str, stage: ScreeningStage) -> tuple[ScreeningDecision, ...]:
        return tuple(item for item in self.decisions if item.source_id == source_id and item.stage is stage)


__all__ = [
    "AdjudicationStatus",
    "DecisionOrigin",
    "ScreeningAdjudication",
    "ScreeningContractError",
    "ScreeningDecision",
    "ScreeningDecisionCode",
    "ScreeningReviewSet",
    "ScreeningStage",
]

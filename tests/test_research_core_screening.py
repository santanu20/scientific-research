from __future__ import annotations

import pytest

from research_core.screening import (
    AdjudicationStatus,
    DecisionOrigin,
    ScreeningAdjudication,
    ScreeningContractError,
    ScreeningDecision,
    ScreeningDecisionCode,
    ScreeningReviewSet,
    ScreeningStage,
)


def human_decision(decision_id: str, reviewer: str, result: ScreeningDecisionCode) -> ScreeningDecision:
    return ScreeningDecision(
        decision_id=decision_id,
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision=result,
        actor_id=reviewer,
        origin=DecisionOrigin.HUMAN,
        decided_at_utc="2026-11-19T09:00:00Z",
        rationale=f"{reviewer} applied the written eligibility criteria.",
        reason_code="wrong_population" if result is ScreeningDecisionCode.EXCLUDE else None,
        criterion_ids=("population", "outcome"),
    )


def test_two_independent_reviewers_can_reach_consensus() -> None:
    decisions = (
        human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE),
        human_decision("d2", "reviewer-b", ScreeningDecisionCode.INCLUDE),
    )
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.CONSENSUS,
        final_decision=ScreeningDecisionCode.INCLUDE,
        rationale="Both independent reviewers selected include.",
        adjudicated_at_utc="2026-11-19T09:05:00Z",
    )
    review = ScreeningReviewSet(decisions, (adjudication,))
    assert review.unresolved == ()
    assert len(review.decisions_for("paper-1", ScreeningStage.TITLE_ABSTRACT)) == 2


def test_disagreement_requires_a_distinct_adjudicator() -> None:
    decisions = (
        human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE),
        human_decision("d2", "reviewer-b", ScreeningDecisionCode.EXCLUDE),
    )
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.ADJUDICATED,
        final_decision=ScreeningDecisionCode.INCLUDE,
        adjudicator_id="reviewer-c",
        rationale="Third reviewer resolved the eligibility conflict.",
        adjudicated_at_utc="2026-11-19T09:10:00Z",
    )
    review = ScreeningReviewSet(decisions, (adjudication,))
    assert review.unresolved == ()


def test_disagreement_can_remain_explicitly_unresolved() -> None:
    decisions = (
        human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE),
        human_decision("d2", "reviewer-b", ScreeningDecisionCode.UNCERTAIN),
    )
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.UNRESOLVED,
        rationale="Full text is needed before a defensible decision can be made.",
        adjudicated_at_utc="2026-11-19T09:10:00Z",
    )
    review = ScreeningReviewSet(decisions, (adjudication,))
    assert len(review.unresolved) == 1


def test_same_person_cannot_be_counted_as_two_reviewers() -> None:
    decisions = (
        human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE),
        human_decision("d2", "reviewer-a", ScreeningDecisionCode.INCLUDE),
    )
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.CONSENSUS,
        final_decision=ScreeningDecisionCode.INCLUDE,
        rationale="Invalid duplicate reviewer.",
        adjudicated_at_utc="2026-11-19T09:05:00Z",
    )
    with pytest.raises(ScreeningContractError, match="distinct human reviewers"):
        ScreeningReviewSet(decisions, (adjudication,))


def test_machine_suggestions_are_not_human_adjudication_votes() -> None:
    machine = ScreeningDecision(
        decision_id="model-1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision=ScreeningDecisionCode.INCLUDE,
        actor_id="model-run-1",
        origin=DecisionOrigin.MODEL,
        decided_at_utc="2026-11-19T09:00:00Z",
        rationale="Model suggested inclusion for human review.",
        model_name="local-model",
        model_version="2026.11",
        prompt_sha256="a" * 64,
    )
    human = human_decision("human-1", "reviewer-a", ScreeningDecisionCode.INCLUDE)
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("model-1", "human-1"),
        status=AdjudicationStatus.CONSENSUS,
        final_decision=ScreeningDecisionCode.INCLUDE,
        rationale="A model suggestion must not be counted as reviewer agreement.",
        adjudicated_at_utc="2026-11-19T09:05:00Z",
    )
    with pytest.raises(ScreeningContractError, match="human reviewer decisions"):
        ScreeningReviewSet((machine, human), (adjudication,))


def test_exclusions_require_a_reason_code() -> None:
    with pytest.raises(ScreeningContractError, match="reason_code"):
        ScreeningDecision(
            decision_id="d1",
            source_id="paper-1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            decision=ScreeningDecisionCode.EXCLUDE,
            actor_id="reviewer-a",
            origin=DecisionOrigin.HUMAN,
            decided_at_utc="2026-11-19T09:00:00Z",
            rationale="The population does not match.",
        )


def test_full_text_decisions_require_reviewed_document_hash() -> None:
    with pytest.raises(ScreeningContractError, match="source_sha256"):
        ScreeningDecision(
            decision_id="d1",
            source_id="paper-1",
            stage=ScreeningStage.FULL_TEXT,
            decision=ScreeningDecisionCode.INCLUDE,
            actor_id="reviewer-a",
            origin=DecisionOrigin.HUMAN,
            decided_at_utc="2026-11-19T09:00:00Z",
            rationale="The full text meets the criteria.",
        )


def test_model_decisions_require_model_and_prompt_provenance() -> None:
    with pytest.raises(ScreeningContractError, match="model_name"):
        ScreeningDecision(
            decision_id="m1",
            source_id="paper-1",
            stage=ScreeningStage.TITLE_ABSTRACT,
            decision=ScreeningDecisionCode.UNCERTAIN,
            actor_id="model-run-1",
            origin=DecisionOrigin.MODEL,
            decided_at_utc="2026-11-19T09:00:00Z",
            rationale="Model could not determine eligibility.",
        )


def test_adjudication_rejects_missing_decision_and_wrong_source() -> None:
    one = human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE)
    two = human_decision("d2", "reviewer-b", ScreeningDecisionCode.INCLUDE)
    missing = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-2",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.CONSENSUS,
        final_decision=ScreeningDecisionCode.INCLUDE,
        rationale="Wrong source.",
        adjudicated_at_utc="2026-11-19T09:05:00Z",
    )
    with pytest.raises(ScreeningContractError, match="same source and stage"):
        ScreeningReviewSet((one, two), (missing,))


def test_adjudicated_outcome_requires_a_third_reviewer() -> None:
    decisions = (
        human_decision("d1", "reviewer-a", ScreeningDecisionCode.INCLUDE),
        human_decision("d2", "reviewer-b", ScreeningDecisionCode.EXCLUDE),
    )
    adjudication = ScreeningAdjudication(
        adjudication_id="a1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision_ids=("d1", "d2"),
        status=AdjudicationStatus.ADJUDICATED,
        final_decision=ScreeningDecisionCode.UNCERTAIN,
        adjudicator_id="reviewer-a",
        rationale="Invalid self-adjudication.",
        adjudicated_at_utc="2026-11-19T09:10:00Z",
    )
    with pytest.raises(ScreeningContractError, match="independent of the original reviewers"):
        ScreeningReviewSet(decisions, (adjudication,))


def test_review_records_reject_timezone_naive_dates() -> None:
    with pytest.raises(ScreeningContractError, match="timezone"):
        human_decision_with_timestamp("2026-11-19T09:00:00")


def human_decision_with_timestamp(timestamp: str) -> ScreeningDecision:
    return ScreeningDecision(
        decision_id="d1",
        source_id="paper-1",
        stage=ScreeningStage.TITLE_ABSTRACT,
        decision=ScreeningDecisionCode.INCLUDE,
        actor_id="reviewer-a",
        origin=DecisionOrigin.HUMAN,
        decided_at_utc=timestamp,
        rationale="A recorded reviewer rationale.",
    )

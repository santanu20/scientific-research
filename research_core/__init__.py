"""Canonical, versioned research-domain contracts.

This package is intentionally standard-library-only. Pipeline/provider adapters
should depend on these domain contracts, not the reverse.
"""

from .models import (
    SCHEMA_VERSION,
    ClaimEvidenceLink,
    ClaimKind,
    ContractError,
    EvidenceRelation,
    EvidenceSpan,
    ProtocolStatus,
    ResearchBundle,
    ResearchClaim,
    ResearchProtocol,
    ReviewStatus,
    SearchExecution,
    SearchExecutionStatus,
    SourceRecord,
)
from .screening import (
    AdjudicationStatus,
    DecisionOrigin,
    ScreeningAdjudication,
    ScreeningContractError,
    ScreeningDecision,
    ScreeningDecisionCode,
    ScreeningReviewSet,
    ScreeningStage,
)

__all__ = [
    "SCHEMA_VERSION",
    "ClaimEvidenceLink",
    "ClaimKind",
    "ContractError",
    "EvidenceRelation",
    "EvidenceSpan",
    "ProtocolStatus",
    "ResearchBundle",
    "ResearchClaim",
    "ResearchProtocol",
    "ReviewStatus",
    "SearchExecution",
    "SearchExecutionStatus",
    "SourceRecord",
    "AdjudicationStatus",
    "DecisionOrigin",
    "ScreeningAdjudication",
    "ScreeningContractError",
    "ScreeningDecision",
    "ScreeningDecisionCode",
    "ScreeningReviewSet",
    "ScreeningStage",
]

"""Claim-level evidence synthesis — structured claims, contradictions, grading.

Transforms the pipeline from paper→summary to claim→evidence→consensus.

Key components:
    - Claim dataclass: structured claim with supporting/conflicting papers
    - extract_claims(): build claims from extracted findings
    - detect_numerical_contradictions(): compare T/P values across papers
    - grade_evidence(): GRADE-adapted per-conclusion scoring
    - cluster_claims_semantic(): BGE cosine similarity clustering

References:
    Guyatt GH et al. GRADE: an emerging consensus on rating quality of evidence
    and strength of recommendations. BMJ 2008;336:924-926.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("scientific_research.claims")


# =============================================================================
# Claim dataclass
# =============================================================================


@dataclass
class Claim:
    """A structured scientific claim backed by evidence.

    Attributes:
        statement: human-readable claim text
        refs: paper reference numbers supporting this claim
        supporting: refs that agree with the claim
        conflicting: refs that disagree
        measurements: {"temperature": [750, 850], "pressure": [8.5, 10.2]}
        consensus: "strong" | "moderate" | "contested" | "uncertain"
        evidence_grade: "A" (high) | "B" (moderate) | "C" (low) | "D" (very low)
        confidence: 0.0-1.0 overall confidence
        contradictions: list of contradiction dicts
    """

    statement: str = ""
    refs: list[int] = field(default_factory=list)
    supporting: list[int] = field(default_factory=list)
    conflicting: list[int] = field(default_factory=list)
    measurements: dict[str, list[float]] = field(default_factory=dict)
    consensus: str = "uncertain"
    evidence_grade: str = "D"
    confidence: float = 0.0
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    topic: str = ""  # cluster label / theme


# =============================================================================
# Claim extraction from findings
# =============================================================================


def extract_claims(items: list[dict]) -> list[Claim]:
    """Build structured claims from extracted findings.

    Each item should have:
        - ref: paper reference number
        - finding: extracted finding text
        - measurements: dict of measurement type → list of values
    """
    claims: list[Claim] = []

    # Group items by claim topic using semantic similarity (if available)
    # or keyword overlap (fallback)
    clusters = _cluster_items(items)

    for cluster in clusters:
        refs = sorted({it.get("ref", 0) for it in cluster})
        if not refs:
            continue

        # Build claim statement from best finding
        best = max(cluster, key=lambda x: len(x.get("finding", "")))
        statement = best.get("finding", "")[:200]
        if not statement:
            continue

        # Collect measurements
        measurements: dict[str, list[float]] = {}
        for it in cluster:
            for mtype, values in (it.get("measurements") or {}).items():
                if isinstance(values, list):
                    measurements.setdefault(mtype, []).extend(
                        v for v in values if isinstance(v, (int, float))
                    )
                elif isinstance(values, (int, float)):
                    measurements.setdefault(mtype, []).append(values)

        # Detect contradictions
        contradictions = detect_numerical_contradictions(measurements, refs)

        # Determine consensus
        if contradictions:
            consensus = "contested"
            supporting = [
                r for r in refs if r not in [c.get("ref2") for c in contradictions]
            ]
            conflicting = sorted(
                {c.get("ref2") for c in contradictions if c.get("ref2")}
            )
        else:
            consensus = (
                "strong"
                if len(refs) >= 5
                else ("moderate" if len(refs) >= 3 else "uncertain")
            )
            supporting = refs
            conflicting = []

        # Grade evidence
        grade, confidence = grade_evidence(
            n_papers=len(refs),
            consensus=consensus,
            has_experimental=any(
                "experimental" in (it.get("finding", "").lower()) for it in cluster
            ),
            has_measurements=bool(measurements),
            has_contradictions=bool(contradictions),
        )

        topic = best.get("topic", "")
        claims.append(
            Claim(
                statement=statement,
                refs=refs,
                supporting=supporting,
                conflicting=conflicting,
                measurements=measurements,
                consensus=consensus,
                evidence_grade=grade,
                confidence=confidence,
                contradictions=contradictions,
                topic=topic,
            )
        )

    return claims


# =============================================================================
# Numerical contradiction detection
# =============================================================================


# Thresholds: relative difference that constitutes a contradiction
_CONTRADICTION_THRESHOLDS = {
    "temperature": 0.15,  # 15% relative difference (e.g., 750 vs 863+)
    "pressure": 0.20,  # 20% relative difference
    "age": 0.25,  # 25% (geochronology often less precise)
    "default": 0.25,
}

# Minimum absolute difference to avoid flagging rounding noise
_MIN_ABSOLUTE_DELTA = {
    "temperature": 50,  # °C
    "pressure": 1.0,  # kbar
    "age": 5.0,  # Ma
    "default": 0.0,
}


def detect_numerical_contradictions(
    measurements: dict[str, list[float]], refs: list[int]
) -> list[dict[str, Any]]:
    """Detect numerical contradictions within a claim's measurements.

    Compares measurement values pairwise and flags pairs where the relative
    difference exceeds the domain-specific threshold.

    Returns list of contradiction dicts:
        {"metric": "temperature", "val1": 750, "val2": 850, "delta": 100,
         "relative_delta": 0.13, "severity": "moderate"}
    """
    contradictions: list[dict[str, Any]] = []

    for mtype, values in measurements.items():
        if len(values) < 2:
            continue

        threshold = _CONTRADICTION_THRESHOLDS.get(
            mtype, _CONTRADICTION_THRESHOLDS["default"]
        )
        min_delta = _MIN_ABSOLUTE_DELTA.get(mtype, _MIN_ABSOLUTE_DELTA["default"])

        # Sort values and check adjacent pairs
        sorted_vals = sorted(values)
        for i in range(len(sorted_vals) - 1):
            v1, v2 = sorted_vals[i], sorted_vals[i + 1]
            delta = abs(v2 - v1)
            if delta < min_delta:
                continue
            mean_val = (v1 + v2) / 2.0
            if mean_val == 0:
                continue
            rel_delta = delta / abs(mean_val)

            if rel_delta > threshold:
                severity = "severe" if rel_delta > threshold * 2 else "moderate"
                contradictions.append(
                    {
                        "metric": mtype,
                        "val1": v1,
                        "val2": v2,
                        "delta": round(delta, 1),
                        "relative_delta": round(rel_delta, 3),
                        "severity": severity,
                        "ref1": refs[i] if i < len(refs) else None,
                        "ref2": refs[i + 1] if i + 1 < len(refs) else None,
                    }
                )

    return contradictions


# =============================================================================
# Evidence grading (GRADE-adapted for Earth Sciences)
# =============================================================================


def grade_evidence(
    n_papers: int,
    consensus: str,
    has_experimental: bool = False,
    has_measurements: bool = False,
    has_contradictions: bool = False,
) -> tuple[str, float]:
    """Grade evidence quality using a GRADE-adapted framework.

    Returns (grade_letter, confidence_float):
        A = High (≥0.8)
        B = Moderate (0.6-0.79)
        C = Low (0.4-0.59)
        D = Very Low (<0.4)
    """
    score = 0.0

    # Evidence volume (0-0.35)
    if n_papers >= 10:
        score += 0.35
    elif n_papers >= 5:
        score += 0.25
    elif n_papers >= 3:
        score += 0.15
    elif n_papers >= 1:
        score += 0.05

    # Consensus level (0-0.30)
    consensus_scores = {
        "strong": 0.30,
        "moderate": 0.20,
        "uncertain": 0.10,
        "contested": 0.05,
    }
    score += consensus_scores.get(consensus, 0.05)

    # Study quality (0-0.20)
    if has_experimental:
        score += 0.20  # experimental calibration is highest quality
    elif has_measurements:
        score += 0.12  # quantitative measurements
    else:
        score += 0.03  # qualitative only

    # Contradiction penalty (-0.15)
    if has_contradictions:
        score -= 0.15

    # Clamp
    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        grade = "A"
    elif score >= 0.6:
        grade = "B"
    elif score >= 0.4:
        grade = "C"
    else:
        grade = "D"

    return grade, round(score, 2)


def render_evidence_grade(claim: Claim) -> str:
    """Render a human-readable evidence grade string for a claim."""
    stars_map = {"A": 5, "B": 4, "C": 3, "D": 2}
    stars = stars_map.get(claim.evidence_grade, 2)
    n = len(claim.refs)
    n_supp = len(claim.supporting)
    n_conf = len(claim.conflicting)

    parts = [
        f"Evidence: {'★' * stars}{'☆' * (5 - stars)}",
        f"Agreement: {n_supp}/{n} papers agree",
    ]
    if n_conf:
        parts.append(f"Conflicts: {n_conf}")
    parts.append(f"Confidence: {claim.confidence:.2f}")

    return " | ".join(parts)


# =============================================================================
# Semantic claim clustering (BGE cosine similarity)
# =============================================================================


def _cluster_items(
    items: list[dict], similarity_threshold: float = 0.65
) -> list[list[dict]]:
    """Cluster items by semantic similarity using BGE embeddings.

    Falls back to keyword overlap if embeddings unavailable.
    """
    if len(items) <= 1:
        return [items] if items else []

    # BGE embeddings — opt-in feature (requires fastembed). Keyword fallback if absent.
    from _embeddings import is_available as _embeddings_available

    if not _embeddings_available():
        log.debug(
            "Semantic clustering skipped — fastembed not installed; using keyword"
        )
        return _cluster_items_keyword(items)

    import numpy as np
    from _embeddings import embed_texts

    texts = [it.get("finding", "") or it.get("topic", "") or "" for it in items]
    if not any(texts):
        return [items]

    try:
        embeddings = embed_texts(texts)
        if embeddings is None:
            return _cluster_items_keyword(items)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        sim_matrix = normalized @ normalized.T

        # Greedy clustering
        assigned: set[int] = set()
        clusters: list[list[dict]] = []
        for i in range(len(items)):
            if i in assigned:
                continue
            cluster = [items[i]]
            assigned.add(i)
            for j in range(i + 1, len(items)):
                if j in assigned:
                    continue
                if sim_matrix[i, j] >= similarity_threshold:
                    cluster.append(items[j])
                    assigned.add(j)
            clusters.append(cluster)
        return clusters
    except Exception as e:
        log.warning("Semantic clustering failed at runtime, using keyword: %s", e)
        return _cluster_items_keyword(items)


def _cluster_items_keyword(
    items: list[dict], threshold: float = 0.20
) -> list[list[dict]]:
    """Fallback: cluster items by keyword Jaccard overlap."""
    assigned: set[int] = set()
    clusters: list[list[dict]] = []

    for i, it in enumerate(items):
        if i in assigned:
            continue
        cluster = [it]
        assigned.add(i)
        words_i = set((it.get("finding", "") or "").lower().split())
        if not words_i:
            continue
        for j in range(i + 1, len(items)):
            if j in assigned:
                continue
            words_j = set((items[j].get("finding", "") or "").lower().split())
            if not words_j:
                continue
            overlap = len(words_i & words_j) / len(words_i | words_j)
            if overlap >= threshold:
                cluster.append(items[j])
                assigned.add(j)
        clusters.append(cluster)

    return clusters


# =============================================================================
# Claim rendering for narrative
# =============================================================================


def render_claim_summary(claims: list[Claim]) -> str:
    """Render a summary section for all claims with evidence grading."""
    if not claims:
        return ""

    lines = ["## Evidence Assessment", ""]

    # Sort by confidence descending
    sorted_claims = sorted(claims, key=lambda c: -c.confidence)

    for i, claim in enumerate(sorted_claims, 1):
        refs_str = ", ".join(f"[{r}]" for r in claim.refs[:8])
        if len(claim.refs) > 8:
            refs_str += f" ... +{len(claim.refs) - 8}"

        lines.append(
            f"### Claim {i}: {claim.statement[:120]}{'...' if len(claim.statement) > 120 else ''}"
        )
        lines.append(f"- **References**: {refs_str}")
        lines.append(f"- **Grade**: {render_evidence_grade(claim)}")

        if claim.contradictions:
            lines.append("- **Contradictions**:")
            for c in claim.contradictions:
                lines.append(
                    f"  - {c['metric'].title()}: [{c.get('ref1', '?')}] {c['val1']:.0f} "
                    f"vs [{c.get('ref2', '?')}] {c['val2']:.0f} "
                    f"(Δ={c['delta']:.0f}, {c['relative_delta']:.0%}, {c['severity']})"
                )

        if claim.measurements:
            parts = []
            for mtype, values in claim.measurements.items():
                if values:
                    med = statistics.median(values)
                    parts.append(f"{mtype}: median {med:.1f} (n={len(values)})")
            if parts:
                lines.append(f"- **Measurements**: {'; '.join(parts)}")
        lines.append("")

    return "\n".join(lines)


def render_contradiction_section(claims: list[Claim]) -> str:
    """Render a dedicated contradictions/debates section."""
    contested = [c for c in claims if c.contradictions]
    if not contested:
        return ""

    lines = ["## Key Disagreements", ""]
    lines.append(
        "The following claims show numerical disagreements across studies, "
        "indicating areas where consensus has not yet been reached."
    )
    lines.append("")

    for claim in contested:
        refs_str = ", ".join(f"[{r}]" for r in claim.refs[:5])
        lines.append(
            f"### {claim.statement[:100]}{'...' if len(claim.statement) > 100 else ''}"
        )
        lines.append(f"*Papers: {refs_str}*")
        lines.append("")

        for c in claim.contradictions:
            lines.append(
                f"- **{c['metric'].title()}**: Paper [{c.get('ref1', '?')}] reports "
                f"{c['val1']:.0f} while Paper [{c.get('ref2', '?')}] finds "
                f"{c['val2']:.0f} (Δ={c['delta']:.0f}, {c['severity']})"
            )
        lines.append("")

    return "\n".join(lines)

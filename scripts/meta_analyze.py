#!/usr/bin/env python3
"""Meta-analysis pooling on extracted effect sizes.

Reads extracted.json (from extract.py) + verified.json, pools effect sizes
using fixed/random-effects DerSimonian-Laird, computes heterogeneity (Cochrane
Q, I², τ²), and emits forest plot data + subgroup analyses.

SOTA approach:
    - DerSimonian-Laird random-effects (DerSimonian R, Laird N. Control Clin
      Trials 1986;7:177-188) — standard for last 40 years.
    - Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment for confidence intervals
      (Int J Evid Based Healthc 2009) — DEFAULT since 2026-08-14 (--no-hksj to
      disable). τ² estimator: REML default (--tau2 dl|reml|pm).
    - I² classification: low <25%, moderate 25-50%, substantial >50% (Higgins
      2003).

Inputs (built by extract.py):
    - extracted.json containing effect size candidates (mean±SD, events, OR/RR/HR)
    - verified.json containing paper metadata

Outputs:
    - meta.json: pooled effects, heterogeneity, weights
    - forest.mmd: ASCII forest plot in markdown code block
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
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _render import (
    render_forest_plot,
    render_forest_plot_png,
    render_funnel_plot_png,
)
from _sources import PaperRecord, load_corpus
from _stats import (
    cohens_d,
    confidence_interval,
    egger_test,
    hedges_g,
    heterogeneity,
    interpret_i_squared,
    leave_one_out,
    log_transform_or,
    log_transform_rr,
    odds_ratio,
    pool_fixed,
    pool_random_advanced,
    risk_ratio,
    se_from_variance,
    trim_and_fill,
    variance_d,
    variance_log_or,
    variance_log_rr,
)

log = logging.getLogger("scientific_research.meta_analyze")


# =============================================================================
# Effect extraction from extracted.json
# =============================================================================
@dataclass
class StudyEffect:
    """Per-study effect size + variance on the analysis scale."""

    paper_id: str
    doi: str | None
    name: str
    effect: float  # on analysis scale (d, log(OR), log(RR), raw MD)
    variance: float
    ci_lower: float
    ci_upper: float
    scale: str  # 'd' | 'log_or' | 'log_rr' | 'md' | 'custom'
    scale_label: str  # human-readable
    subgroup: str = ""  # optional subgroup label (year-bucket, design, etc.)
    notes: str = ""


def _safe_float(val) -> float | None:
    """Convert to float, return None if invalid."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def collect_continuous_effects(
    extractions: list[dict], papers: list[PaperRecord], correction: str = "hedges"
) -> list[StudyEffect]:
    """Build StudyEffects from mean±SD groups in extracted.json.
    Pairs of (mean, sd, n) — uses Hedges' g by default."""
    by_id = {p.primary_id: p for p in papers}
    out: list[StudyEffect] = []
    for ex in extractions:
        pid = ex.get("paper_id") or ""
        paper = by_id.get(pid)
        if not paper:
            continue
        groups = ex.get("effect_sizes", {}).get("mean_sd_groups", [])
        if not groups:
            continue

        # Handle both formats:
        # LLM format: each entry is a PAIR with m1+m2 (intervention + control)
        # Regex format: entries are separate groups, need 2 for a pair
        paired_entries = [
            g for g in groups if g.get("m2") is not None and g.get("m1") is not None
        ]

        if paired_entries:
            # LLM format: each entry has both groups
            for g in paired_entries:
                try:
                    m1 = _safe_float(g["m1"])
                    m2 = _safe_float(g["m2"])
                    sd1 = _safe_float(g["sd1"]) if g.get("sd1") is not None else None
                    sd2 = (
                        _safe_float(g.get("sd2")) if g.get("sd2") is not None else None
                    )
                    n1 = int(g.get("n") or 0)
                    n2 = int(g.get("n2") or g.get("n") or 0)
                    ci_lo = (
                        _safe_float(g["ci_lower"])
                        if g.get("ci_lower") is not None
                        else None
                    )
                    ci_hi = (
                        _safe_float(g["ci_upper"])
                        if g.get("ci_upper") is not None
                        else None
                    )
                    outcome = g.get("outcome", "") or ""
                    if m1 is None or m2 is None:
                        continue  # means missing → no effect computable

                    # Determine subgroup: use outcome type if available, else year
                    subgroup = outcome[:40] if outcome else str(paper.year or "")

                    if sd1 is not None and sd2 is not None and n1 >= 2 and n2 >= 2:
                        # Full data: compute Cohen's d
                        d = cohens_d(m1, sd1, n1, m2, sd2, n2)
                        eff = hedges_g(d, n1, n2) if correction == "hedges" else d
                        v = variance_d(n1, n2, d)
                        se = se_from_variance(v)
                        ci = confidence_interval(eff, se)
                        out.append(
                            StudyEffect(
                                paper_id=pid,
                                doi=paper.doi,
                                name=(paper.title or pid)[:50],
                                effect=float(eff),
                                variance=float(v),
                                ci_lower=float(ci[0]),
                                ci_upper=float(ci[1]),
                                scale="d",
                                scale_label=f"Hedges g (correction={correction})",
                                subgroup=subgroup,
                                notes=f"n1={n1},n2={n2}",
                            )
                        )
                    elif (
                        ci_lo is not None and ci_hi is not None and n1 >= 2 and n2 >= 2
                    ):
                        # SDs missing but CI available: compute variance from CI
                        # CI for mean difference: SE = (CI_upper - CI_lower) / (2 * z)
                        # For 95% CI, z = 1.96
                        md = m1 - m2
                        se_ci = (ci_hi - ci_lo) / (2 * 1.96)
                        v = se_ci**2
                        ci_out = confidence_interval(md, se_ci)
                        out.append(
                            StudyEffect(
                                paper_id=pid,
                                doi=paper.doi,
                                name=(paper.title or pid)[:50],
                                effect=float(md),
                                variance=float(v),
                                ci_lower=float(ci_out[0]),
                                ci_upper=float(ci_out[1]),
                                scale="md",
                                scale_label="Mean difference (CI-based variance)",
                                subgroup=subgroup,
                                notes=f"n1={n1},n2={n2} (CI→variance)",
                            )
                        )
                        log.info("Effect from %s: MD=%.2f (CI-based variance)", pid, md)
                    elif n1 >= 2 and n2 >= 2:
                        # No SDs, no CI: use raw mean difference with placeholder variance
                        md = m1 - m2
                        v = 1.0 / n1 + 1.0 / n2  # conservative placeholder
                        se = se_from_variance(v)
                        ci = confidence_interval(md, se)
                        out.append(
                            StudyEffect(
                                paper_id=pid,
                                doi=paper.doi,
                                name=(paper.title or pid)[:50],
                                effect=float(md),
                                variance=float(v),
                                ci_lower=float(ci[0]),
                                ci_upper=float(ci[1]),
                                scale="md",
                                scale_label="Mean difference (SD+CI unavailable)",
                                subgroup=subgroup,
                                notes=f"n1={n1},n2={n2} (no SD/CI — raw mean diff)",
                            )
                        )
                        log.info("Effect from %s: raw MD=%.2f (no SD/CI)", pid, md)
                except (KeyError, ValueError, TypeError) as e:
                    log.debug("Skipping effect from %s: %s", pid, e)
        elif len(groups) >= 2:
            # Regex format: first 2 entries as intervention vs control
            g1, g2 = groups[0], groups[1]
            try:
                m1, sd1, n1 = (
                    _safe_float(g1["m1"]),
                    _safe_float(g1["sd1"]),
                    int(g1.get("n") or 0),
                )
                m2, sd2, n2 = (
                    _safe_float(g2["m1"]),
                    _safe_float(g2["sd1"]),
                    int(g2.get("n") or 0),
                )
                if m1 is None or m2 is None or sd1 is None or sd2 is None:
                    continue  # incomplete stats → no effect computable
                if n1 < 2 or n2 < 2:
                    continue
                d = cohens_d(m1, sd1, n1, m2, sd2, n2)
                eff = hedges_g(d, n1, n2) if correction == "hedges" else d
                v = variance_d(n1, n2, d)
                se = se_from_variance(v)
                ci = confidence_interval(eff, se)
                out.append(
                    StudyEffect(
                        paper_id=pid,
                        doi=paper.doi,
                        name=(paper.title or pid)[:50],
                        effect=float(eff),
                        variance=float(v),
                        ci_lower=float(ci[0]),
                        ci_upper=float(ci[1]),
                        scale="d",
                        scale_label=f"Hedges g (correction={correction})",
                        subgroup=str(paper.year or ""),
                        notes=f"n1={n1},n2={n2} (regex format)",
                    )
                )
            except (KeyError, ValueError, TypeError) as e:
                log.debug("Skipping effect from %s: %s", pid, e)
    return out


def collect_single_measurements(
    extractions: list[dict], papers: list[PaperRecord]
) -> list[StudyEffect]:
    """Build StudyEffects from single measurements (no paired groups).

    For non-clinical fields (geology, chemistry, physics) where papers report
    single measured values (e.g., δ18O = 5.2‰, Fe3+/ΣFe = 0.15) rather than
    intervention vs control groups.

    B5b unit-partition gate (2026-08-15): pooling only happens WITHIN a
    (measurement, unit) group. Incommensurable units (%, °C, GPa, Ma) are
    NEVER pooled together — a cross-unit pool is scientifically meaningless
    even when the arithmetic succeeds. Singleton groups are logged and
    dropped (k=1 has no pooling meaning).
    """
    by_id = {p.primary_id: p for p in papers}
    grouped: dict[
        tuple[str, str], list[tuple[PaperRecord, dict, float, float | None, int, str]]
    ] = {}
    skipped_unbound = 0
    from _units import BASE_MEASUREMENT, canonical_unit, to_canonical

    def _classify(outcome: str, unit: str, val: float, unc: float | None):
        """Unit-canonicalize one row → ((outcome, base_unit), val, unc, orig_unit).

        Generic label + convertible unit → family default measurement
        (fixes 'measurement (ka)' misfiling: ka is age family).
        """
        if outcome == "measurement" and unit not in ("", "unitless"):
            cu = canonical_unit(unit)
            if cu:
                outcome = BASE_MEASUREMENT[cu]
        if unit in ("", "unitless"):
            return (outcome, "unitless"), val, unc, unit
        conv = to_canonical(val, unit, unc)
        if conv is None:
            return (
                (outcome, unit),
                val,
                unc,
                unit,
            )  # exotic unit: no conversion, pools alone
        base_u, val_c, unc_c = conv
        return (outcome, base_u), val_c, unc_c, unit

    for ex in extractions:
        pid = ex.get("paper_id") or ""
        paper = by_id.get(pid)
        if not paper:
            continue

        # Source 1: single_measurements field (from _effect_parser)
        singles = ex.get("effect_sizes", {}).get("single_measurements", [])
        for s in singles:
            try:
                val = s.get("value")
                if val is None:
                    continue
                val = float(val)
                unc = s.get("uncertainty")
                if unc is not None:
                    unc = float(unc)
                n = s.get("n")
                if n is not None:
                    n = int(n)
                else:
                    n = 1

                outcome = s.get("measurement", "") or "measurement"
                unit = s.get("unit", "") or "unitless"
                # B5b: no label AND no unit → unbound noise, never pool
                if outcome == "measurement" and unit == "unitless":
                    skipped_unbound += 1
                    continue
                key, val_c, unc_c, orig = _classify(outcome, unit, val, unc)
                grouped.setdefault(key, []).append((paper, s, val_c, unc_c, n, orig))
            except (KeyError, ValueError, TypeError) as e:
                log.debug("Skipping single measurement from %s: %s", pid, e)

    # Source 2: mean_sd_groups with m2=None (old regex format) — same
    # unit gate + canonicalization as Source 1 (unified 2026-08-15)
    for ex in extractions:
        pid = ex.get("paper_id") or ""
        paper = by_id.get(pid)
        if not paper:
            continue
        groups = ex.get("effect_sizes", {}).get("mean_sd_groups", [])
        for g in groups:
            if g.get("m2") is not None:
                continue  # paired — handled by collect_continuous_effects
            try:
                val = g.get("m1")
                if val is None:
                    continue
                val = float(val)
                unc = g.get("sd1")
                if unc is not None:
                    unc = float(unc)
                n = int(g.get("n") or 1)

                outcome = g.get("outcome", "") or "measurement"
                unit = g.get("unit", "") or "unitless"
                if outcome == "measurement" and unit == "unitless":
                    skipped_unbound += 1
                    continue
                key, val_c, unc_c, orig = _classify(outcome, unit, val, unc)
                grouped.setdefault(key, []).append((paper, g, val_c, unc_c, n, orig))
            except (KeyError, ValueError, TypeError) as e:
                log.debug("Skipping single measurement from %s: %s", pid, e)

    out: list[StudyEffect] = []
    for (outcome, unit), rows in sorted(grouped.items()):
        if len(rows) < 2:
            log.info(
                "Unit gate: skipping singleton group %s (%s) — k=1, no pooling",
                outcome,
                unit,
            )
            continue
        converted = sum(1 for r in rows if r[5] and r[5] != unit)
        conv_note = f" [{converted} converted to base unit]" if converted else ""
        log.info(
            "Unit gate: pooling %d values for %s (%s)%s",
            len(rows),
            outcome,
            unit,
            conv_note,
        )
        for paper, s, val, unc, n, orig_unit in rows:
            pid = paper.primary_id
            if unc is not None and unc > 0:
                v = (unc / (n**0.5 if n > 1 else 1.0)) ** 2
            else:
                v = 1.0  # no uncertainty → equal weighting
            _eff_obj = StudyEffect(
                paper_id=pid,
                doi=paper.doi,
                name=(paper.title or pid)[:50],
                effect=val,
                variance=v,
                ci_lower=val - 1.96 * (v**0.5) if unc else val,
                ci_upper=val + 1.96 * (v**0.5) if unc else val,
                scale="value",
                scale_label=f"{outcome} ({unit})",
                subgroup=outcome[:40],
                notes=f"n={n}, unit={unit}"
                + (f", orig={orig_unit}" if orig_unit != unit else "")
                + (f", ±{unc}" if unc else ""),
            )
            _eff_obj._year = paper.year  # decade decomposition input (Phase 3)
            out.append(_eff_obj)
    if skipped_unbound:
        log.info(
            "Unit gate: dropped %d unbound numbers (no unit AND no label)",
            skipped_unbound,
        )
    return out


def collect_dichotomous_effects(
    extractions: list[dict], papers: list[PaperRecord], measure: str = "OR"
) -> list[StudyEffect]:
    """Build StudyEffects from event counts (n/N). Uses OR or RR.
    Each paper needs ≥2 event-count entries (intervention vs control).
    measure: 'OR' | 'RR'."""
    by_id = {p.primary_id: p for p in papers}
    out: list[StudyEffect] = []
    for ex in extractions:
        pid = ex.get("paper_id") or ""
        paper = by_id.get(pid)
        if not paper:
            continue
        events = ex.get("effect_sizes", {}).get("event_counts", [])
        if len(events) < 2:
            continue
        e1, e2 = events[0], events[1]
        try:
            a = int(e1["events"])
            b = int(e1["total"]) - a
            c = int(e2["events"])
            d = int(e2["total"]) - c
            if a + b == 0 or c + d == 0 or a < 0 or c < 0:
                continue
            if measure == "OR":
                value = odds_ratio(a, b, c, d)
                eff = log_transform_or(value)
                v = variance_log_or(a, b, c, d)
                label = f"log(OR), back-transform OR={value:.3f}"
            else:
                value = risk_ratio(a, b, c, d)
                eff = log_transform_rr(value)
                v = variance_log_rr(a, b, c, d)
                label = f"log(RR), back-transform RR={value:.3f}"
            se = se_from_variance(v)
            ci = confidence_interval(eff, se)
            out.append(
                StudyEffect(
                    paper_id=pid,
                    doi=paper.doi,
                    name=(paper.title or pid)[:50],
                    effect=float(eff),
                    variance=float(v),
                    ci_lower=float(ci[0]),
                    ci_upper=float(ci[1]),
                    scale=f"log_{measure.lower()}",
                    scale_label=label,
                    subgroup=str(paper.year or ""),
                    notes=f"2x2=({a},{b},{c},{d})",
                )
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Skipping dichotomous from %s: %s", pid, e)
    return out


def collect_precomputed_effects(
    extractions: list[dict], papers: list[PaperRecord]
) -> list[StudyEffect]:
    """Build StudyEffects from pre-computed Cohen's d or OR/RR/HR in abstracts."""
    by_id = {p.primary_id: p for p in papers}
    out: list[StudyEffect] = []
    for ex in extractions:
        pid = ex.get("paper_id") or ""
        paper = by_id.get(pid)
        if not paper:
            continue
        eff_data = ex.get("effect_sizes", {})
        # Cohen's d pre-computed
        for d in eff_data.get("cohens_d", []):
            try:
                d_val = float(d)
                # Approximate variance with n1=n2=30 (placeholder; LLM should refine)
                v = variance_d(30, 30, d_val)
                se = se_from_variance(v)
                ci = confidence_interval(d_val, se)
                out.append(
                    StudyEffect(
                        paper_id=pid,
                        doi=paper.doi,
                        name=(paper.title or pid)[:50],
                        effect=d_val,
                        variance=float(v),
                        ci_lower=float(ci[0]),
                        ci_upper=float(ci[1]),
                        scale="d",
                        scale_label="Cohen's d (from abstract)",
                        subgroup=str(paper.year or ""),
                        notes="variance approximated with n1=n2=30",
                    )
                )
            except (ValueError, TypeError):
                continue
        # OR/RR/HR with CI
        for es in eff_data.get("effect_sizes", []):
            try:
                val = float(es.get("value", 0))
                lo = float(es.get("ci_lower", 0)) if es.get("ci_lower") else None
                hi = float(es.get("ci_upper", 0)) if es.get("ci_upper") else None
                if lo and hi and hi > lo and val > 0:
                    log_val = math.log(val)
                    se = (
                        (math.log(hi) - math.log(lo)) / (2 * 1.959964)
                        if lo > 0
                        else 0.5
                    )
                    v = se * se
                    out.append(
                        StudyEffect(
                            paper_id=pid,
                            doi=paper.doi,
                            name=(paper.title or pid)[:50],
                            effect=log_val,
                            variance=float(v),
                            ci_lower=math.log(lo),
                            ci_upper=math.log(hi),
                            scale="log_or",
                            scale_label=f"log(OR/RR/HR) back-transform={val:.3f}",
                            subgroup=str(paper.year or ""),
                            notes=f"CI: [{lo}, {hi}]",
                        )
                    )
            except (ValueError, TypeError):
                continue
    return out


# =============================================================================
# Pooling
# =============================================================================
@dataclass
class MetaAnalysisResult:
    scale: str
    scale_label: str
    k: int
    effects: list[dict]
    pooled_fixed: dict | None
    pooled_random: dict | None
    heterogeneity: dict
    interpretation: str
    forest_plot_ascii: str
    publication_bias: dict | None = None  # Egger's test + trim-and-fill
    subgroup_results: list[dict] = field(default_factory=list)
    leave_one_out: list[dict] = field(default_factory=list)  # influence analysis


def pool_effects(
    studies: list[StudyEffect],
    model: str = "random",
    tau2_method: str = "reml",
    hksj: bool = True,
) -> dict:
    effects = [s.effect for s in studies]
    variances = [s.variance for s in studies]
    if len(studies) < 2:
        raise ValueError("need ≥2 studies for pooling")
    het = heterogeneity(effects, variances)
    if model == "random":
        pooled = pool_random_advanced(
            effects, variances, tau2_method=tau2_method, hksj=hksj
        )
    else:
        pooled = pool_fixed(effects, variances)
    out = {
        "model": model,
        "effect": pooled.effect,
        "se": pooled.se,
        "ci_lower": pooled.ci_lower,
        "ci_upper": pooled.ci_upper,
        "z": pooled.z_score,
        "p_value": pooled.p_value,
        "weights": pooled.weights,
        "heterogeneity": {
            "q": het.q,
            "df": het.df,
            "p_value": het.p_value,
            "i_squared": het.i_squared,
            "tau_squared": het.tau_squared,
            "h_squared": het.h_squared,
            "interpretation": interpret_i_squared(het.i_squared),
        },
    }
    if model == "random":
        out["tau2_method"] = pooled.tau2_method
        out["tau_squared_model"] = pooled.tau_squared
        out["ci_method"] = pooled.ci_method
        if pooled.pi_lower is not None:
            out["prediction_interval"] = [pooled.pi_lower, pooled.pi_upper]
    return out


def grade_certainty(
    k: int, i2: float, effect: float, ci_lower: float, ci_upper: float
) -> dict:
    """GRADE-style certainty downgrade (deterministic rules, 2026-08-15).

    Starts High; downgrades one level per rule fired:
      - imprecision: k < 10
      - inconsistency: I² > 75
      - very wide CI: CI spans zero AND upper/lower ratio > 4 (or crosses
        zero when all effects should be positive-scale)
    Floors at "very low". This is a SCREENING rating for triage, not a
    substitute for full GRADE (RoB/indirectness need human assessment —
    noted in output).
    """
    level = 0
    reasons = []
    if k < 10:
        level += 1
        reasons.append(f"imprecision (k={k} < 10)")
    if i2 > 75:
        level += 1
        reasons.append(f"inconsistency (I²={i2:.0f}% > 75)")
    if ci_lower is not None and ci_upper is not None:
        ratio = (abs(ci_upper) + 1e-12) / (max(abs(ci_lower), 1e-12))
        if ci_lower < 0 < ci_upper and ratio > 4:
            level += 1
            reasons.append(f"very wide CI (crosses zero, ratio {ratio:.1f})")
    labels = ["high", "moderate", "low", "very low"]
    return {
        "certainty": labels[min(level, 3)],
        "downgrades": reasons,
        "note": "screening rating; full GRADE requires RoB + indirectness assessment",
    }


def run_meta_analysis(
    studies: list[StudyEffect],
    model: str = "random",
    do_subgroups: bool = True,
    tau2_method: str = "reml",
    hksj: bool = True,
) -> MetaAnalysisResult:
    """Pool + heterogeneity + forest + optional subgroups.

    Defaults changed 2026-08-14 (Cochrane MECIR-aligned, intentional change):
    τ² estimator DL → REML, z-CI → HKSJ. Escape hatches: --tau2 dl, --no-hksj.
    """
    if not studies:
        return MetaAnalysisResult(
            scale="none",
            scale_label="(no effects)",
            k=0,
            effects=[],
            pooled_fixed=None,
            pooled_random=None,
            heterogeneity={},
            interpretation="no studies",
            forest_plot_ascii="```text\n(no studies)\n```",
        )
    if len(studies) == 1:
        s = studies[0]
        return MetaAnalysisResult(
            scale=s.scale,
            scale_label=s.scale_label,
            k=1,
            effects=[asdict(s)],
            pooled_fixed=None,
            pooled_random=None,
            heterogeneity={
                "q": 0,
                "df": 0,
                "i_squared": 0,
                "interpretation": "n/a (1 study)",
            },
            interpretation="single study — no pooling",
            forest_plot_ascii="```text\n(single study — no pooling)\n```",
        )
    try:
        fixed = pool_effects(studies, model="fixed")
        random = pool_effects(
            studies, model="random", tau2_method=tau2_method, hksj=hksj
        )
    except Exception as e:
        log.error("Pooling failed: %s", e)
        return MetaAnalysisResult(
            scale=studies[0].scale,
            scale_label=studies[0].scale_label,
            k=len(studies),
            effects=[asdict(s) for s in studies],
            pooled_fixed=None,
            pooled_random=None,
            heterogeneity={},
            interpretation=f"pooling failed: {e}",
            forest_plot_ascii="```text\n(pooling failed)\n```",
        )
    chosen = random if model == "random" else fixed
    # Forest plot
    studies_render = [
        {
            "name": s.name,
            "effect": s.effect,
            "ci_lower": s.ci_lower,
            "ci_upper": s.ci_upper,
            "weight": chosen["weights"][i] / sum(chosen["weights"]),
        }
        for i, s in enumerate(studies)
    ]
    forest = render_forest_plot(
        studies_render,
        pooled={
            "effect": chosen["effect"],
            "ci_lower": chosen["ci_lower"],
            "ci_upper": chosen["ci_upper"],
        },
        effect_label=studies[0].scale_label,
    )
    interpretation = (
        f"Pooled effect ({model}): {chosen['effect']:.3f} "
        f"(95% CI {chosen['ci_lower']:.3f} to {chosen['ci_upper']:.3f}), "
        f"p={chosen['p_value']:.4f}. "
        f"I²={chosen['heterogeneity']['i_squared']:.1f}% "
        f"({chosen['heterogeneity']['interpretation']}), "
        f"Q({chosen['heterogeneity']['df']})={chosen['heterogeneity']['q']:.2f}, "
        f"p={chosen['heterogeneity']['p_value']:.3f}."
    )
    # Subgroup analysis
    subgroup_results: list[dict] = []
    # Phase 3: I²>75% on single-measure pools → decompose by year-decade
    # (calibration-era proxy) so heavy heterogeneity is EXPLAINED, not just
    # flagged. Labels carry the decade; downstream tables stay honest.
    if studies and studies[0].scale == "value" and len(studies) >= 4:
        try:
            i2 = chosen["heterogeneity"].get("i_squared", 0.0)
        except (KeyError, TypeError):
            i2 = 0.0
        if isinstance(i2, (int, float)) and i2 > 75.0:
            from collections import defaultdict as _dd

            by_dec: dict[str, list[StudyEffect]] = _dd(list)
            for st in studies:
                yr = None
                for eff_d in st.notes or "":
                    pass
                yr = getattr(st, "_year", None)
                if yr is None:
                    # recover from notes 'n=.., unit=..' — not present; use doi year guess via paper title? keep simple:
                    by_dec["mixed"].append(st)
                else:
                    by_dec[str((int(yr) // 10) * 10) + "s"].append(st)
            if len(by_dec) >= 2 and "mixed" not in by_dec:
                for lbl, grp in sorted(by_dec.items()):
                    if len(grp) < 2:
                        continue
                    pr = pool_effects(
                        [g for g in grp],
                        model=model,
                        tau2_method=tau2_method,
                        hksj=hksj,
                    )
                    subgroup_results.append(
                        {
                            "label": f"decade {lbl}",
                            "k": len(grp),
                            "effect": pr["effect"],
                            "ci_lower": pr["ci_lower"],
                            "ci_upper": pr["ci_upper"],
                            "source": "year-decade decomposition (Phase 3)",
                        }
                    )
    if do_subgroups and len(studies) >= 4:
        from collections import defaultdict

        groups: dict[str, list[StudyEffect]] = defaultdict(list)
        for s in studies:
            if s.subgroup:
                groups[s.subgroup].append(s)
        if len(groups) >= 2:
            from _stats import subgroup_analysis

            try:
                sub_eff = subgroup_analysis(
                    [s.effect for s in studies],
                    [s.variance for s in studies],
                    [s.subgroup for s in studies],
                )
                for g, pe in sub_eff.subgroup_effects.items():
                    subgroup_results.append(
                        {
                            "group": g,
                            "effect": pe.effect,
                            "ci_lower": pe.ci_lower,
                            "ci_upper": pe.ci_upper,
                            "k": pe.k,
                            "pooled_model": pe.model,
                        }
                    )
                subgroup_results.append(
                    {
                        "group": "Q_BETWEEN",
                        "effect": sub_eff.q_between,
                        "p_value": sub_eff.p_between,
                    }
                )
            except Exception as e:
                log.warning("Subgroup analysis failed: %s", e)
    # Publication bias: Egger's test + trim-and-fill (requires ≥3 studies)
    pub_bias_data: dict | None = None
    if len(studies) >= 3:
        try:
            effects_arr = [s.effect for s in studies]
            variances_arr = [s.variance for s in studies]
            pb = egger_test(effects_arr, variances_arr)
            tf = trim_and_fill(effects_arr, variances_arr)
            pub_bias_data = {
                "egger_intercept": pb.egger_intercept,
                "egger_se": pb.egger_se,
                "egger_p_value": pb.egger_p_value,
                "begg_tau": pb.begg_tau,
                "interpretation": pb.interpretation,
                "funnel_plot_data": pb.funnel_plot_data,
                "trim_and_fill": tf,
            }
            # Selection model (Vevea-Hedges 3-PSM) + p-curve — complements
            # Egger/T&F per Cochrane Handbook 6.5.4 recommendation
            try:
                from _stats import p_curve_test, vevea_hedges_selection_model

                vh = vevea_hedges_selection_model(effects_arr, variances_arr)
                if vh is not None:
                    pub_bias_data["selection_model"] = vh
                pub_bias_data["p_curve"] = p_curve_test(effects_arr, variances_arr)
            except Exception as e:  # noqa: BLE001 — optional diagnostics
                log.debug("Selection model / p-curve failed: %s", e)
        except Exception as e:
            log.warning("Egger test failed: %s", e)
            pub_bias_data = {"error": str(e)}
    # Leave-one-out influence analysis (k ≥ 4)
    loo_rows: list[dict] = []
    if len(studies) >= 4 and model == "random":
        try:
            loo_rows = leave_one_out(
                [s.effect for s in studies],
                [s.variance for s in studies],
                tau2_method=tau2_method,
                hksj=hksj,
            )
        except Exception as e:
            log.warning("Leave-one-out analysis failed: %s", e)
    return MetaAnalysisResult(
        scale=studies[0].scale,
        scale_label=studies[0].scale_label,
        k=len(studies),
        effects=[asdict(s) for s in studies],
        pooled_fixed=fixed,
        pooled_random=random,
        heterogeneity=chosen["heterogeneity"],
        interpretation=interpretation,
        forest_plot_ascii=forest,
        publication_bias=pub_bias_data,
        subgroup_results=subgroup_results,
        leave_one_out=loo_rows,
    )


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="meta_analyze",
        description="Meta-analysis pooling on extracted effect sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("extractions", type=Path, help="extracted.json from extract.py")
    p.add_argument("verified", type=Path, help="verified.json from verify.py")
    p.add_argument(
        "--measure",
        choices=["continuous", "dichotomous", "single", "auto"],
        default="auto",
        help="effect measure (default: auto-detect)",
    )
    p.add_argument(
        "--dichotomous-type",
        choices=["OR", "RR"],
        default="OR",
        help="OR or RR for dichotomous data",
    )
    p.add_argument(
        "--correction",
        choices=["hedges", "cohen"],
        default="hedges",
        help="small-sample correction for continuous (default Hedges g)",
    )
    p.add_argument(
        "--model",
        choices=["fixed", "random"],
        default="random",
        help="pooling model (default random)",
    )
    p.add_argument(
        "--tau2",
        choices=["dl", "reml", "pm"],
        default="reml",
        help="tau-squared estimator for random-effects (default reml — Cochrane "
        "MECIR preferred; dl = DerSimonian-Laird legacy, pm = Paule-Mandel)",
    )
    p.add_argument(
        "--hksj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hartung-Knapp-Sidik-Jonkman CI for random-effects (default on — "
        "preferred for small k; use --no-hksj for classic z CI)",
    )
    p.add_argument(
        "--no-subgroups",
        action="store_true",
        help="skip subgroup analysis by year/design",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("research_outputs/meta.json")
    )
    p.add_argument(
        "--report", type=Path, default=Path("research_outputs/meta_report.md")
    )
    p.add_argument("--forest", type=Path, default=Path("research_outputs/forest.mmd"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    papers = load_corpus(args.verified)
    ex_data = json.loads(args.extractions.read_text())
    extractions = ex_data.get("extractions", [])
    log.info("Loaded %d papers, %d extractions", len(papers), len(extractions))

    # Collect effects
    continuous = collect_continuous_effects(
        extractions, papers, correction=args.correction
    )
    dichotomous = collect_dichotomous_effects(
        extractions, papers, measure=args.dichotomous_type
    )
    precomputed = collect_precomputed_effects(extractions, papers)
    singles = collect_single_measurements(extractions, papers)
    log.info(
        "Effects: %d continuous, %d dichotomous, %d pre-computed, %d single-measurements",
        len(continuous),
        len(dichotomous),
        len(precomputed),
        len(singles),
    )

    if args.measure == "continuous":
        studies = continuous
    elif args.measure == "dichotomous":
        studies = dichotomous
    elif args.measure == "single":
        studies = singles
    else:  # auto
        # Prefer continuous (clinical), else singles (non-clinical), else dichotomous, else pre-computed
        studies = continuous or singles or dichotomous or precomputed

    if not studies:
        print("No effect sizes found in extractions. Skipping meta-analysis.")
        return 1

    # B5b (pool level): single measurements span incommensurable units —
    # one global pool is invalid even when collection grouped them.
    # Partition by scale_label (measurement+unit) and pool WITHIN each group.
    unit_pools: list[dict] = []
    if args.measure in ("single", "auto") and studies and studies[0].scale == "value":
        groups: dict[str, list[StudyEffect]] = {}
        for s in studies:
            groups.setdefault(s.scale_label, []).append(s)
        pools = [(lbl, g) for lbl, g in sorted(groups.items()) if len(g) >= 2]
        if not pools:
            print(
                "No homogeneous (measurement, unit) group has k ≥ 2 — "
                "cross-unit pooling refused. Nothing to pool."
            )
            return 1
        # R3 phenomenon screen: unit-commensurable is necessary but not
        # sufficient. Experimental run pressures (piston-cylinder calibration
        # conditions, vacuum/ambient ~0 kbar) are a DIFFERENT phenomenon than
        # magma-storage barometry; pooling them dragged the pressure pool
        # toward 0 in the E2E run. Screen by physical plausibility band for
        # the known crustal-storage quantities; excluded rows are LOGGED with
        # counts (never silently dropped), and only storage-scale pools keep
        # the clean label. Bands are documented earth-science priors:
        #   storage T 650-1400 °C (silicate magmas, anhydrous to hydrous)
        #   storage P 0.3-15 kbar (mid-crust to Moho; >15 = mantle/xenolith)
        # Other quantities pass through unscreened (no band asserted).
        _PHENOMENON_BANDS: dict[tuple[str, str], tuple[float, float]] = {
            ("temperature", "°C"): (650.0, 1400.0),
            ("pressure", "kbar"): (0.3, 15.0),
        }
        screened_pools: list[tuple[str, list[StudyEffect]]] = []
        for lbl, g in pools:
            base_unit = lbl.rsplit("(", 1)[-1].rstrip(")")
            meas = lbl.rsplit("(", 1)[0].strip()
            band = _PHENOMENON_BANDS.get((meas, base_unit))
            if band is None:
                screened_pools.append((lbl, g))
                continue
            kept = [s for s in g if band[0] <= s.effect <= band[1]]
            n_out = len(g) - len(kept)
            if n_out:
                print(
                    f"Phenomenon screen [{lbl}]: {n_out}/{len(g)} rows outside "
                    f"storage band {band[0]}-{band[1]} — excluded from pool "
                    f"(logged; experimental/vacuum run conditions, not storage "
                    f"estimates)"
                )
            if len(kept) >= 2:
                screened_pools.append((lbl, kept))
        pools = screened_pools
        if not pools:
            print("All pools emptied by phenomenon screen — nothing commensurable.")
            return 1
        pools.sort(key=lambda t: -len(t[1]))
        for lbl, g in pools:
            try:
                r = run_meta_analysis(
                    g,
                    model=args.model,
                    do_subgroups=False,
                    tau2_method=args.tau2,
                    hksj=args.hksj,
                )
                _pr = r.pooled_random or {}
                _het = _pr.get("heterogeneity") or {}
                _grade = grade_certainty(
                    len(g),
                    _het.get("i_squared", 0.0) or 0.0,
                    _pr.get("effect", 0.0) or 0.0,
                    _pr.get("ci_lower", 0.0) or 0.0,
                    _pr.get("ci_upper", 0.0) or 0.0,
                )
                unit_pools.append(
                    {"group": lbl, "k": len(g), "certainty": _grade, **asdict(r)}
                )
                pr = r.pooled_random if args.model == "random" else r.pooled_fixed
                assert pr is not None  # run_meta_analysis sets both pools
                print(
                    f"Unit pool [{lbl}] k={len(g)}: effect {pr['effect']:.3f} "
                    f"({pr['ci_lower']:.3f} to {pr['ci_upper']:.3f}), "
                    f"I²={pr['heterogeneity']['i_squared']:.1f}%"
                )
            except Exception as e:  # noqa: BLE001 — one bad group must not kill others
                log.warning("Unit pool %s failed: %s", lbl, e)
        # headline = largest group; others preserved in payload
        result = None
        headline = unit_pools[0]
        for up in unit_pools:
            if up["k"] > headline["k"]:
                headline = up
        result = run_meta_analysis(
            [s for s in studies if s.scale_label == headline["group"]],
            model=args.model,
            do_subgroups=not args.no_subgroups,
            tau2_method=args.tau2,
            hksj=args.hksj,
        )
        payload = asdict(result)
        payload["unit_pools"] = unit_pools
        payload["unit_pool_note"] = (
            f"Single-measurement data partitioned into {len(unit_pools)} "
            f"commensurable (measurement, unit) pools; headline = largest "
            f"({headline['group']}, k={headline['k']}). Cross-unit pooling "
            f"refused (B5b)."
        )
        payload["meta"] = {
            "extractions": str(args.extractions),
            "verified": str(args.verified),
            "measure": args.measure,
            "dichotomous_type": args.dichotomous_type,
            "correction": args.correction,
            "model": args.model,
            "tau2_method": args.tau2,
            "hksj": args.hksj,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _write_outputs(args, result, payload, unit_pools)
        return 0

    result = run_meta_analysis(
        studies,
        model=args.model,
        do_subgroups=not args.no_subgroups,
        tau2_method=args.tau2,
        hksj=args.hksj,
    )
    payload = asdict(result)
    payload["meta"] = {
        "extractions": str(args.extractions),
        "verified": str(args.verified),
        "measure": args.measure,
        "dichotomous_type": args.dichotomous_type,
        "correction": args.correction,
        "model": args.model,
        "tau2_method": args.tau2,
        "hksj": args.hksj,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_outputs(args, result, payload, [])
    return 0


def _write_outputs(args, result, payload: dict, unit_pools: list[dict]) -> None:
    """Write meta.json + report.md + forest. Shared by single-unit and
    unit-pooled paths (B5b)."""
    from _artifact import save_artifact

    save_artifact(payload, args.output)
    print(f"Wrote meta.json → {args.output}")
    # Build report — include publication bias if computed
    pub_bias_section = ""
    if result.publication_bias and "error" not in result.publication_bias:
        pb = result.publication_bias
        pub_bias_section = (
            f"\n## Publication bias\n\n"
            f"- **Egger's test**: intercept = {pb['egger_intercept']:.3f} ± {pb['egger_se']:.3f}, "
            f"p = {pb['egger_p_value']:.4f}\n"
            f"- **Begg's tau**: {pb['begg_tau']:.3f}\n"
            f"- **Interpretation**: {pb['interpretation']}\n"
        )
        tf = pb.get("trim_and_fill") or {}
        if tf.get("n_filled", 0) > 0:
            pub_bias_section += (
                f"- **Trim-and-fill**: {tf['n_filled']} studies filled, "
                f"adjusted effect = {tf.get('adjusted_effect', 0):.3f} "
                f"(shift = {tf.get('shift', 0):.3f})\n"
            )
    elif result.publication_bias and "error" in result.publication_bias:
        pub_bias_section = f"\n## Publication bias\n\nEgger test failed: {result.publication_bias['error']}\n"
    loo_section = ""
    if result.leave_one_out:
        names = [s["name"] for s in result.effects]
        rows = "\n".join(
            f"- omit **{names[r['omit_index']]}**: effect {r['effect']:.3f} "
            f"({r['ci_lower']:.3f} to {r['ci_upper']:.3f}), shift {r['shift']:+.3f}"
            for r in result.leave_one_out[:5]
        )
        loo_section = f"\n## Leave-one-out influence (top 5)\n\n{rows}\n"
    args.report.write_text(
        f"# Meta-Analysis Report\n\n{result.scale_label}\n\n"
        f"**k = {result.k}**\n\n"
        f"## Interpretation\n\n{result.interpretation}\n\n"
        f"## Forest plot (ASCII)\n\n{result.forest_plot_ascii}\n"
        f"{pub_bias_section}"
        f"{loo_section}"
    )
    print(f"Wrote meta_report.md → {args.report}")
    args.forest.parent.mkdir(parents=True, exist_ok=True)
    args.forest.write_text(result.forest_plot_ascii)
    print(f"Wrote forest ASCII → {args.forest}")
    # Matplotlib publication-grade forest + funnel plots
    if result.k >= 2 and result.pooled_random:
        try:
            studies_render = [
                {
                    "name": s["name"],
                    "effect": s["effect"],
                    "ci_lower": s["ci_lower"],
                    "ci_upper": s["ci_upper"],
                    "weight": w / sum(result.pooled_random["weights"]),
                }
                for s, w in zip(result.effects, result.pooled_random["weights"])
            ]
            pooled_render = {
                "effect": result.pooled_random["effect"],
                "ci_lower": result.pooled_random["ci_lower"],
                "ci_upper": result.pooled_random["ci_upper"],
            }
            forest_png = args.output.parent / "forest.png"
            render_forest_plot_png(
                studies_render,
                pooled=pooled_render,
                effect_label=result.scale_label,
                output_path=forest_png,
            )
            print(f"Wrote forest PNG → {forest_png}")
        except Exception as e:
            log.warning("Forest PNG render failed: %s", e)
    if result.publication_bias and result.publication_bias.get("funnel_plot_data"):
        try:
            funnel_png = args.output.parent / "funnel.png"
            render_funnel_plot_png(
                result.publication_bias["funnel_plot_data"],
                egger_intercept=result.publication_bias.get("egger_intercept"),
                pooled_effect=result.pooled_random["effect"]
                if result.pooled_random
                else None,
                output_path=funnel_png,
            )
            print(f"Wrote funnel PNG → {funnel_png}")
        except Exception as e:
            log.warning("Funnel PNG render failed: %s", e)
    print(f"\n{result.interpretation}")


if __name__ == "__main__":
    sys.exit(main())

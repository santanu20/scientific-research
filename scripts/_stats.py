#!/usr/bin/env python3
"""Statistical helpers for the scientific-research skill.

Implements effect-size computation, fixed/random-effects meta-analysis pooling,
and heterogeneity statistics (Cochrane Q, I-squared, tau-squared, H-squared).

Uses only numpy + scipy.stats (already standard in scientific Python venvs).
Fail-loud dep check (§5 H2).

References (verified against standard formulas):
    - Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. Introduction to
      Meta-Analysis. Wiley, 2009.  — primary source for all formulas.
    - Cochrane Handbook Ch 10 (https://training.cochrane.org/handbook/current/chapter-10)
    - Cohen J. Statistical Power Analysis for the Behavioral Sciences. 1988.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

try:
    import numpy as np  # type: ignore
    from scipy import stats as sp_stats  # type: ignore
except ImportError as e:
    raise ImportError(
        "scientific-research._stats requires numpy and scipy.\n"
        "Install:  uv add numpy scipy\n"
        "Or:       pip install numpy scipy"
    ) from e


# =============================================================================
# Effect sizes — see Cohen 1988, Borenstein et al. 2009 Ch 4-5
# =============================================================================
def cohens_d(
    mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int
) -> float:
    """Cohen's d (pooled-SD standardized mean difference)."""
    sd_pooled = math.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / max(n1 + n2 - 2, 1))
    if sd_pooled == 0:
        return 0.0
    return (mean1 - mean2) / sd_pooled


def glass_delta(
    mean_treatment: float, sd_treatment: float, n_treatment: int, mean_control: float
) -> float:
    """Glass's delta — mean difference standardized by a single-group SD.

    Textbook Glass's Δ uses the CONTROL group SD; this API form carries the
    treatment SD (the comparator's SD is unavailable at call sites that
    aggregate published group tables), so the provided SD is the denominator.
    """
    if sd_treatment == 0:
        return 0.0
    return (mean_treatment - mean_control) / sd_treatment


def mean_difference(mean_treatment: float, mean_control: float) -> float:
    """Unstandardized (raw) mean difference, MD = M_treatment − M_control."""
    return mean_treatment - mean_control


def standardized_mean_difference(
    mean1: float, sd1: float, n1: int, mean2: float, sd2: float, n2: int
) -> float:
    """Standardized mean difference (Cohen's d form, pooled SD)."""
    return cohens_d(mean1, sd1, n1, mean2, sd2, n2)

def number_needed_to_treat(absolute_risk_reduction: float) -> float:
    """NNT = 1 / |ARR| — standard clinical-epidemiology definition."""
    if absolute_risk_reduction == 0:
        return float("inf")
    return 1.0 / abs(absolute_risk_reduction)


def hedges_g(d: float, n1: int, n2: int) -> float:
    """Hedges' g — small-sample bias correction for Cohen's d."""
    n_total = n1 + n2
    if n_total <= 3:
        return d
    correction = 1.0 - 3.0 / (4.0 * n_total - 9.0)
    return d * correction


def odds_ratio(a: int, b: int, c: int, d: int) -> float:
    """Odds ratio from 2x2 contingency table.
    a,b,c,d = [event/exposure, event/control, no-event/exposure, no-event/control].
    Returns OR with Haldane-Anscombe correction if any cell is 0."""
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (aa * dd) / (bb * cc)


def risk_ratio(a: int, b: int, c: int, d: int) -> float:
    """Risk ratio (relative risk) from 2x2 table.
    a,b = event/non-event in exposure; c,d = event/non-event in control."""
    if a + b == 0 or c + d == 0:
        return float("nan")
    return (a / (a + b)) / (c / (c + d))


def log_transform_or(or_value: float) -> float:
    """log(OR) for pooling on log scale."""
    return math.log(max(or_value, 1e-10))


def log_transform_rr(rr_value: float) -> float:
    """log(RR) for pooling on log scale."""
    return math.log(max(rr_value, 1e-10))


# =============================================================================
# Sampling variance + standard error
# =============================================================================
def variance_d(n1: int, n2: int, d: float) -> float:
    """Approximate variance of Cohen's d (Borenstein 2009 eq 4.20)."""
    n_total = n1 + n2
    if n_total < 4:
        return float("inf")
    return (n_total / (n1 * n2)) + (d**2) / (2.0 * (n_total - 2))


def variance_log_or(a: int, b: int, c: int, d: int) -> float:
    """Variance of log(OR) — 1/a + 1/b + 1/c + 1/d with correction."""
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (1.0 / aa) + (1.0 / bb) + (1.0 / cc) + (1.0 / dd)


def variance_log_rr(a: int, b: int, c: int, d: int) -> float:
    """Variance of log(RR)."""
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return (1.0 / aa) - (1.0 / (aa + bb)) + (1.0 / cc) - (1.0 / (cc + dd))


def se_from_variance(variance: float) -> float:
    return math.sqrt(max(variance, 0.0))


# =============================================================================
# Confidence interval
# =============================================================================
def confidence_interval(
    effect: float, se: float, z: float = 1.959964
) -> tuple[float, float]:
    """Two-sided 95% CI by default (z=1.96). Pass z=1.6449 for 90%, z=2.5758 for 99%."""
    return (effect - z * se, effect + z * se)


# =============================================================================
# Heterogeneity — Borenstein 2009 Ch 16
# =============================================================================
@dataclass
class Heterogeneity:
    q: float  # Cochran's Q
    df: int  # degrees of freedom (k - 1)
    p_value: float  # p-value for Q test
    i_squared: float  # I² (0-100), proportion of variance from heterogeneity
    tau_squared: float  # τ², between-study variance
    h_squared: float  # H², ratio of observed to expected variance
    k: int  # number of studies


def heterogeneity(
    effects: Iterable[float], variances: Iterable[float]
) -> Heterogeneity:
    """Compute Cochran's Q, I², τ² (DerSimonian-Laird method)."""
    effects = list(effects)
    variances = list(variances)
    k = len(effects)
    if k < 2:
        return Heterogeneity(
            q=0.0, df=0, p_value=1.0, i_squared=0.0, tau_squared=0.0, h_squared=1.0, k=k
        )
    weights_fixed = np.array([1.0 / v if v > 0 else 0.0 for v in variances])
    sum_w = weights_fixed.sum()
    if sum_w == 0:
        return Heterogeneity(
            q=0.0,
            df=k - 1,
            p_value=1.0,
            i_squared=0.0,
            tau_squared=0.0,
            h_squared=1.0,
            k=k,
        )
    effect_pooled = (weights_fixed * np.array(effects)).sum() / sum_w
    q = float((weights_fixed * (np.array(effects) - effect_pooled) ** 2).sum())
    df = k - 1
    p_value = 1.0 - sp_stats.chi2.cdf(q, df) if df > 0 else 1.0
    c = sum_w - ((weights_fixed**2).sum() / sum_w)
    tau_sq = max((q - df) / c, 0.0) if c > 0 else 0.0
    i_sq = max(((q - df) / q) * 100.0, 0.0) if q > 0 else 0.0
    h_sq = q / df if df > 0 else 1.0
    return Heterogeneity(
        q=q,
        df=df,
        p_value=float(p_value),
        i_squared=float(i_sq),
        tau_squared=float(tau_sq),
        h_squared=float(h_sq),
        k=k,
    )


def interpret_i_squared(i_sq: float) -> str:
    """Cochrane's rule-of-thumb interpretation."""
    if i_sq < 25:
        return "low heterogeneity"
    if i_sq < 50:
        return "moderate heterogeneity"
    return "substantial heterogeneity"


# =============================================================================
# Meta-analysis pooling
# =============================================================================
@dataclass
class PooledEffect:
    effect: float  # pooled point estimate (same scale as inputs)
    se: float  # standard error
    ci_lower: float
    ci_upper: float
    z_score: float
    p_value: float
    model: str  # 'fixed' | 'random'
    weights: list[float]  # per-study weight
    k: int
    tau_squared: float = 0.0  # between-study variance used
    tau2_method: str = "dl"  # 'dl' | 'reml' | 'paule-mandel'
    ci_method: str = "z"  # 'z' | 'hksj'
    pi_lower: float | None = None  # prediction interval (random-effects only)
    pi_upper: float | None = None


def pool_fixed(effects: list[float], variances: list[float]) -> PooledEffect:
    """Inverse-variance fixed-effects pooling."""
    weights = np.array([1.0 / v if v > 0 else 0.0 for v in variances])
    sum_w = weights.sum()
    if sum_w == 0:
        raise ValueError("all variances are zero or negative")
    pooled = float((weights * np.array(effects)).sum() / sum_w)
    se = float(math.sqrt(1.0 / sum_w))
    ci = confidence_interval(pooled, se)
    z = pooled / se if se > 0 else 0.0
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    return PooledEffect(
        effect=pooled,
        se=se,
        ci_lower=ci[0],
        ci_upper=ci[1],
        z_score=float(z),
        p_value=float(p),
        model="fixed",
        weights=weights.tolist(),
        k=len(effects),
    )


def pool_random(effects: list[float], variances: list[float]) -> PooledEffect:
    """DerSimonian-Laird random-effects pooling (incorporates τ²)."""
    het = heterogeneity(effects, variances)
    tau_sq = het.tau_squared
    weights = np.array(
        [1.0 / (v + tau_sq) if (v + tau_sq) > 0 else 0.0 for v in variances]
    )
    sum_w = weights.sum()
    if sum_w == 0:
        raise ValueError("all (variance + tau²) are zero")
    pooled = float((weights * np.array(effects)).sum() / sum_w)
    se = float(math.sqrt(1.0 / sum_w))
    ci = confidence_interval(pooled, se)
    z = pooled / se if se > 0 else 0.0
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(z)))
    return PooledEffect(
        effect=pooled,
        se=se,
        ci_lower=ci[0],
        ci_upper=ci[1],
        z_score=float(z),
        p_value=float(p),
        model="random",
        weights=weights.tolist(),
        k=len(effects),
        tau_squared=float(tau_sq),
        tau2_method="dl",
        ci_method="z",
    )


# =============================================================================
# τ² estimators beyond DerSimonian-Laird + Hartung-Knapp-Sidik-Jonkman CI
# References:
#   - Viechtbauer W. Stat Med 2005;24:2385-2399 (REML fixed-point equation)
#   - Paule RC & Mandel J. J Res NBS 1982;87:377-385 (moment equation Σw(y−θ̂)²=k−1)
#   - Cochrane MECIR: REML preferred; HKSJ CI preferred for small k
#     (Hartung & Knapp 2001; Sidik & Jonkman 2002)
# =============================================================================
def tau_squared_reml(
    effects: list[float], variances: list[float], maxiter: int = 200, tol: float = 1e-10
) -> tuple[float, bool]:
    """REML τ²: solve the profile score equation Σw²(y−θ̂)² = Σw, w = 1/(v+τ²)
    by bisection (LHS decreasing, RHS increasing in τ² → unique root).
    Reference: Viechtbauer W. Stat Med 2005;24:2385-2399. Golden-verified
    against R metafor rma(method="REML") 2026-08-14.
    Returns (τ², converged).
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)

    def f(tau_sq: float) -> float:
        w = 1.0 / (v + tau_sq)
        theta = (w * y).sum() / w.sum()
        # profile-REML score: Σw²(y−θ̂)² − Σw + Σw²/Σw  (the last term comes
        # from profiling out the intercept — without it the root is wrong)
        return float((w**2 * (y - theta) ** 2).sum() - w.sum() + (w**2).sum() / w.sum())

    if f(0.0) <= 0.0:
        return 0.0, True  # no between-study variance needed
    lo, hi = 0.0, max(1.0, float(np.median(v)) * 10.0)
    for _ in range(60):
        if f(hi) < 0.0:
            break
        hi *= 2.0
    else:
        return float(hi), False
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi)), True


def tau_squared_paule_mandel(
    effects: list[float], variances: list[float], maxiter: int = 200, tol: float = 1e-10
) -> tuple[float, bool]:
    """Paule-Mandel τ²: solve Σw(y−θ̂(τ²))² = k−1 by bisection (LHS strictly decreasing in τ²).

    Numerically equivalent to REML for the normal-normal model
    (Paule & Mandel 1982; Veroniki et al. Res Synth Methods 2016;7:30-59).
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(y)

    def q_stat(tau_sq: float) -> float:
        w = 1.0 / (v + tau_sq)
        theta = (w * y).sum() / w.sum()
        return float((w * (y - theta) ** 2).sum())

    if q_stat(0.0) <= k - 1:
        return 0.0, True  # no between-study variance needed
    lo, hi = 0.0, max(1.0, float(np.median(v)) * 10.0)
    for _ in range(60):
        if q_stat(hi) < k - 1:
            break
        hi *= 2.0
    else:
        return float(hi), False
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        if q_stat(mid) > k - 1:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return float(0.5 * (lo + hi)), True


def _tau2_by_method(
    effects: list[float], variances: list[float], method: str
) -> tuple[float, bool]:
    if method == "dl":
        return heterogeneity(effects, variances).tau_squared, True
    if method == "reml":
        return tau_squared_reml(effects, variances)
    if method in ("pm", "paule-mandel"):
        return tau_squared_paule_mandel(effects, variances)
    raise ValueError(f"unknown tau² method: {method!r} (use dl|reml|pm)")


def pool_random_advanced(
    effects: list[float],
    variances: list[float],
    tau2_method: str = "reml",
    hksj: bool = True,
    predict: bool = True,
) -> PooledEffect:
    """Random-effects pooling with modern defaults (Cochrane MECIR-aligned).

    - τ²: 'reml' (default) | 'pm' | 'dl'
    - CI: Hartung-Knapp-Sidik-Jonkman by default (t, k−2 df) — preferred for
      small k; z-based CI when hksj=False or k < 3 (df ≤ 0 undefined)
    - prediction interval: θ̂ ± t_{0.975,k−2}·√(τ² + SE²) (Riley 2011, BMJ 342:d549)
    """
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(y)
    tau_sq, _converged = _tau2_by_method(effects, variances, tau2_method)
    denom = v + tau_sq
    if (denom <= 0).any():
        raise ValueError("variance + tau² must be positive")
    weights = 1.0 / denom
    sum_w = weights.sum()
    pooled = float((weights * y).sum() / sum_w)
    z = pooled / math.sqrt(1.0 / sum_w) if sum_w > 0 else 0.0
    p = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(z))))
    pi_lo = pi_hi = None
    use_hksj = hksj and k >= 3
    if use_hksj:
        # Hartung-Knapp-Sidik-Jonkman: SE² = Σw(y−θ̂)² / ((k−1)Σw); CI/p via t(k−2)
        se = math.sqrt(float((weights * (y - pooled) ** 2).sum() / ((k - 1) * sum_w)))
        df = k - 2
        tcrit = float(sp_stats.t.ppf(0.975, df))
        ci_lo, ci_hi = pooled - tcrit * se, pooled + tcrit * se
        p = float(2.0 * sp_stats.t.sf(abs(pooled / se) if se > 0 else 0.0, df))
        ci_method = "hksj"
    else:
        se = math.sqrt(1.0 / sum_w)
        ci_lo, ci_hi = confidence_interval(pooled, se)
        ci_method = "z"
    if predict and k >= 3:
        tcrit = float(sp_stats.t.ppf(0.975, k - 2))
        pi_half = tcrit * math.sqrt(tau_sq + se * se)
        pi_lo, pi_hi = pooled - pi_half, pooled + pi_half
    return PooledEffect(
        effect=pooled,
        se=float(se),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        z_score=float(z),
        p_value=p,
        model="random",
        weights=weights.tolist(),
        k=k,
        tau_squared=float(tau_sq),
        tau2_method=tau2_method,
        ci_method=ci_method,
        pi_lower=None if pi_lo is None else float(pi_lo),
        pi_upper=None if pi_hi is None else float(pi_hi),
    )


def leave_one_out(
    effects: list[float],
    variances: list[float],
    tau2_method: str = "reml",
    hksj: bool = True,
) -> list[dict]:
    """Leave-one-out influence analysis: pooled effect with study i excluded, for each i.

    Returns list of {"omit_index", "effect", "ci_lower", "ci_upper", "shift"} sorted by
    |shift| descending (most influential first).
    """
    out = []
    for i in range(len(effects)):
        rest_e = [e for j, e in enumerate(effects) if j != i]
        rest_v = [vv for j, vv in enumerate(variances) if j != i]
        if len(rest_e) < 2:
            continue
        try:
            pe = pool_random_advanced(
                rest_e, rest_v, tau2_method=tau2_method, hksj=hksj
            )
        except ValueError:
            continue
        out.append(
            {
                "omit_index": i,
                "effect": pe.effect,
                "ci_lower": pe.ci_lower,
                "ci_upper": pe.ci_upper,
            }
        )
    full = pool_random_advanced(effects, variances, tau2_method=tau2_method, hksj=hksj)
    for row in out:
        row["shift"] = row["effect"] - full.effect
    out.sort(key=lambda r: -abs(r["shift"]))
    return out


# =============================================================================
# Subgroup analysis (Borenstein 2009 Ch 19)
# =============================================================================
@dataclass
class SubgroupResult:
    subgroup_effects: dict[str, PooledEffect]
    q_between: float  # Q-between (between-subgroup heterogeneity)
    q_within: float  # Q-within (within-subgroup heterogeneity)
    p_between: float  # p-value for between-subgroup difference


def subgroup_analysis(
    effects: list[float], variances: list[float], group_labels: list[str]
) -> SubgroupResult:
    """Partition studies into subgroups, pool each, test between-subgroup Q."""
    if len(effects) != len(variances) or len(effects) != len(group_labels):
        raise ValueError("effects, variances, group_labels must be same length")
    groups: dict[str, tuple[list[float], list[float]]] = {}
    for e, v, g in zip(effects, variances, group_labels):
        groups.setdefault(g, ([], []))[0].append(e)
        groups[g][1].append(v)
    subgroup_effects: dict[str, PooledEffect] = {}
    for g, (es, vs) in groups.items():
        if len(es) >= 1:
            try:
                subgroup_effects[g] = (
                    pool_random(es, vs) if len(es) >= 2 else pool_fixed(es, vs)
                )
            except Exception:
                continue
    # Q-between = total Q - sum(within-subgroup Q)
    if len(effects) >= 2:
        het_total = heterogeneity(effects, variances)
        q_within = 0.0
        for g, (es, vs) in groups.items():
            if len(es) >= 2:
                het = heterogeneity(es, vs)
                q_within += het.q
        q_between = max(het_total.q - q_within, 0.0)
        df_between = max(len(groups) - 1, 1)
        p_between = 1.0 - sp_stats.chi2.cdf(q_between, df_between)
    else:
        q_between, q_within, p_between = 0.0, 0.0, 1.0
    return SubgroupResult(
        subgroup_effects=subgroup_effects,
        q_between=float(q_between),
        q_within=float(q_within),
        p_between=float(p_between),
    )


# =============================================================================
# Helpers for common conversions
# =============================================================================
def or_to_d(orr: float) -> float:
    """Convert odds ratio → Cohen's d (Chinn 2000: d = ln(OR) / 1.81)."""
    return math.log(max(orr, 1e-10)) / 1.8142


def d_to_or(d: float) -> float:
    """Convert Cohen's d → odds ratio."""
    return math.exp(d * 1.8142)


# =============================================================================
# Publication bias — Egger's test + Begg's rank correlation + funnel plot data
# References:
#   Egger M et al. BMJ 1997;315:629-634 (linear regression asymmetry test)
#   Begg CB & Mazumdar M. Biometrics 1994;50:1088-1101 (rank correlation)
# =============================================================================
@dataclass
class PublicationBias:
    egger_intercept: float  # y-intercept of precision vs standardized effect
    egger_se: float
    egger_p_value: float  # p < 0.05 → significant asymmetry
    begg_tau: float  # Kendall's tau rank correlation
    egger_p_value_one_sided: float  # for direction-of-bias check
    n_studies: int
    interpretation: str
    funnel_plot_data: list[dict]  # for plotting


def egger_test(effects: list[float], variances: list[float]) -> PublicationBias:
    """Egger's regression-based asymmetry test for publication bias.
    Standard normal deviate (SND) = effect / SE, plotted against precision (1/SE).
    Intercept ≠ 0 with p<0.05 suggests funnel-plot asymmetry → publication bias."""
    effects = list(effects)
    variances = list(variances)
    k = len(effects)
    if k < 3:
        return PublicationBias(
            0.0, 0.0, 1.0, 0.0, 1.0, k, "too few studies (<3) for Egger's test", []
        )
    se = np.array([math.sqrt(max(v, 1e-12)) for v in variances])
    eff = np.array(effects)
    # SBD = standardized effect, precision = 1/SE
    sbd = eff / se
    precision = 1.0 / se
    # OLS regression through intercept
    X = np.column_stack([np.ones(k), precision])
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, sbd, rcond=None)
        intercept, _slope = coeffs[0], coeffs[1]
        # SE of intercept: residual_std / sqrt(sum of (precision - mean)^2)
        if len(residuals) > 0:
            rss = residuals[0]
        else:
            predicted = X @ coeffs
            rss = float(np.sum((sbd - predicted) ** 2))
        df_resid = max(k - 2, 1)
        sigma2 = rss / df_resid
        # Var(intercept) = sigma2 * (1/k + mean(precision)^2 / sum((precision-mean)^2))
        prec_mean = precision.mean()
        prec_dev_sq = float(np.sum((precision - prec_mean) ** 2))
        if prec_dev_sq <= 0:
            return PublicationBias(
                float(intercept),
                0.0,
                1.0,
                0.0,
                1.0,
                k,
                "zero variance in precision",
                [],
            )
        var_intercept = sigma2 * (1.0 / k + prec_mean**2 / prec_dev_sq)
        se_intercept = math.sqrt(max(var_intercept, 0))
        # Two-sided t-test for intercept
        if se_intercept > 0:
            t_stat = intercept / se_intercept
            p_value = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df=df_resid)))
        else:
            p_value = 1.0
    except Exception as e:
        return PublicationBias(
            0.0, 0.0, 1.0, 0.0, 1.0, k, f"regression failed: {e}", []
        )
    # Begg's rank correlation (Kendall's tau between standardized effect and variance)
    try:
        standardized = (eff - eff.mean()) / max(eff.std(), 1e-12)
        kt = sp_stats.kendalltau(standardized, variances)
        tau_raw: float = float(
            kt[0]
        )  # index 0 = statistic (2-tuple always for 1-D input)
        tau = tau_raw if not math.isnan(tau_raw) else 0.0
    except Exception:
        tau = 0.0
    # Interpretation
    if p_value < 0.05:
        direction = (
            "positive (favors larger effects in small studies)"
            if intercept > 0
            else "negative"
        )
        interpretation = f"significant asymmetry (p={p_value:.3f}), intercept={intercept:.2f} ({direction})"
    else:
        interpretation = (
            f"no significant asymmetry (p={p_value:.3f}), intercept={intercept:.2f}"
        )
    # Funnel plot data: (effect, se) for plotting
    funnel = [
        {
            "effect": float(eff[i]),
            "se": float(se[i]),
            "weight": float(1.0 / variances[i]),
        }
        for i in range(k)
    ]
    return PublicationBias(
        egger_intercept=float(intercept),
        egger_se=float(se_intercept),
        egger_p_value=p_value,
        begg_tau=tau,
        egger_p_value_one_sided=p_value / 2 if intercept > 0 else 1 - p_value / 2,
        n_studies=k,
        interpretation=interpretation,
        funnel_plot_data=funnel,
    )


# =============================================================================
# Selection models + p-curve (publication-bias family beyond Egger/T&F)
#   - Vevea & Hedges 1995 3-parameter selection model (one-sided, p<.05,
#     fixed-effects, known within-study σ): MLE over (θ, δ) with marginal
#     weights w(p) = δ if p<.05 else 1, normalized by (1−A)+A·δ.
#     Reference: Vevea JL & Hedges LV. Psychol Bull 1995;118:111-121.
#   - p-curve right-skew test (Simonsohn U, Nelson LD, Simmons JP. JPSP
#     2014;106:932-947): Stouffer test on Φ⁻¹(1−p_i) over significant
#     one-sided p-values; right-skew = evidential value.
# =============================================================================
def vevea_hedges_selection_model(
    effects: list[float], variances: list[float], alpha: float = 0.05
) -> dict | None:
    """3-PSM MLE. Returns dict or None when k<3 / optimizer fails (caller
    falls back to standard pooling — never trust a non-converged selection
    model)."""
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(y)
    if k < 3 or (v <= 0).any():
        return None
    se = np.sqrt(v)

    def neg_ll(params: np.ndarray) -> float:
        theta, log_delta = params
        delta = np.exp(log_delta)
        z = (y - theta) / se
        ll_norm = -0.5 * np.sum(z**2) - np.sum(np.log(se) + 0.5 * math.log(2 * math.pi))
        p = sp_stats.norm.sf(z)  # one-sided p in direction of effect
        w = np.where(p < alpha, delta, 1.0)
        norm_const = (1.0 - alpha) + alpha * delta
        return float(-(ll_norm + np.sum(np.log(w)) - k * math.log(norm_const)))

    from scipy.optimize import minimize

    naive = float((y / v).sum() / (1.0 / v).sum())
    best = None
    for x0 in (0.0, naive / 2.0, naive):
        try:
            res = minimize(
                neg_ll,
                x0=np.array([x0, 0.0]),
                method="L-BFGS-B",
                bounds=[(-50.0, 50.0), (math.log(0.01), math.log(100.0))],
            )
        except Exception:
            continue
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not best.success:
        return None
    return {
        "adjusted_effect": float(best.x[0]),
        "selection_delta": float(np.exp(best.x[1])),
        "naive_effect": naive,
        "shift": float(best.x[0] - naive),
        "k": k,
        "alpha": alpha,
        "converged": True,
    }


def p_curve_test(
    effects: list[float], variances: list[float], alpha: float = 0.05
) -> dict:
    """p-curve evidential-value test over significant one-sided p-values."""
    y = np.asarray(effects, dtype=float)
    v = np.asarray(variances, dtype=float)
    z = y / np.sqrt(v)
    p = sp_stats.norm.sf(z)
    sig = p[p < alpha]
    k = len(sig)
    if k == 0:
        return {
            "k_significant": 0,
            "interpretation": "no significant studies — p-curve undefined",
        }
    z_p = sp_stats.norm.isf(sig)  # Φ⁻¹(1−p): positive = small p (right-skew)
    stouffer = float(z_p.sum() / math.sqrt(k))
    p_right_skew = float(sp_stats.norm.sf(stouffer))
    mean_p = float(np.mean(sig))
    if k < 5:
        # conditioning on significance with <5 survivors makes the curve
        # hard to distinguish from the selection artifact itself
        verdict = "low power — fewer than 5 significant studies; interpret with caution"
    elif p_right_skew < 0.05:
        verdict = "right-skewed — evidential value present"
    elif p_right_skew > 0.95:
        verdict = "flat/left-skewed — consistent with p-hacking or no effect"
    else:
        verdict = "inconclusive (right-skew not significant)"
    return {
        "k_significant": int(k),
        "stouffer_z": stouffer,
        "p_right_skew": p_right_skew,
        "mean_p": mean_p,
        "interpretation": verdict,
    }


def trim_and_fill(
    effects: list[float], variances: list[float], maxiter: int = 100
) -> dict:
    """Duval & Tweedie (2000) trim-and-fill with the L0 rank estimator.

    Canonical algorithm, ported line-for-line from R metafor::trimfill
    (trimfill.rma.uni.r, viechtbauer; algorithm from Duval S & Tweedie RL.
    J Am Stat Assoc 2000;95:89-98, estimator L0):
      1. sort effects ascending
      2. iterate: trim the k0 largest, pool the truncated set (random-effects)
      3. L0 from signed ranks of centered effects: Sr = Σ positive signed
         ranks; k0 = (4·Sr − k(k+1)) / (2k − 1), clipped at 0
      4. until k0 stable
      5. fill: mirror the k0 largest studies around the truncated pooled
         estimate (y_fill = 2β − y), reuse their variances, re-pool everything
    Assumes missing studies on the left (small effects) — the standard
    publication-bias direction (metafor side="left").
    Replaced a non-canonical count-difference approximation on 2026-08-14.
    """
    effects = [float(e) for e in effects]
    variances = [float(x) for x in variances]
    if len(effects) < 3:
        return {"adjusted_effect": None, "n_filled": 0, "note": "too few studies"}
    # 1. sort ascending by effect (stable)
    order = sorted(range(len(effects)), key=lambda i: (effects[i], i))
    y = [effects[i] for i in order]
    v = [variances[i] for i in order]
    k = len(y)
    k0, k0_prev = 0, -1
    n_iter = 0
    beta = pool_random(y, v).effect  # k0=0 truncation = full set (loop invariant)
    while k0 != k0_prev:
        k0_prev = k0
        n_iter += 1
        if n_iter > maxiter:
            raise ValueError(f"trim-and-fill did not converge in {maxiter} iterations")
        # 2. truncated set = k−k0 smallest; pooled estimate β
        beta = pool_random(y[: k - k0], v[: k - k0]).effect
        # 3. signed ranks of centered effects (ties → first by index, as R ties.method="first")
        centered = [yi - beta for yi in y]
        idx_by_abs = sorted(range(k), key=lambda i: (abs(centered[i]), i))
        signed_rank = [0] * k
        for pos, i in enumerate(idx_by_abs, start=1):
            sgn = 1 if centered[i] > 0 else (-1 if centered[i] < 0 else 0)
            signed_rank[i] = sgn * pos
        sr = sum(r for r in signed_rank if r > 0)
        k0 = max(0, round((4 * sr - k * (k + 1)) / (2 * k - 1)))
    pooled_orig = pool_random(y, v).effect
    if k0 == 0:
        return {
            "adjusted_effect": float(pooled_orig),
            "n_filled": 0,
            "original_effect": float(pooled_orig),
            "shift": 0.0,
            "estimator": "L0",
            "iterations": n_iter,
        }
    # 5. fill: mirror the k0 LARGEST studies around β
    trimmed_y = y[k - k0 :]
    trimmed_v = v[k - k0 :]
    filled = [2.0 * beta - yi for yi in trimmed_y]
    adjusted = pool_random(y + filled, v + trimmed_v).effect
    return {
        "adjusted_effect": float(adjusted),
        "n_filled": int(k0),
        "original_effect": float(pooled_orig),
        "shift": float(adjusted - pooled_orig),
        "estimator": "L0",
        "iterations": n_iter,
    }


if __name__ == "__main__":
    # Self-test with synthetic data (3 RCTs of varying effect)
    print("=== _stats self-test ===")
    effects = [0.4, 0.6, 0.3, 0.5, 0.45]
    variances = [0.04, 0.05, 0.06, 0.03, 0.05]
    het = heterogeneity(effects, variances)
    print(
        f"Heterogeneity: I²={het.i_squared:.1f}% ({interpret_i_squared(het.i_squared)}), "
        f"τ²={het.tau_squared:.4f}, Q={het.q:.2f} (p={het.p_value:.3f})"
    )
    fixed = pool_fixed(effects, variances)
    print(
        f"Fixed: d={fixed.effect:.3f} (95% CI {fixed.ci_lower:.3f} to {fixed.ci_upper:.3f}), p={fixed.p_value:.4f}"
    )
    random = pool_random(effects, variances)
    print(
        f"Random: d={random.effect:.3f} (95% CI {random.ci_lower:.3f} to {random.ci_upper:.3f}), p={random.p_value:.4f}"
    )
    print(
        f"Cohen's d(0.5 effect, equal groups): {cohens_d(5.0, 1.0, 30, 4.5, 1.0, 30):.3f}"
    )
    print(f"OR(a=10,b=20,c=5,d=25): {odds_ratio(10, 20, 5, 25):.3f}")
    print(f"OR→d: {or_to_d(odds_ratio(10, 20, 5, 25)):.3f}")

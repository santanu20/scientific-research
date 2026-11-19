# ADR 0005: Typed quantitative estimands and pooling compatibility

- Status: Accepted as an input contract; current pooling engine remains unchanged
- Date: 2026-11-19

## Context

A meta-analysis must not pool values merely because they are numeric. Study-level estimates need a common effect measure and estimand, compatible outcome, population, timepoint, study-design stratum, and compatible units/analysis scale. Duplicate cohorts can otherwise be double-counted. Every input should also be traceable to a source passage.

## Decision

Introduce a typed AnalysisPlan, EffectEstimate, and AnalysisDataset.

- The plan specifies effect measure, target outcome, population, timepoint, study-design family, intervention/comparator, units, fixed/random-effects model, tau-squared estimator, and confidence-interval method.
- Estimates carry point estimate, standard error, optional confidence interval, sample sizes, source ID, and one or more evidence-span IDs.
- Log ratio measures and Fisher-z correlations are explicitly represented on their analysis scales; standard errors and confidence limits must use that same scale.
- A univariate dataset requires two or more unique studies and rejects incompatible estimand dimensions or duplicate study IDs.
- A bundle cross-check requires every effect to cite evidence from its own source.

## Non-goals

This change does not calculate pooled effects, heterogeneity, prediction intervals, or small-sample corrections; it does not prove extracted estimates are correct. The existing meta-analysis implementation is not switched over in this PR. Numerical pooling parity and sensitivity analysis require a separate integration and validation PR.

## Consequences

The repository gains a scientific input gate that rejects a common class of invalid pooling before numerical computation. Multiple correlated outcomes or arms from one study require an appropriate multivariate/multilevel model and are intentionally excluded from this univariate contract.

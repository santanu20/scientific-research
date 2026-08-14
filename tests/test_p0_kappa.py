"""P0-7 Cohen's kappa tests — hand formula cross-checked vs sklearn."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_screening import cohen_kappa


def _sklearn_kappa(tp, fp, fn, tn):
    from sklearn.metrics import cohen_kappa_score

    # elementwise pairs must encode the matrix exactly:
    # tp:(1,1) fp:(1,0) fn:(0,1) tn:(0,0)
    y1 = [1] * tp + [1] * fp + [0] * fn + [0] * tn  # machine
    y2 = [1] * tp + [0] * fp + [1] * fn + [0] * tn  # human
    return cohen_kappa_score(y1, y2)


class TestCohenKappa:
    def test_matches_sklearn_on_various_matrices(self):
        cases = [
            (9, 2, 1, 10),  # typical screening outcome
            (10, 0, 0, 12),  # perfect agreement
            (5, 5, 5, 5),  # chance-level agreement → κ ≈ 0
            (0, 8, 8, 0),  # total disagreement → κ < 0
            (1, 0, 0, 99),  # highly imbalanced
        ]
        for tp, fp, fn, tn in cases:
            mine = cohen_kappa(tp, fp, fn, tn)
            theirs = _sklearn_kappa(tp, fp, fn, tn)
            assert mine == pytest.approx(theirs, abs=1e-10), (
                f"mismatch for {(tp, fp, fn, tn)}"
            )

    def test_perfect_agreement_is_one(self):
        assert cohen_kappa(10, 0, 0, 10) == 1.0

    def test_empty_matrix_returns_none(self):
        assert cohen_kappa(0, 0, 0, 0) is None

    def test_degenerate_single_class_returns_none(self):
        # all predictions AND labels identical → pe = 1 → undefined
        assert cohen_kappa(20, 0, 0, 0) is None

    def test_chance_agreement_near_zero(self):
        # 50/50 marginals with 50% agreement → κ ≈ 0
        kappa_chance = cohen_kappa(5, 5, 5, 5)
        assert kappa_chance is not None and abs(kappa_chance) < 1e-12

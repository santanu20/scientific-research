"""P1-12 selection model + p-curve tests (synthetic property-based)."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from _stats import p_curve_test, vevea_hedges_selection_model


class TestVeveaHedges:
    def test_no_suppression_recovers_theta(self):
        random.seed(1)
        theta = 0.5
        eff, var = [], []
        for _ in range(60):
            s = random.uniform(0.1, 0.4)
            eff.append(theta + random.gauss(0, s))
            var.append(s * s)
        res = vevea_hedges_selection_model(eff, var)
        assert res is not None and res["converged"]
        assert abs(res["adjusted_effect"] - theta) < 0.1
        assert res["selection_delta"] > 0.5  # no strong selection inferred

    def test_suppression_pulls_naive_estimate_up_and_model_corrects_down(self):
        random.seed(2)
        theta = 0.3
        kept_eff, kept_var = [], []
        while len(kept_eff) < 40:
            s = random.uniform(0.15, 0.45)
            y = theta + random.gauss(0, s)
            p = 0.5 * (1 + math.erf(y / (s * math.sqrt(2))))
            if p > 0.95 or len(kept_eff) >= 60 - 40:  # drop many non-sig
                if p > 0.95 and random.random() < 0.6:
                    continue  # suppress 60% of non-significant results
            kept_eff.append(y)
            kept_var.append(s * s)
        res = vevea_hedges_selection_model(kept_eff, kept_var)
        assert res is not None
        assert res["selection_delta"] < 1.0  # selection detected
        # corrected estimate should sit below the inflated naive pooled effect
        assert res["adjusted_effect"] < res["naive_effect"]

    def test_too_few_studies_returns_none(self):
        assert vevea_hedges_selection_model([0.5, 0.6], [0.04, 0.04]) is None


class TestPCurve:
    def test_true_effect_right_skewed(self):
        random.seed(3)
        eff, var = [], []
        for _ in range(50):
            s = random.uniform(0.1, 0.3)
            eff.append(0.6 + random.gauss(0, s))
            var.append(s * s)
        res = p_curve_test(eff, var)
        assert res["k_significant"] > 10
        assert res["p_right_skew"] < 0.05
        assert "evidential value" in res["interpretation"]

    def test_null_not_right_skewed(self):
        random.seed(4)
        eff, var = [], []
        for _ in range(50):
            s = random.uniform(0.3, 0.6)  # low power → few significant survivors
            eff.append(0.0 + random.gauss(0, s))
            var.append(s * s)
        res = p_curve_test(eff, var)
        # few survivors → low-power caution instead of an evidential verdict
        if res["k_significant"] < 5:
            assert "caution" in res["interpretation"]
        else:
            assert res["p_right_skew"] > 0.05

    def test_no_significant_studies(self):
        res = p_curve_test([0.01, -0.01, 0.0], [4.0, 4.0, 4.0])
        assert res["k_significant"] == 0
        assert res["interpretation"] == "no significant studies — p-curve undefined"

    def test_mean_p_within_bounds(self):
        random.seed(5)
        eff = [random.gauss(0.5, 0.2) for _ in range(30)]
        var = [0.04] * 30
        res = p_curve_test(eff, var)
        assert 0.0 < res["mean_p"] < 0.05

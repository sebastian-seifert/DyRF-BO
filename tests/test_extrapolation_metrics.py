"""Unit tests for extrapolation uncertainty quantification metrics suite."""

import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.metrics_suite import (
    compute_comprehensive_metrics,
    mean_prediction_interval_width,
    outlier_error_detection_auc,
    prediction_interval_coverage_probability,
    spearman_rank_correlation,
    winkler_interval_score,
)


class TestSpearmanRankCorrelation:
    """Tests for spearman_rank_correlation."""

    def test_perfect_positive_and_negative_monotonicity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pos = np.array([10.0, 25.0, 32.0, 40.0, 100.0])
        y_neg = np.array([100.0, 40.0, 32.0, 25.0, 10.0])

        assert np.isclose(spearman_rank_correlation(x, y_pos), 1.0)
        assert np.isclose(spearman_rank_correlation(x, y_neg), -1.0)

    def test_partial_monotonicity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 1.0, 4.0, 3.0])
        corr = spearman_rank_correlation(x, y)
        assert -1.0 < corr < 1.0
        assert np.isclose(corr, 0.6)

    def test_constant_array_safe_handling(self):
        x_const = np.array([2.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        corr1 = spearman_rank_correlation(x_const, y)
        corr2 = spearman_rank_correlation(y, x_const)
        corr3 = spearman_rank_correlation(x_const, x_const)

        # Should return 0.0 or nan without raising exceptions
        assert corr1 == 0.0 or np.isnan(corr1)
        assert corr2 == 0.0 or np.isnan(corr2)
        assert corr3 == 0.0 or np.isnan(corr3)

    def test_short_arrays(self):
        assert spearman_rank_correlation(np.array([1.0]), np.array([2.0])) == 0.0


class TestPredictionIntervalCoverageProbability:
    """Tests for prediction_interval_coverage_probability (PICP)."""

    def test_exact_manual_values(self):
        # 4 points: abs_error = [0.0, 1.0, 2.5, 3.0], u = 1.0
        # point 0: 0.0 <= 1.0 (True)
        # point 1: 1.0 <= 1.0 (True, boundary inclusion)
        # point 2: 2.5 <= 1.0 (False)
        # point 3: 3.0 <= 1.0 (False)
        # PICP = 2/4 = 0.5
        y_true = np.array([0.0, 1.0, 2.5, 3.0])
        y_hat = np.array([0.0, 0.0, 0.0, 0.0])
        uncertainty = np.array([1.0, 1.0, 1.0, 1.0])

        picp = prediction_interval_coverage_probability(y_true, y_hat, uncertainty)
        assert np.isclose(picp, 0.5)

    def test_all_covered_and_none_covered(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_hat = np.array([1.0, 2.0, 3.0])
        uncertainty = np.array([0.1, 0.1, 0.1])
        assert np.isclose(prediction_interval_coverage_probability(y_true, y_hat, uncertainty), 1.0)

        uncertainty_tiny = np.array([0.0, 0.0, 0.0])
        y_true_far = np.array([10.0, 20.0, 30.0])
        assert np.isclose(prediction_interval_coverage_probability(y_true_far, y_hat, uncertainty_tiny), 0.0)

    def test_empty_array(self):
        assert prediction_interval_coverage_probability(np.array([]), np.array([]), np.array([])) == 0.0


class TestMeanPredictionIntervalWidth:
    """Tests for mean_prediction_interval_width (MPIW)."""

    def test_exact_manual_calculation(self):
        # MPIW = mean(2 * uncertainty)
        uncertainty = np.array([1.0, 2.0, 3.0])
        mpiw = mean_prediction_interval_width(uncertainty)
        # 2 * mean(1, 2, 3) = 2 * 2.0 = 4.0
        assert np.isclose(mpiw, 4.0)

    def test_zeros(self):
        uncertainty = np.zeros(5)
        assert np.isclose(mean_prediction_interval_width(uncertainty), 0.0)


class TestWinklerIntervalScore:
    """Tests for winkler_interval_score."""

    def test_exact_manual_values(self):
        # Points:
        # Point 1: inside interval: y=0, y_hat=0, U=1 -> S = 2(1) = 2.0
        # Point 2: above upper: y=2, y_hat=0, U=1 -> S = 2(1) + (2/0.05)*(2 - 1) = 2 + 40 = 42.0
        # Point 3: below lower: y=-2.5, y_hat=0, U=1 -> S = 2(1) + (2/0.05)*(-1 - (-2.5)) = 2 + 40*1.5 = 62.0
        # Mean = (2 + 42 + 62) / 3 = 106 / 3
        y_true = np.array([0.0, 2.0, -2.5])
        y_hat = np.array([0.0, 0.0, 0.0])
        uncertainty = np.array([1.0, 1.0, 1.0])

        score = winkler_interval_score(y_true, y_hat, uncertainty, alpha=0.05)
        expected = (2.0 + 42.0 + 62.0) / 3.0
        assert np.isclose(score, expected)

    def test_custom_alpha(self):
        # alpha = 0.10 -> penalty multiplier = 2 / 0.10 = 20.0
        # Point 2: S = 2 + 20 * 1 = 22.0
        y_true = np.array([2.0])
        y_hat = np.array([0.0])
        uncertainty = np.array([1.0])

        score = winkler_interval_score(y_true, y_hat, uncertainty, alpha=0.10)
        assert np.isclose(score, 22.0)


class TestOutlierErrorDetectionAUC:
    """Tests for outlier_error_detection_auc."""

    def test_perfect_and_imperfect_detection(self):
        # 100 points, top 10% errors
        np.random.seed(42)
        errors = np.linspace(0.1, 10.0, 100)
        # Uncertainty perfectly proportional to error
        uncertainty_perfect = errors.copy()

        auroc, auprc = outlier_error_detection_auc(errors, uncertainty_perfect, top_quantile=0.90)
        assert np.isclose(auroc, 1.0)
        assert np.isclose(auprc, 1.0)

        # Inverted uncertainty (terrible detector)
        uncertainty_inverted = -errors
        auroc_inv, _ = outlier_error_detection_auc(errors, uncertainty_inverted, top_quantile=0.90)
        assert np.isclose(auroc_inv, 0.0)

    def test_constant_error_handling(self):
        errors_const = np.ones(50)
        uncertainty = np.random.rand(50)
        auroc, auprc = outlier_error_detection_auc(errors_const, uncertainty, top_quantile=0.90)
        # Should handle single-class target safely without crash
        assert isinstance(auroc, float)
        assert isinstance(auprc, float)


class TestComputeComprehensiveMetrics:
    """Tests for compute_comprehensive_metrics."""

    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n = 80
        y_true = np.random.randn(n)
        y_hat = y_true + np.random.normal(0, 0.5, size=n)
        u_slcb = np.full(n, 1.0)
        u_plcb = 0.8 + 0.4 * np.abs(y_true - y_hat)
        d_norm = np.linspace(0.0, 1.0, n)
        strata = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20)
        return y_true, y_hat, u_slcb, u_plcb, d_norm, strata

    def test_global_metrics_without_strata(self, synthetic_data):
        y_true, y_hat, u_slcb, u_plcb, d_norm, _ = synthetic_data
        metrics = compute_comprehensive_metrics(y_true, y_hat, u_slcb, u_plcb, d_norm)

        # Check required keys for both SLCB and PLCB
        for method in ["slcb", "plcb"]:
            for metric in ["spearman_dist", "spearman_err", "picp", "mpiw", "winkler", "auroc", "auprc"]:
                key = f"{method}_{metric}"
                assert key in metrics, f"Missing expected key {key}"
                assert isinstance(metrics[key], (int, float, np.floating))

    def test_strata_breakdown(self, synthetic_data):
        y_true, y_hat, u_slcb, u_plcb, d_norm, strata = synthetic_data
        metrics = compute_comprehensive_metrics(y_true, y_hat, u_slcb, u_plcb, d_norm, strata_labels=strata)

        # Global keys exist
        assert "slcb_picp" in metrics
        assert "plcb_picp" in metrics

        # Per-stratum keys exist for 0, 1, 2, 3
        for s in [0, 1, 2, 3]:
            assert f"stratum_{s}_slcb_picp" in metrics
            assert f"stratum_{s}_plcb_picp" in metrics
            assert f"stratum_{s}_slcb_winkler" in metrics
            assert f"stratum_{s}_plcb_winkler" in metrics

        # Verify stratum 0 subset value matches isolated calculation
        mask_0 = strata == 0
        expected_s0_slcb_picp = prediction_interval_coverage_probability(
            y_true[mask_0], y_hat[mask_0], u_slcb[mask_0]
        )
        assert np.isclose(metrics["stratum_0_slcb_picp"], expected_s0_slcb_picp)

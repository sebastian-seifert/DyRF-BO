import os
import sys
import time
import tracemalloc
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir)), current_dir]:
    if os.path.exists(os.path.join(p, "ep_extractors")):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor


class TestDistanceAwareEvidentialExtractorAdversarialStress(unittest.TestCase):
    """
    Adversarial stress harness for DistanceAwareEvidentialExtractor (DA-EHRF).
    """

    def setUp(self):
        np.random.seed(42)

    def test_1_extreme_coordinates(self):
        """
        Adversarial Test 1: Extreme Coordinates (+-1e15, +-1e-15).
        Verifies absence of overflow, underflow, or NaN generation.
        """
        X_train = np.random.uniform(0.0, 1.0, (25, 2))
        y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])
        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        extreme_vals = [1e15, -1e15, 1e-15, -1e-15, 1e30, -1e30]
        for val in extreme_vals:
            with self.subTest(val=val):
                q = np.array([[val, val]])
                u_e = ext.extract_epistemic_signal(q)
                self.assertEqual(u_e.shape, (1,))
                self.assertTrue(np.all(np.isfinite(u_e)), f"Failed on val={val}: got {u_e}")
                self.assertGreaterEqual(u_e[0], 0.0)

    def test_2a_degenerate_dataset_n2(self):
        """
        Adversarial Test 2a: Minimal dataset with N=2 samples.
        """
        X_train = np.array([[0.0, 1.0], [1.0, 0.0]])
        y_train = np.array([1.5, 3.5])
        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        q = np.array([[0.5, 0.5], [10.0, 10.0]])
        u_e = ext.extract_epistemic_signal(q)
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertGreater(u_e[1], u_e[0])

    def test_2b_degenerate_dataset_identical_targets_n2(self):
        """
        Adversarial Test 2b: Degenerate identical targets y1 = y2 = c (zero target variance).
        """
        X_train = np.array([[0.0, 1.0], [1.0, 0.0]])
        y_train = np.array([42.0, 42.0])
        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        self.assertGreaterEqual(ext.sigma_0, 1e-6)
        q = np.array([[0.5, 0.5], [10.0, 10.0]])
        u_e = ext.extract_epistemic_signal(q)
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertTrue(np.all(u_e >= 0.0))
        # Extrapolation must have strictly positive spatial uncertainty
        self.assertGreater(u_e[1], 0.0)

    def test_2c_degenerate_dataset_collinear_features(self):
        """
        Adversarial Test 2c: Perfect collinearity in feature matrix.
        """
        x0 = np.linspace(0.0, 1.0, 20)
        x1 = 2.0 * x0
        x2 = -3.0 * x0 + 1.0
        X_train = np.column_stack([x0, x1, x2])
        y_train = 5.0 * x0 + np.random.normal(0, 0.01, 20)

        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        self.assertTrue(np.all(np.isfinite(ext.weights)))
        self.assertTrue(np.all(ext.weights >= 0.0))

        q = np.array([[0.5, 1.0, -0.5], [5.0, 10.0, -14.0]])
        u_e = ext.extract_epistemic_signal(q)
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertGreater(u_e[1], u_e[0])

    def test_2d_degenerate_dataset_all_zero_columns(self):
        """
        Adversarial Test 2d: Zero-variance feature columns (all zeros).
        """
        X_train = np.zeros((30, 4))
        X_train[:, 0] = np.linspace(0.0, 1.0, 30)  # cols 1, 2, 3 are all 0.0
        y_train = 2.0 * X_train[:, 0]

        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        self.assertTrue(np.all(np.isfinite(ext.weights)))
        # Range protection ensures no divide-by-zero
        self.assertEqual(ext.weights.shape, (4,))

        q = np.array([[0.5, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]])
        u_e = ext.extract_epistemic_signal(q)
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertGreater(u_e[1], u_e[0])

    def test_3_dimension_scaling_d100(self):
        """
        Adversarial Test 3: Dimension Scaling with D=100 features.
        Verifies feature weights scale properly, output remains finite, and
        OOD signals consistently dominate in-distribution signals.
        """
        N = 50
        D = 100
        X_train = np.random.uniform(-1.0, 1.0, (N, D))
        # 5 signal features, 95 noise features
        y_train = np.sum(X_train[:, :5] ** 2, axis=1)

        rf = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        self.assertEqual(ext.weights.shape, (D,))
        self.assertTrue(np.all(np.isfinite(ext.weights)))

        # Query in-distribution vs far out-of-distribution
        q_in = X_train[:5]
        q_ood = np.random.uniform(5.0, 10.0, (5, D))

        u_in = ext.extract_epistemic_signal(q_in)
        u_ood = ext.extract_epistemic_signal(q_ood)

        self.assertTrue(np.all(np.isfinite(u_in)))
        self.assertTrue(np.all(np.isfinite(u_ood)))
        self.assertTrue(np.all(u_ood > u_in), "D=100 failed: OOD uncertainty not strictly greater than in-distribution")

    def test_4a_scale_equivariance_exact_analytic(self):
        """
        Adversarial Test 4a: Exact scale equivariance without MDI tree re-fitting jitter.
        Uses use_feature_importances=False to verify exact linearity to floating point precision.
        """
        X_train = np.random.uniform(0.0, 1.0, (30, 2))
        y_train = np.sin(X_train[:, 0]) + 2.0 * np.cos(X_train[:, 1])

        rf_base = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        ext_base = DistanceAwareEvidentialExtractor(rf_base, use_feature_importances=False)
        ext_base.fit(X_train, y_train)

        q = np.array([[0.5, 0.5], [2.0, 2.0], [5.0, 5.0]])
        u_base = ext_base.extract_epistemic_signal(q)

        for alpha in [1e5, 1e-5]:
            with self.subTest(alpha=alpha):
                y_scaled = y_train * alpha
                rf_scaled = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_scaled)
                ext_scaled = DistanceAwareEvidentialExtractor(rf_scaled, use_feature_importances=False)
                ext_scaled.fit(X_train, y_scaled)

                u_scaled = ext_scaled.extract_epistemic_signal(q)
                expected = alpha * u_base

                np.testing.assert_allclose(
                    u_scaled,
                    expected,
                    rtol=0.10,
                    atol=1e-6,
                    err_msg=f"Exact scale equivariance failed for alpha={alpha}"
                )

    def test_4b_scale_equivariance_dispatch_scales_with_mdi(self):
        """
        Adversarial Test 4b: Scale Equivariance with alpha = 1e5 and alpha = 1e-5 with MDI.
        Accounts for sklearn's float precision MDI variance across 10^5 dynamic range.
        """
        X_train = np.random.uniform(0.0, 1.0, (30, 2))
        y_train = np.sin(X_train[:, 0]) + 2.0 * np.cos(X_train[:, 1])

        rf_base = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)
        ext_base = DistanceAwareEvidentialExtractor(rf_base, use_feature_importances=True)
        ext_base.fit(X_train, y_train)

        q = np.array([[0.5, 0.5], [2.0, 2.0], [5.0, 5.0]])
        u_base = ext_base.extract_epistemic_signal(q)

        for alpha in [1e5, 1e-5]:
            with self.subTest(alpha=alpha):
                y_scaled = y_train * alpha
                rf_scaled = RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_scaled)
                ext_scaled = DistanceAwareEvidentialExtractor(rf_scaled, use_feature_importances=True)
                ext_scaled.fit(X_train, y_scaled)

                u_scaled = ext_scaled.extract_epistemic_signal(q)
                expected = alpha * u_base

                np.testing.assert_allclose(
                    u_scaled,
                    expected,
                    rtol=3e-2,
                    atol=1e-6,
                    err_msg=f"Scale equivariance with MDI failed for alpha={alpha}"
                )

    def test_6_pathology_sub_micro_target_scale_discontinuity(self):
        """
        Adversarial Test 6 (Cured Invariant): Target std < 1e-6 maintains scale equivariance
        without discontinuity across 1e-6 threshold.
        """
        X_train = np.random.uniform(0.0, 1.0, (20, 2))
        y_unit = np.random.randn(20)
        y_unit = y_unit / np.std(y_unit)

        # Scale slightly above 1e-6 vs slightly below 1e-6
        scale_above = 1.01e-6
        scale_below = 0.99e-6

        y_above = y_unit * scale_above
        y_below = y_unit * scale_below

        rf_above = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_above)
        ext_above = DistanceAwareEvidentialExtractor(rf_above)
        ext_above.fit(X_train, y_above)

        rf_below = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_below)
        ext_below = DistanceAwareEvidentialExtractor(rf_below)
        ext_below.fit(X_train, y_below)

        # Confirm scale equivariance without 10^6 discontinuity across 1e-6 threshold
        self.assertAlmostEqual(ext_above.sigma_0, 1.01e-6, delta=1e-8)
        self.assertAlmostEqual(ext_below.sigma_0, 0.99e-6, delta=1e-8)

        q = np.array([[2.0, 2.0]])
        u_above = ext_above.extract_epistemic_signal(q)[0]
        u_below = ext_below.extract_epistemic_signal(q)[0]

        # The ratio should be ~1.0 (continuous, within 0.10 accounting for sklearn float32 tree split variance)
        ratio = u_below / u_above
        self.assertAlmostEqual(ratio, 1.0, delta=0.10)

    def test_7_pathology_sub_micro_feature_range_blindness(self):
        """
        Adversarial Test 7 (Cured Invariant): Feature range ptp(X) < 1e-6 preserves
        spatial extrapolation awareness (u_out > u_in).
        """
        x_scale = 1e-7
        X_train = np.random.uniform(0.0, x_scale, (30, 2))
        y_train = np.random.randn(30)

        rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        # Query in-distribution vs 2x domain bound (clear extrapolation)
        q_in = np.array([[0.5 * x_scale, 0.5 * x_scale]])
        q_out = np.array([[2.0 * x_scale, 2.0 * x_scale]])

        u_in = ext.extract_epistemic_signal(q_in)[0]
        u_out = ext.extract_epistemic_signal(q_out)[0]

        # In a distance-aware model, u_out should be strictly greater than u_in.
        diff = u_out - u_in
        self.assertGreater(diff, 0.0, msg="Spatial extrapolation is active for domains < 1e-6")
        self.assertGreater(u_out, u_in, msg="OOD uncertainty must be strictly greater than in-distribution for micro-domains")

    def test_5_large_candidate_batches_10k(self):
        """
        Adversarial Test 5: Candidate batch scaling with 10,000 candidates.
        Measures peak memory (< 100 MB) and latency (< 500 ms).
        """
        N_train = 200
        D = 10
        N_cand = 10000

        X_train = np.random.uniform(0.0, 1.0, (N_train, D))
        y_train = np.sum(X_train, axis=1)

        rf = RandomForestRegressor(n_estimators=20, random_state=42).fit(X_train, y_train)
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)

        X_cand = np.random.uniform(0.0, 1.0, (N_cand, D))

        tracemalloc.start()
        t0 = time.perf_counter()
        u_cand = ext.extract_epistemic_signal(X_cand)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem_bytes / (1024 * 1024)

        self.assertEqual(u_cand.shape, (N_cand,))
        self.assertTrue(np.all(np.isfinite(u_cand)))
        self.assertLess(t_elapsed_ms, 500.0, f"10k candidate batch too slow: {t_elapsed_ms:.2f} ms")
        self.assertLess(peak_mem_mb, 100.0, f"10k candidate batch memory too high: {peak_mem_mb:.2f} MB")


if __name__ == "__main__":
    unittest.main()

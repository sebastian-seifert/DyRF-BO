import os
import sys
import unittest
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.dirname(current_dir), os.path.dirname(os.path.dirname(current_dir)), current_dir]:
    if os.path.exists(os.path.join(p, "ep_extractors")):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor


class TestDistanceAwareEvidentialExtractorInvariants(unittest.TestCase):
    """
    Validates fundamental mathematical invariants of the Distance-Aware Evidential
    Hybrid Random Forest (DA-EHRF) epistemic uncertainty extractor:
        U_E(x) = sqrt( V_ens(x) + V_leaf(x) + V_spatial(x) )
    """
    def setUp(self):
        np.random.seed(42)
        # Bounded training domain in [0, 1]^2
        self.X_train = np.random.uniform(0.0, 1.0, size=(40, 2))
        self.y_train = (np.sin(self.X_train[:, 0] * np.pi) + 
                        2.0 * np.cos(self.X_train[:, 1] * np.pi))
        
        self.rf = RandomForestRegressor(n_estimators=10, random_state=42)
        self.rf.fit(self.X_train, self.y_train)
        
        self.extractor = DistanceAwareEvidentialExtractor(self.rf, spatial_metric="euclidean")
        self.extractor.fit(self.X_train, self.y_train)

    def test_non_negativity(self):
        """Invariant 1: U_E(x) >= 0 for all points in R^D (in-distribution and out-of-distribution)."""
        # Test grid covering both inside and outside of [0, 1]^2
        x_grid = np.linspace(-5.0, 5.0, 30)
        y_grid = np.linspace(-5.0, 5.0, 30)
        X_test = np.column_stack([g.flatten() for g in np.meshgrid(x_grid, y_grid)])
        
        u_e = self.extractor.extract_epistemic_signal(X_test)
        
        self.assertEqual(u_e.shape, (len(X_test),))
        self.assertTrue(np.all(np.isfinite(u_e)), "Epistemic uncertainty contains NaNs or Infs.")
        self.assertTrue(np.all(u_e >= 0.0), f"Epistemic uncertainty has negative values: {u_e[u_e < 0]}")

    def test_scale_unit_and_equivariance(self):
        """
        Invariant 2: U_E(x) outputs linear standard deviation units [y] and is scale-equivariant.
        Scaling targets by alpha > 0 must scale U_E by exactly alpha.
        """
        alpha = 7.5
        y_scaled = self.y_train * alpha
        
        rf_scaled = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_scaled.fit(self.X_train, y_scaled)
        
        ext_scaled = DistanceAwareEvidentialExtractor(rf_scaled, spatial_metric="euclidean")
        ext_scaled.fit(self.X_train, y_scaled)
        
        X_test = np.array([
            [0.5, 0.5],    # In-distribution
            [1.5, 1.5],    # Extrapolation
            [-1.0, 2.0]    # Far extrapolation
        ])
        
        u_e_orig = self.extractor.extract_epistemic_signal(X_test)
        u_e_scaled = ext_scaled.extract_epistemic_signal(X_test)
        
        # Scale equivariance: U_E(alpha * y) == alpha * U_E(y)
        np.testing.assert_allclose(
            u_e_scaled, 
            alpha * u_e_orig, 
            rtol=1e-4, 
            atol=1e-6,
            err_msg="Extractor failed scale equivariance test (output must be linear standard deviation units [y])."
        )

    def test_monotonic_distance_growth_extrapolation(self):
        """
        Invariant 3: Overconfidence immunity. U_E(x) increases monotonically
        along an extrapolation ray moving away from training data.
        """
        # Linear ray moving from boundary x = 1.0 along x-axis
        t_values = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0])
        ray_points = np.column_stack([1.0 + t_values, np.full_like(t_values, 0.5)])
        
        u_e_ray = self.extractor.extract_epistemic_signal(ray_points)
        diffs = np.diff(u_e_ray)
        
        self.assertTrue(
            np.all(diffs > 0.0),
            f"U_E failed strict monotonic growth on extrapolation ray. Diffs: {diffs}"
        )

    def test_extrapolation_saturation_bounds(self):
        """
        Invariant 4: As d(x, D) -> inf, U_E(x) saturates smoothly to a finite upper bound
        and does NOT explode to infinity or collapse to zero.
        """
        extreme_pts = np.array([
            [1000.0, 1000.0],
            [2000.0, 2000.0],
            [10000.0, 10000.0]
        ])
        u_e_extreme = self.extractor.extract_epistemic_signal(extreme_pts)
        
        # In extreme extrapolation, kernel distance saturates to 1.0, so values must be practically identical
        np.testing.assert_allclose(
            u_e_extreme[0], u_e_extreme[1], rtol=1e-4, atol=1e-5,
            err_msg="U_E did not saturate asymptotically in far extrapolation."
        )
        
        # Must be strictly bounded by theoretical maximum:
        # V_ens <= (y_max - y_min)^2, V_leaf <= sigma_0^2 * kappa_leaf, V_spatial <= sigma_0^2 * c_spatial
        y_range = float(np.ptp(self.y_train))
        sigma_0 = float(np.std(self.y_train))
        u_max_bound = np.sqrt(y_range**2 + (sigma_0**2) * self.extractor.kappa_leaf + (sigma_0**2) * self.extractor.c_spatial)
        
        self.assertLessEqual(u_e_extreme[0], u_max_bound + 1e-4)
        self.assertGreater(u_e_extreme[0], sigma_0 * 0.5)

    def test_in_distribution_convergence_dense_support(self):
        """
        Invariant 5: When data is dense around x (d -> 0, N_leaf >> 1),
        U_E(x) is significantly smaller than in an unobserved gap.
        """
        np.random.seed(42)
        # Synthetic 1D gap: dense cluster at [0, 1] (N=50), dense cluster at [9, 10] (N=50)
        # Empty gap in [1.0, 9.0]
        X_cluster1 = np.random.uniform(0.0, 1.0, size=(50, 1))
        X_cluster2 = np.random.uniform(9.0, 10.0, size=(50, 1))
        X_gap_train = np.vstack([X_cluster1, X_cluster2])
        y_gap_train = np.sin(X_gap_train[:, 0])
        
        rf_gap = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=42)
        rf_gap.fit(X_gap_train, y_gap_train)
        
        ext_gap = DistanceAwareEvidentialExtractor(rf_gap, lengthscale=0.5)
        ext_gap.fit(X_gap_train, y_gap_train)
        
        x_dense_center = np.array([[0.5]])
        x_gap_center = np.array([[5.0]])  # Deep inside empty gap
        
        u_dense = ext_gap.extract_epistemic_signal(x_dense_center)[0]
        u_gap = ext_gap.extract_epistemic_signal(x_gap_center)[0]
        
        # Uncertainty in the center of the gap must be substantially higher than in dense training data
        self.assertGreater(u_gap, u_dense, "Gap uncertainty is not greater than dense ID uncertainty.")
        self.assertGreater(
            u_gap, 
            1.5 * u_dense, 
            f"Gap uncertainty ({u_gap:.4f}) is not at least 1.5x dense ID uncertainty ({u_dense:.4f})."
        )

    def test_zero_noise_robustness(self):
        """
        Invariant 6: Noise independence. For deterministic training data (sigma_noise = 0),
        U_E(x) remains strictly positive in unseen/gap regions.
        """
        # Exact deterministic sine wave without noise
        X_clean = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
        y_clean = np.sin(2.0 * np.pi * X_clean[:, 0])
        
        rf_clean = RandomForestRegressor(n_estimators=10, random_state=42)
        rf_clean.fit(X_clean, y_clean)
        
        ext_clean = DistanceAwareEvidentialExtractor(rf_clean)
        ext_clean.fit(X_clean, y_clean)
        
        X_ood = np.array([[3.0]])
        u_ood = ext_clean.extract_epistemic_signal(X_ood)[0]
        
        sigma_0 = np.std(y_clean)
        self.assertGreater(sigma_0, 0.1)
        self.assertGreater(
            u_ood, 
            0.5 * sigma_0, 
            "Epistemic uncertainty collapsed to near-zero in unseen region under zero observation noise!"
        )


class TestDistanceAwareEvidentialExtractorEdgeCases(unittest.TestCase):
    """
    Validates boundary conditions, edge cases, zero-variance components,
    and numerical robustness of DistanceAwareEvidentialExtractor.
    """
    def test_small_sample_size_n3(self):
        """Edge Case 1: Smallest dataset (N=3) with single-sample leaves."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y_train = np.array([1.0, 2.0, 3.0])
        
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf, spatial_metric="euclidean")
        ext.fit(X_train, y_train)
        
        X_test = np.array([[0.3, 0.3], [5.0, 5.0]])
        u_e = ext.extract_epistemic_signal(X_test)
        
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertTrue(np.all(u_e >= 0.0))
        self.assertGreater(u_e[1], u_e[0])

    def test_constant_target_zero_prior_variance(self):
        """
        Edge Case 2: Constant target y = c (zero sample variance, std(y) == 0.0).
        Must be protected by epsilon fallback (sigma_0 = 1.0) without division by zero.
        """
        X_train = np.random.uniform(0.0, 1.0, (20, 2))
        y_train = np.full(20, 42.0)  # std(y) = 0.0
        
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)
        
        self.assertGreaterEqual(ext.sigma_0, 1e-6, "Prior sigma_0 was not protected against 0 variance.")
        
        X_test = np.array([[0.5, 0.5], [10.0, 10.0]])
        u_e = ext.extract_epistemic_signal(X_test)
        
        self.assertEqual(u_e.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_e)))
        self.assertTrue(np.all(u_e >= 0.0))
        # In far extrapolation, spatial uncertainty should still activate because of fallback sigma_0
        self.assertGreater(u_e[1], 0.0)

    def test_collinear_and_zero_variance_features(self):
        """
        Edge Case 3: Dataset with a zero-variance (constant) feature column
        and a perfectly collinear feature column. Range clamping must prevent 1/0.
        """
        np.random.seed(42)
        x0 = np.random.uniform(0.0, 1.0, 30)
        x1 = np.full(30, 3.14159)     # Constant feature (range = 0.0)
        x2 = 2.5 * x0                # Perfectly collinear feature
        
        X_train = np.column_stack([x0, x1, x2])
        y_train = 2.0 * x0 + np.random.normal(0, 0.05, 30)
        
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)
        
        # Check weights are finite and non-negative
        self.assertTrue(np.all(np.isfinite(ext.weights)), "Feature weights contain NaNs or Infs.")
        self.assertTrue(np.all(ext.weights >= 0.0))
        
        X_test = np.array([
            [0.5, 3.14159, 1.25],
            [0.5, 999.0, 1.25],     # Point varying along constant feature
            [5.0, 3.14159, 12.5]
        ])
        u_e = ext.extract_epistemic_signal(X_test)
        self.assertEqual(u_e.shape, (3,))
        self.assertTrue(np.all(np.isfinite(u_e)))

    def test_high_dimensional_inputs_d10_d20(self):
        """Edge Case 4: High-dimensional inputs (D=10, 20) with feature importances."""
        for D in [10, 20]:
            with self.subTest(dimension=D):
                np.random.seed(42)
                X_train = np.random.uniform(-1.0, 1.0, (50, D))
                # Only first 2 dimensions are active (sparse importances)
                y_train = X_train[:, 0]**2 + np.sin(X_train[:, 1])
                
                rf = RandomForestRegressor(n_estimators=10, random_state=42)
                rf.fit(X_train, y_train)
                
                ext = DistanceAwareEvidentialExtractor(rf)
                ext.fit(X_train, y_train)
                
                X_test = np.random.uniform(-2.0, 2.0, (20, D))
                u_e = ext.extract_epistemic_signal(X_test)
                
                self.assertEqual(u_e.shape, (20,))
                self.assertTrue(np.all(np.isfinite(u_e)))
                self.assertTrue(np.all(u_e >= 0.0))

    def test_boundary_threshold_split_vs_far_outside(self):
        """
        Edge Case 5: Exact boundary threshold.
        Point exactly on the bounding box boundary (x=1.0) vs near-boundary (x=1.05) vs far outside (x=3.0).
        While tree disagreement is constant, U_E must strictly increase.
        """
        X_train = np.linspace(0.0, 1.0, 25).reshape(-1, 1)
        y_train = X_train[:, 0] ** 2
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf, spatial_metric="euclidean")
        ext.fit(X_train, y_train)
        
        pts = np.array([[1.0], [1.05], [1.5], [3.0]])
        u_e = ext.extract_epistemic_signal(pts)
        
        # Strict inequality from boundary to far outside
        self.assertLess(u_e[0], u_e[1])
        self.assertLess(u_e[1], u_e[2])
        self.assertLess(u_e[2], u_e[3])

    def test_extreme_inputs_numerical_stability(self):
        """Edge Case 6: Extreme floating point inputs (+-1e8), 1D inputs, single sample."""
        X_train = np.random.uniform(0.0, 1.0, (20, 2))
        y_train = np.random.randn(20)
        
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)
        
        # 1. Extreme magnitude coordinates
        X_extreme = np.array([
            [1e8, -1e8],
            [-1e8, 1e8]
        ])
        u_extreme = ext.extract_epistemic_signal(X_extreme)
        self.assertEqual(u_extreme.shape, (2,))
        self.assertTrue(np.all(np.isfinite(u_extreme)))
        
        # 2. Single query point (1, 2)
        X_single = np.array([[0.5, 0.5]])
        u_single = ext.extract_epistemic_signal(X_single)
        self.assertEqual(u_single.shape, (1,))
        self.assertTrue(np.isfinite(u_single[0]))
        
        # 3. 1D query array (2,) -> auto-converted to 2D
        X_1d = np.array([0.5, 0.5])
        u_1d = ext.extract_epistemic_signal(X_1d)
        self.assertEqual(u_1d.shape, (1,))
        
        # 4. Empty query array (0, 2)
        X_empty = np.empty((0, 2))
        u_empty = ext.extract_epistemic_signal(X_empty)
        self.assertEqual(u_empty.shape, (0,))


class TestDistanceAwareEvidentialExtractorInterface(unittest.TestCase):
    """
    Validates registry integration, SMAC3 interface compatibility,
    hyperparameter controls, and execution latency.
    """
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.uniform(0.0, 1.0, (30, 2))
        self.y_train = self.X_train[:, 0] + self.X_train[:, 1]
        self.rf = RandomForestRegressor(n_estimators=5, random_state=42)
        self.rf.fit(self.X_train, self.y_train)

    def test_registry_registration(self):
        """Interface 1: Verify DistanceAwareEvidentialExtractor is registered in UQExtractorRegistry."""
        registered_keys = UQExtractorRegistry.list_registered()
        self.assertIn("distance_evidential", registered_keys)
        
        extractor = UQExtractorRegistry.get("distance_evidential", self.rf)
        self.assertIsInstance(extractor, BaseEpistemicExtractor)
        self.assertIsInstance(extractor, DistanceAwareEvidentialExtractor)

    def test_unfitted_extractor_raises_runtime_error(self):
        """Interface 2: Calling extract_epistemic_signal before fit() must raise RuntimeError."""
        ext = DistanceAwareEvidentialExtractor(self.rf)
        with self.assertRaises(RuntimeError):
            ext.extract_epistemic_signal(np.array([[0.5, 0.5]]))

    def test_hyperparameter_controls(self):
        """Interface 3: Verify hyperparameters kappa_leaf, c_spatial, lengthscale, use_feature_importances."""
        X_far = np.array([[5.0, 5.0]])
        
        # 1. Base extractor
        ext_base = DistanceAwareEvidentialExtractor(
            self.rf, kappa_leaf=1.0, c_spatial=1.0, lengthscale=1.0, spatial_metric="euclidean"
        )
        ext_base.fit(self.X_train, self.y_train)
        u_base = ext_base.extract_epistemic_signal(X_far)[0]
        
        # 2. Doubled spatial scaling
        ext_high_spatial = DistanceAwareEvidentialExtractor(
            self.rf, kappa_leaf=1.0, c_spatial=4.0, lengthscale=1.0, spatial_metric="euclidean"
        )
        ext_high_spatial.fit(self.X_train, self.y_train)
        u_high_spatial = ext_high_spatial.extract_epistemic_signal(X_far)[0]
        self.assertGreater(u_high_spatial, u_base)
        
        # 3. Doubled leaf ignorance scaling
        ext_high_leaf = DistanceAwareEvidentialExtractor(
            self.rf, kappa_leaf=4.0, c_spatial=1.0, lengthscale=1.0, spatial_metric="euclidean"
        )
        ext_high_leaf.fit(self.X_train, self.y_train)
        u_high_leaf = ext_high_leaf.extract_epistemic_signal(X_far)[0]
        self.assertGreater(u_high_leaf, u_base)

    def test_smac3_epm_rf_compatibility(self):
        """Interface 4: Verify compatibility with CustomUncertaintyRandomForest and SMAC3 facade."""
        from ConfigSpace import ConfigurationSpace, Float
        from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
        
        cs = ConfigurationSpace()
        cs.add([Float("x1", (0.0, 1.0)), Float("x2", (0.0, 1.0))])
        
        custom_rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs,
            n_trees=5,
            seed=42
        )
        custom_rf.train(self.X_train, self.y_train.reshape(-1, 1))
        
        X_test = np.array([[0.2, 0.3], [0.8, 0.9]])
        mean, var = custom_rf._predict(X_test)
        
        self.assertEqual(mean.shape, (2, 1))
        self.assertEqual(var.shape, (2, 1))
        self.assertTrue(np.all(var >= 0.0))
        self.assertTrue(np.all(np.isfinite(var)))

    def test_runtime_latency_5000_candidates(self):
        """Interface 5: Profile execution speed on 5000 candidates to guarantee < 20% BO overhead (< 100 ms)."""
        np.random.seed(42)
        N_train = 100
        D = 5
        N_cand = 5000
        
        X_train = np.random.uniform(0.0, 1.0, (N_train, D))
        y_train = np.sum(X_train, axis=1)
        
        rf = RandomForestRegressor(n_estimators=20, random_state=42)
        rf.fit(X_train, y_train)
        
        ext = DistanceAwareEvidentialExtractor(rf)
        ext.fit(X_train, y_train)
        
        X_cand = np.random.uniform(0.0, 1.0, (N_cand, D))
        
        # Warmup
        _ = ext.extract_epistemic_signal(X_cand[:10])
        
        t0 = time.perf_counter()
        u_cand = ext.extract_epistemic_signal(X_cand)
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        
        self.assertEqual(u_cand.shape, (N_cand,))
        self.assertTrue(np.all(np.isfinite(u_cand)))
        self.assertLess(
            t_elapsed, 
            100.0, 
            f"Candidate evaluation took {t_elapsed:.2f} ms (expected < 100 ms)."
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from ep_extractors.standard_disagreement import StandardDisagreementExtractor
from ep_extractors.shaker_entropy import ShakerEntropyExtractor


class TestEmpiricalChallengerM12(unittest.TestCase):
    """
    Empirical Challenge Harness by Challenger M1-2:
    Directly tests DistanceAwareEvidentialExtractor against StandardDisagreementExtractor
    and ShakerEntropyExtractor across:
    1. Overconfidence Trap in Extrapolation (DA-EHRF vs Standard Disagreement vs Shaker)
    2. Zero-Noise Invariant & Baseline Collapse (DA-EHRF vs Shaker Entropy)
    3. Default Bandwidth Suppression Defect (V_spatial drowned out in 2D interior gaps)
    4. Calibrated Bandwidth Gap Discovery (Verification of resolution mechanism)
    5. Density Monotonicity & Tree Leaf Regularization Sensitivity
    6. Extreme Asymptotic Saturation & Numerical Stability
    """

    def setUp(self):
        np.random.seed(42)

    def test_extrapolation_overconfidence_trap_comparison(self):
        """
        Challenge 1: Extrapolation Overconfidence Trap Comparison.
        Fit models on 1D training clusters [-5, -1] U [1, 5] (N=100 total).
        Evaluate along extrapolation ray x in [5.0, 10.0].
        - Standard Disagreement and Shaker Entropy freeze at a flat constant (spread < 1e-4).
        - DA-EHRF strictly increases monotonically with distance from training support.
        """
        x_left = np.random.uniform(-5.0, -1.0, 50)
        x_right = np.random.uniform(1.0, 5.0, 50)
        X_train = np.concatenate([x_left, x_right]).reshape(-1, 1)
        y_train = np.sin(X_train[:, 0]) + 0.1 * np.random.randn(len(X_train))

        rf = RandomForestRegressor(n_estimators=25, min_samples_leaf=2, random_state=42)
        rf.fit(X_train, y_train)

        ext_da = DistanceAwareEvidentialExtractor(rf, lengthscale=1.0)
        ext_da.fit(X_train, y_train)

        ext_std = StandardDisagreementExtractor(rf)
        ext_std.fit(X_train, y_train)

        ext_shaker = ShakerEntropyExtractor(rf, num_samples=2000, backend="cpu")
        ext_shaker.fit(X_train, y_train)

        x_extrap = np.linspace(5.0, 10.0, 11).reshape(-1, 1)
        u_da_extrap = ext_da.extract_epistemic_signal(x_extrap)
        u_std_extrap = ext_std.extract_epistemic_signal(x_extrap)
        u_shaker_extrap = ext_shaker.extract_epistemic_signal(x_extrap)

        std_extrap_spread = float(np.ptp(u_std_extrap[1:]))
        shaker_extrap_spread = float(np.ptp(u_shaker_extrap[1:]))
        da_extrap_diffs = np.diff(u_da_extrap)

        print("\n--- 1D Extrapolation Ray [5.0 -> 10.0] ---")
        for i in range(len(x_extrap)):
            print(f"x={x_extrap[i,0]:4.1f} | DA-EHRF: {u_da_extrap[i]:.4f} | StdDis: {u_std_extrap[i]:.4f} | Shaker: {u_shaker_extrap[i]:.4f}")

        # Verification 1A: Standard disagreement plateaus completely in extrapolation
        self.assertLess(
            std_extrap_spread,
            1e-5,
            f"Standard Disagreement failed expected plateau! spread={std_extrap_spread}"
        )
        # Verification 1B: Shaker entropy also plateaus completely in extrapolation
        self.assertLess(
            shaker_extrap_spread,
            1e-4,
            f"Shaker Entropy failed expected plateau! spread={shaker_extrap_spread}"
        )
        # Verification 1C: DA-EHRF strictly increases monotonically with distance
        self.assertTrue(
            np.all(da_extrap_diffs > 0.0),
            f"DA-EHRF failed monotonic growth! diffs={da_extrap_diffs}"
        )

    def test_zero_noise_invariant_vs_shaker_collapse(self):
        """
        Challenge 2: Zero-Noise Invariant (Deterministic Targets sigma_eps = 0).
        Train on exact quadratic y = x^2 on [-3, -1] U [1, 3] with 0 noise.
        Query gap center x = 0 and extrapolation x = 5.
        - Shaker Entropy collapses towards zero due to aleatoric variance coupling (~1e-6).
        - DA-EHRF retains strong epistemic signal scaled by target prior std (sigma_0).
        """
        x_left = np.linspace(-3.0, -1.0, 20)
        x_right = np.linspace(1.0, 3.0, 20)
        X_train = np.concatenate([x_left, x_right]).reshape(-1, 1)
        y_train = X_train[:, 0] ** 2  # Exactly zero observation noise

        rf = RandomForestRegressor(n_estimators=20, min_samples_leaf=2, random_state=42)
        rf.fit(X_train, y_train)

        ext_da = DistanceAwareEvidentialExtractor(rf)
        ext_da.fit(X_train, y_train)

        ext_std = StandardDisagreementExtractor(rf)
        ext_std.fit(X_train, y_train)

        ext_shaker = ShakerEntropyExtractor(rf, num_samples=2000, backend="cpu")
        ext_shaker.fit(X_train, y_train)

        pts = np.array([[0.0], [5.0]])
        u_da = ext_da.extract_epistemic_signal(pts)
        u_std = ext_std.extract_epistemic_signal(pts)
        u_shaker = ext_shaker.extract_epistemic_signal(pts)

        sigma_0 = float(np.std(y_train))
        print("\n--- Zero-Noise (sigma_eps = 0) Invariant Test ---")
        print(f"Target prior sigma_0: {sigma_0:.4f}")
        print(f"Gap Center x=0.0  | DA-EHRF: {u_da[0]:.4f} | StdDis: {u_std[0]:.4f} | Shaker: {u_shaker[0]:.6f}")
        print(f"Extrap     x=5.0  | DA-EHRF: {u_da[1]:.4f} | StdDis: {u_std[1]:.4f} | Shaker: {u_shaker[1]:.6f}")

        # DA-EHRF maintains high epistemic signal (> 0.5 * sigma_0)
        self.assertGreater(u_da[0], 0.5 * sigma_0)
        self.assertGreater(u_da[1], 0.5 * sigma_0)

        # Confirm Shaker Entropy collapses due to zero aleatoric noise coupling
        self.assertLess(
            u_shaker[0],
            0.05 * sigma_0,
            f"Expected Shaker to collapse under zero noise, but got {u_shaker[0]}"
        )
        collapse_factor = sigma_0 / max(float(u_shaker[0]), 1e-6)
        print(f"Empirical Finding: Shaker Entropy collapsed by {collapse_factor:.1f}x under zero observation noise.")

    def test_default_bandwidth_spatial_suppression_defect(self):
        """
        Challenge 3: Default Bandwidth Suppression Defect in 2D Interior Gaps.
        In 2D domain [-4, 4]^2 with central gap [-1.5, 1.5]^2:
        Because fit() maps coordinates to [0, 1]^2 by dividing by range (ptp=8.0),
        the default lengthscale=1.0 covers the entire domain.
        Consequently, V_spatial at gap center (0, 0) is tiny (~0.003), allowing
        boundary tree disagreement fluctuations to exceed the gap center uncertainty.
        This reproduces the empirical defect where gap center uncertainty does not peak.
        """
        points = []
        while len(points) < 200:
            cand = np.random.uniform(-4.0, 4.0, 2)
            if not (-1.5 <= cand[0] <= 1.5 and -1.5 <= cand[1] <= 1.5):
                points.append(cand)
        X_train = np.array(points)
        y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.05 * np.random.randn(len(X_train))

        rf = RandomForestRegressor(n_estimators=30, random_state=42)
        rf.fit(X_train, y_train)

        # Default extractor with lengthscale=1.0
        ext_default = DistanceAwareEvidentialExtractor(rf, lengthscale=1.0)
        ext_default.fit(X_train, y_train)

        pts = np.array([[0.0, 0.0], [0.8, 0.8], [1.5, 1.5]])
        u_def = ext_default.extract_epistemic_signal(pts)

        print("\n--- Defect Reproduction: Default Bandwidth (lengthscale=1.0) in 2D Gap ---")
        print(f"Gap Center   (0.0, 0.0): DA-EHRF U_E = {u_def[0]:.5f}")
        print(f"Gap Interior (0.8, 0.8): DA-EHRF U_E = {u_def[1]:.5f}")
        print(f"Gap Boundary (1.5, 1.5): DA-EHRF U_E = {u_def[2]:.5f}")

        # Empirically verify that under default lengthscale=1.0, gap center is NOT the local maximum
        # (center uncertainty is lower than off-center interior or boundary)
        center_is_suboptimal = (u_def[0] < u_def[1]) or (u_def[0] < u_def[2])
        print(f"Empirical Defect Confirmed: Center is lower than interior/boundary: {center_is_suboptimal}")
        self.assertTrue(
            center_is_suboptimal,
            "Failed to reproduce default bandwidth suppression defect."
        )

    def test_calibrated_bandwidth_gap_discovery(self):
        """
        Challenge 4: Calibrated Bandwidth Gap Discovery.
        When lengthscale is properly scaled to normalized inter-point distance (e.g. ell_0 = 0.15),
        V_spatial increases by > 15x at the gap center, restoring correct ordering:
        Gap Center (0,0) > Gap Interior (0.8,0.8) > Gap Boundary (1.5,1.5) > Dense ID (3,3).
        """
        points = []
        while len(points) < 200:
            cand = np.random.uniform(-4.0, 4.0, 2)
            if not (-1.5 <= cand[0] <= 1.5 and -1.5 <= cand[1] <= 1.5):
                points.append(cand)
        X_train = np.array(points)
        y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.05 * np.random.randn(len(X_train))

        rf = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, random_state=42)
        rf.fit(X_train, y_train)

        # Calibrated extractor with lengthscale=0.15
        ext_calibrated = DistanceAwareEvidentialExtractor(rf, lengthscale=0.15)
        ext_calibrated.fit(X_train, y_train)

        pts = np.array([
            [3.0, 3.0],    # Dense ID
            [1.5, 1.5],    # Gap Boundary
            [0.8, 0.8],    # Gap Interior
            [0.0, 0.0],    # Gap Center
            [6.0, 6.0],    # Extrapolation
        ])
        u_cal = ext_calibrated.extract_epistemic_signal(pts)

        print("\n--- Resolution Verification: Calibrated Bandwidth (lengthscale=0.15) ---")
        labels = ["Dense ID (3,3)", "Gap Boundary (1.5,1.5)", "Gap Interior (0.8,0.8)", "Gap Center (0,0)", "Extrapolation (6,6)"]
        for i in range(len(pts)):
            print(f"{labels[i]:25s}: DA-EHRF U_E = {u_cal[i]:.5f}")

        # Calibrated ordering verified
        self.assertGreater(u_cal[3], u_cal[2], "Center should exceed interior with calibrated lengthscale")
        self.assertGreater(u_cal[2], u_cal[1], "Interior should exceed boundary with calibrated lengthscale")
        self.assertGreater(u_cal[1], u_cal[0], "Boundary should exceed dense ID with calibrated lengthscale")
        self.assertGreater(u_cal[4], u_cal[0], "Extrapolation should exceed dense ID with calibrated lengthscale")

    def test_density_monotonicity_regularization_sensitivity(self):
        """
        Challenge 5: Density Monotonicity & Tree Regularization Sensitivity.
        Region A (dense, N=100) vs Region B (sparse, N=4).
        - Under unpruned trees (min_samples_leaf=1), leaf sample counts are 1 everywhere,
          so V_leaf is a global constant and fails to provide strong density penalty (ratio < 1.2x).
        - Under regularized trees (min_samples_leaf=2), V_leaf properly differentiates
          dense from sparse support, achieving a > 1.4x uncertainty ratio.
        """
        X_dense = np.random.uniform(0.0, 1.0, 100).reshape(-1, 1)
        X_sparse = np.random.uniform(4.0, 5.0, 4).reshape(-1, 1)
        X_train = np.vstack([X_dense, X_sparse])
        y_train = 2.0 * X_train[:, 0] + 0.1 * np.random.randn(len(X_train))

        pts = np.array([[0.5], [4.5]])

        # Case A: Default unpruned RF (min_samples_leaf=1)
        rf_unpruned = RandomForestRegressor(n_estimators=30, min_samples_leaf=1, random_state=42)
        rf_unpruned.fit(X_train, y_train)
        ext_unpruned = DistanceAwareEvidentialExtractor(rf_unpruned, kappa_leaf=2.0)
        ext_unpruned.fit(X_train, y_train)
        u_unpruned = ext_unpruned.extract_epistemic_signal(pts)
        ratio_unpruned = float(u_unpruned[1] / u_unpruned[0])

        # Case B: Regularized RF (min_samples_leaf=2)
        rf_reg = RandomForestRegressor(n_estimators=30, min_samples_leaf=2, random_state=42)
        rf_reg.fit(X_train, y_train)
        ext_reg = DistanceAwareEvidentialExtractor(rf_reg, kappa_leaf=2.0)
        ext_reg.fit(X_train, y_train)
        u_reg = ext_reg.extract_epistemic_signal(pts)
        ratio_reg = float(u_reg[1] / u_reg[0])

        print("\n--- Density Monotonicity & Tree Regularization Sensitivity ---")
        print(f"min_samples_leaf=1: Dense={u_unpruned[0]:.4f}, Sparse={u_unpruned[1]:.4f}, Ratio={ratio_unpruned:.2f}x")
        print(f"min_samples_leaf=2: Dense={u_reg[0]:.4f}, Sparse={u_reg[1]:.4f}, Ratio={ratio_reg:.2f}x")

        # Unpruned trees have suppressed ratio (< 1.25x)
        self.assertLess(ratio_unpruned, 1.25)
        # Regularized trees have strong density discrimination (> 1.40x)
        self.assertGreater(ratio_reg, 1.40)

    def test_extreme_saturation_and_numerical_invariants(self):
        """
        Challenge 6: Extreme Saturation Bounds and Numerical Invariants.
        Query at x = 1e2, 1e4, 1e8, -1e8.
        Ensure finite values, strict non-negativity, and asymptotic saturation.
        """
        X_train = np.linspace(-1.0, 1.0, 30).reshape(-1, 1)
        y_train = np.sin(X_train[:, 0])

        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        ext_da = DistanceAwareEvidentialExtractor(rf)
        ext_da.fit(X_train, y_train)

        extreme_pts = np.array([[1e2], [1e4], [1e8], [-1e8]])
        u_da = ext_da.extract_epistemic_signal(extreme_pts)

        print("\n--- Extreme Saturation Test ---")
        for i in range(len(extreme_pts)):
            print(f"x={extreme_pts[i,0]:10.1e} | DA-EHRF: {u_da[i]:.6f}")

        self.assertTrue(np.all(np.isfinite(u_da)))
        self.assertTrue(np.all(u_da > 0.0))
        np.testing.assert_allclose(u_da[1], u_da[2], atol=1e-5)


if __name__ == "__main__":
    unittest.main()

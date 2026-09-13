"""
Unit and Integration Test Suite for Milestone M2: Epistemic Acquisition Function Integration.

Covers:
1. CustomUncertaintyRandomForest with uncertainty_func="distance_evidential":
   - Variance parity: var == U_E(X)**2
   - Direct epistemic variance injection into SMAC3
   - SMAC3 native acquisition functions (EI, LCB, PI) evaluation
   - Output shape consistency (N, 1), non-negativity (var >= 0), numerical stability
   - Zero-variance, flat targets, and degenerate input edge cases
   - Inactive hyperparameter imputation in hierarchical spaces
2. AdditiveEpistemicAcquisition:
   - Decoupled score formulation: alpha(x) = norm(alpha_base(x)) + beta_t * norm(U_E(x))
   - Max-relative normalization invariants and scale equivariance
   - LowerConfidenceBound (LCB) unconditional translation shift and objective shift invariance
   - Strict Pareto dominance preservation under arbitrary positive beta_t
   - Flat base acquisition fallback (epistemic uncertainty drives search)
   - Flat epistemic uncertainty fallback (base acquisition drives search)
   - Edge cases: both flat, 1-candidate sets, dimension broadcasting guard
3. WarmupCosineScheduler:
   - Warmup stage plateau: beta_t == beta_max for t <= t_warmup
   - Cosine decay phase: analytical midpoint, continuity, and monotonic non-increasing property
   - Terminal budget: beta_t == beta_min for t >= T
   - Boundary conditions: T=0, T=1, warmup_ratio=0.0, warmup_ratio=1.0, out-of-bounds t
4. End-to-End Integration:
   - Multi-step simulated BO trajectory coupling CustomUncertaintyRandomForest,
     DistanceAwareEvidentialExtractor, AdditiveEpistemicAcquisition, and WarmupCosineScheduler.
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np
from scipy.stats import norm

# Ensure project root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ConfigSpace import ConfigurationSpace, Float, Categorical, EqualsCondition
from smac.acquisition.function import EI, LCB, PI

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.acquisitions import (
    BaseAcquisitionFunction,
    ExpectedImprovement,
    LowerConfidenceBound,
    ProbabilityOfImprovement,
    AdditiveEpistemicAcquisition,
    WarmupCosineScheduler,
    normalize_max_relative,
    AcquisitionRegistry,
)
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from ep_extractors import UQExtractorRegistry


class TestCustomUncertaintyRandomForestDistanceEvidential(unittest.TestCase):
    """
    Test suite verifying CustomUncertaintyRandomForest configured with
    DistanceAwareEvidentialExtractor ('distance_evidential').
    """

    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })

        # Training set in [-2, 2]^2
        self.n_train = 25
        self.X_train = np.random.uniform(-2.0, 2.0, size=(self.n_train, 2))
        # Non-linear quadratic objective
        self.y_train = (
            self.X_train[:, 0] ** 2
            + 0.5 * self.X_train[:, 1] ** 2
            + np.sin(self.X_train[:, 0] * np.pi)
        ).reshape(-1, 1)

        # In-distribution test points and out-of-distribution extrapolation points
        self.X_test_id = np.array([
            [0.0, 0.0],
            [1.0, -1.0],
            [-0.5, 0.5],
            [1.5, 1.5],
        ])
        self.X_test_ood = np.array([
            [4.5, 4.5],
            [-4.5, -4.5],
            [10.0, 10.0],
            [-10.0, 0.0],
        ])
        self.X_test_all = np.vstack([self.X_test_id, self.X_test_ood])

        self.rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=15,
            seed=42,
        )
        self.rf.train(self.X_train, self.y_train)

    def test_variance_parity_direct_formula(self):
        """
        Requirement 1: Variance Parity.
        Verify _predict(X) returns (mean, var) where var == U_E(X)**2
        and matches standalone DistanceAwareEvidentialExtractor.
        """
        mean, var = self.rf._predict(self.X_test_all)

        # Standalone ground-truth extraction directly from self.rf._rf
        standalone_extractor = DistanceAwareEvidentialExtractor(self.rf._rf)
        standalone_extractor.fit(self.X_train, self.y_train.flatten())
        expected_u_e = standalone_extractor.extract_epistemic_signal(self.X_test_all)
        expected_var = (expected_u_e ** 2).reshape(-1, 1)

        # 1. Variance parity against directly extracted epistemic standard deviation
        np.testing.assert_allclose(
            var,
            expected_var,
            rtol=1e-6,
            atol=1e-8,
            err_msg="var returned by _predict does not match (U_E(X)**2).reshape(-1, 1)",
        )

        # 2. Check internal extractor reference
        self.assertIsNotNone(self.rf.uq_extractor)
        self.assertIsInstance(self.rf.uq_extractor, DistanceAwareEvidentialExtractor)
        internal_u_e = self.rf.uq_extractor.extract_epistemic_signal(self.X_test_all)
        np.testing.assert_allclose(
            var,
            (internal_u_e ** 2).reshape(-1, 1),
            rtol=1e-7,
            atol=1e-9,
            err_msg="var mismatch with internal uq_extractor",
        )

    def test_shape_preservation_and_consistency(self):
        """
        Requirement 1: Shape Preservation.
        Verify mean.shape == (N, 1) and var.shape == (N, 1) across single and batch evaluations.
        """
        # Batch evaluation
        n_samples = len(self.X_test_all)
        mean_batch, var_batch = self.rf._predict(self.X_test_all)
        self.assertEqual(mean_batch.shape, (n_samples, 1))
        self.assertEqual(var_batch.shape, (n_samples, 1))

        # Single candidate evaluation (N = 1)
        x_single = np.array([[1.0, 2.0]])
        mean_single, var_single = self.rf._predict(x_single)
        self.assertEqual(mean_single.shape, (1, 1))
        self.assertEqual(var_single.shape, (1, 1))

        # Large candidate evaluation (N = 100)
        x_large = np.random.uniform(-4.0, 4.0, size=(100, 2))
        mean_large, var_large = self.rf._predict(x_large)
        self.assertEqual(mean_large.shape, (100, 1))
        self.assertEqual(var_large.shape, (100, 1))

        # 1D array input (D,) must raise ValueError per SMAC3 2D-array API contract
        x_1d = np.array([0.5, -0.5])
        with self.assertRaises(ValueError):
            self.rf._predict(x_1d)

    def test_non_negativity_and_numerical_stability(self):
        """
        Requirement 1: Non-negativity & Numerical Finiteness.
        Verify var >= 0 everywhere, with no NaNs or Infs even at extreme coordinates.
        """
        extreme_points = np.array([
            [1e4, 1e4],
            [-1e4, -1e4],
            [1e5, -1e5],
            [0.0, 0.0],
        ])
        mean, var = self.rf._predict(extreme_points)

        self.assertTrue(np.all(var >= 0.0), f"Negative variance detected: {var[var < 0]}")
        self.assertTrue(np.all(np.isfinite(mean)), "Non-finite values found in mean predictions.")
        self.assertTrue(np.all(np.isfinite(var)), "Non-finite values found in variance predictions.")

    def test_zero_variance_and_constant_targets(self):
        """
        Requirement 1: Zero-Variance & Degenerate Scenarios.
        Verify behavior when y_train is constant (zero variance in target space).
        """
        # Constant targets: y = 3.1415
        y_const = np.full((self.n_train, 1), 3.1415)
        rf_const = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        rf_const.train(self.X_train, y_const)

        mean_const, var_const = rf_const._predict(self.X_test_all)
        self.assertEqual(mean_const.shape, (len(self.X_test_all), 1))
        self.assertEqual(var_const.shape, (len(self.X_test_all), 1))
        self.assertTrue(np.all(var_const >= 0.0))
        self.assertTrue(np.all(np.isfinite(mean_const)))
        self.assertTrue(np.all(np.isfinite(var_const)))
        # Mean predictions should match the constant target value
        np.testing.assert_allclose(mean_const, 3.1415, rtol=1e-4)

    def test_zero_uncertainty_callable(self):
        """
        Requirement 1: Verify zero-variance behavior when custom callable returns strictly zero.
        """
        def exact_zero_unc(model, X, y):
            return np.zeros(len(X))

        rf_zero = CustomUncertaintyRandomForest(
            uncertainty_func=exact_zero_unc,
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        rf_zero.train(self.X_train, self.y_train)

        mean, var = rf_zero._predict(self.X_test_id)
        self.assertEqual(var.shape, (len(self.X_test_id), 1))
        self.assertTrue(np.all(var >= 1e-10), "Variance was not clamped to at least 1e-10")
        np.testing.assert_allclose(var, 1e-10, atol=1e-12)

    def test_train_capital_Y_kwarg_compatibility(self):
        """Verify train() accepts Y=... keyword argument (SMAC3 signature convention)."""
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        # Train using capital Y
        rf.train(self.X_train, Y=self.y_train)
        self.assertIsNotNone(rf.last_y)
        np.testing.assert_allclose(rf.last_y, self.y_train.flatten())
        self.assertIsNotNone(rf.uq_extractor)
        mean, var = rf._predict(self.X_test_id)
        self.assertEqual(mean.shape, (len(self.X_test_id), 1))
        self.assertEqual(var.shape, (len(self.X_test_id), 1))

        # Train using lowercase y
        rf_lower = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        rf_lower.train(self.X_train, y=self.y_train)
        self.assertIsNotNone(rf_lower.last_y)
        np.testing.assert_allclose(rf_lower.last_y, self.y_train.flatten())

    def test_extractor_property_alias(self):
        """Verify model.extractor property alias returns self.uq_extractor."""
        self.assertIsNotNone(self.rf.extractor)
        self.assertIs(self.rf.extractor, self.rf.uq_extractor)
        sig = self.rf.extractor.extract_epistemic_signal(self.X_test_id)
        self.assertEqual(sig.shape, (len(self.X_test_id),))

    def test_extractor_kwargs_passthrough(self):
        """Verify extractor_kwargs in __init__ is stored and passed to extractor."""
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=10,
            seed=42,
            extractor_kwargs={"kappa_leaf": 2.5, "c_spatial": 0.5},
        )
        self.assertEqual(rf.extractor_kwargs, {"kappa_leaf": 2.5, "c_spatial": 0.5})
        rf.train(self.X_train, self.y_train)
        self.assertEqual(rf.uq_extractor.kappa_leaf, 2.5)
        self.assertEqual(rf.uq_extractor.c_spatial, 0.5)

    def test_smac3_native_ei_evaluation(self):
        """
        Requirement 1: Native SMAC3 Acquisition Compatibility - Expected Improvement (EI).
        Verify EI._compute(X) executes without error, returns shape (N, 1), and is non-negative.
        """
        y_best = float(np.min(self.y_train))
        ei = EI(xi=0.0)
        ei.update(model=self.rf, eta=y_best)

        ei_scores = ei._compute(self.X_test_all)
        self.assertEqual(ei_scores.shape, (len(self.X_test_all), 1))
        self.assertTrue(np.all(ei_scores >= 0.0), f"Negative EI score detected: {ei_scores[ei_scores < 0]}")
        self.assertTrue(np.all(np.isfinite(ei_scores)), "Non-finite EI score detected.")

        # OOD points with high epistemic uncertainty should have non-zero exploration value
        ei_ood = ei._compute(self.X_test_ood)
        self.assertTrue(np.all(ei_ood >= 0.0))

    def test_smac3_native_lcb_evaluation(self):
        """
        Requirement 1: Native SMAC3 Acquisition Compatibility - Lower Confidence Bound (LCB).
        Verify LCB._compute(X) executes without error, returns shape (N, 1), and is finite.
        """
        lcb = LCB(beta=2.0)
        lcb.update(model=self.rf, num_data=len(self.X_train))

        lcb_scores = lcb._compute(self.X_test_all)
        self.assertEqual(lcb_scores.shape, (len(self.X_test_all), 1))
        self.assertTrue(np.all(np.isfinite(lcb_scores)), "Non-finite LCB score detected.")

        # In SMAC3, LCB returns -(mean - beta*std) = -mean + beta*std (higher is better)
        mean, var = self.rf._predict(self.X_test_all)
        std = np.sqrt(var)
        # Beta schedule from SMAC3
        beta_t = 2.0 * np.log((self.X_test_all.shape[1] * (len(self.X_train) ** 2)) / 2.0)
        expected_lcb = -mean + np.sqrt(beta_t) * std
        np.testing.assert_allclose(lcb_scores, expected_lcb, rtol=1e-5)

    def test_smac3_native_pi_evaluation(self):
        """
        Requirement 1: Native SMAC3 Acquisition Compatibility - Probability of Improvement (PI).
        Verify PI._compute(X) executes without error, returns shape (N, 1), and lies in [0, 1].
        """
        y_best = float(np.min(self.y_train))
        pi = PI(xi=0.0)
        pi.update(model=self.rf, eta=y_best)

        pi_scores = pi._compute(self.X_test_all)
        self.assertEqual(pi_scores.shape, (len(self.X_test_all), 1))
        self.assertTrue(np.all(np.isfinite(pi_scores)), "Non-finite PI score detected.")
        self.assertTrue(
            np.all((pi_scores >= 0.0) & (pi_scores <= 1.0 + 1e-7)),
            f"PI scores outside [0, 1]: {pi_scores[(pi_scores < 0) | (pi_scores > 1)]}",
        )

    def test_hierarchical_conditional_space_imputation(self):
        """
        Requirement 1: Hierarchical / Conditional Configuration Spaces.
        Verify inactive hyperparameters (NaNs) are cleanly imputed during train and predict.
        """
        cs_hier = ConfigurationSpace(seed=42)
        model_type = Categorical("model_type", ["rf", "svm"], default="rf")
        rf_trees = Float("rf_trees", (10.0, 100.0), default=50.0)
        svm_c = Float("svm_c", (0.1, 10.0), default=1.0)
        cs_hier.add([model_type, rf_trees, svm_c])
        cs_hier.add(EqualsCondition(rf_trees, model_type, "rf"))
        cs_hier.add(EqualsCondition(svm_c, model_type, "svm"))

        # Synthetic dataset with conditional NaNs
        # col 0: model_type (0=rf, 1=svm), col 1: rf_trees, col 2: svm_c
        X_hier = np.array([
            [0.0, 50.0, np.nan],
            [0.0, 75.0, np.nan],
            [1.0, np.nan, 2.5],
            [1.0, np.nan, 5.0],
            [0.0, 20.0, np.nan],
            [1.0, np.nan, 0.5],
        ])
        y_hier = np.array([[1.0], [2.0], [1.5], [3.0], [0.8], [2.2]])

        rf_hier = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs_hier,
            n_trees=10,
            seed=42,
        )
        rf_hier.train(X_hier, y_hier)

        X_query = np.array([
            [0.0, 60.0, np.nan],
            [1.0, np.nan, 1.2],
        ])
        mean, var = rf_hier._predict(X_query)
        self.assertEqual(mean.shape, (2, 1))
        self.assertEqual(var.shape, (2, 1))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(var)))
        self.assertTrue(np.all(var >= 0.0))


class TestAdditiveEpistemicAcquisitionProperties(unittest.TestCase):
    """
    Test suite verifying mathematical invariants, normalization properties,
    translation invariance, and boundary conditions of AdditiveEpistemicAcquisition.
    """

    def setUp(self):
        np.random.seed(42)
        self.preds = np.array([1.0, 2.0, 0.5, 3.0])
        self.unc_tot = np.array([0.5, 1.0, 0.1, 0.8])
        self.u_ep = np.array([0.2, 0.8, 0.1, 0.6])
        self.y_best = 1.0
        self.base_ei = ExpectedImprovement(xi=0.0)
        self.additive_ei = AdditiveEpistemicAcquisition(base_acq=self.base_ei)

    def test_decoupled_score_exact_formula(self):
        """
        Requirement 2: Decoupled Score Formulation.
        Verify alpha(x) = norm(alpha_base(x)) + beta_t * norm(U_E(x)).
        """
        raw_base = self.base_ei.compute(self.preds, self.unc_tot, self.y_best)
        norm_base = normalize_max_relative(raw_base)
        norm_ep = normalize_max_relative(self.u_ep)

        # beta_t = 0.0 -> pure base acquisition
        score_0 = self.additive_ei.compute_additive(
            self.preds, self.unc_tot, self.u_ep, self.y_best, beta_t=0.0
        )
        np.testing.assert_allclose(score_0, norm_base, rtol=1e-6)

        # beta_t = 1.0 -> norm_base + norm_ep
        score_1 = self.additive_ei.compute_additive(
            self.preds, self.unc_tot, self.u_ep, self.y_best, beta_t=1.0
        )
        np.testing.assert_allclose(score_1, norm_base + norm_ep, rtol=1e-6)

        # Arbitrary beta_t = 2.75 -> linear scaling in epistemic term
        beta = 2.75
        score_beta = self.additive_ei.compute_additive(
            self.preds, self.unc_tot, self.u_ep, self.y_best, beta_t=beta
        )
        np.testing.assert_allclose(score_beta, norm_base + beta * norm_ep, rtol=1e-6)

    def test_max_relative_normalization_invariants(self):
        """
        Requirement 2: Max-Relative Normalization to [0, 1].
        Verify scale invariance, non-negativity, and numerical stability.
        """
        arr = np.array([0.0, 2.5, 5.0, 10.0])
        normed = normalize_max_relative(arr)

        # Max element should map to approximately 1.0
        self.assertAlmostEqual(normed[-1], 1.0, places=6)
        # Relative ratios preserved
        np.testing.assert_allclose(normed, np.array([0.0, 0.25, 0.5, 1.0]), rtol=1e-5)
        self.assertTrue(np.all(normed >= 0.0))

        # Scale invariance: norm(c * arr) == norm(arr) for any positive scalar c
        for c in [1e-4, 0.1, 2.5, 100.0, 1e6]:
            normed_scaled = normalize_max_relative(c * arr)
            np.testing.assert_allclose(
                normed_scaled,
                normed,
                rtol=1e-5,
                err_msg=f"Scale invariance failed for c={c}",
            )

        # Degenerate all-zeros or negative inputs
        arr_zero = np.zeros(5)
        np.testing.assert_allclose(normalize_max_relative(arr_zero), np.zeros(5))

        arr_neg = np.array([-5.0, -2.0, -0.1])
        np.testing.assert_allclose(normalize_max_relative(arr_neg), np.zeros(3))

        # NaN / Inf sanitization
        arr_dirty = np.array([np.nan, np.inf, -np.inf, 4.0, 2.0])
        normed_dirty = normalize_max_relative(arr_dirty)
        self.assertTrue(np.all(np.isfinite(normed_dirty)))
        self.assertAlmostEqual(normed_dirty[3], 1.0, places=5)

    def test_lcb_translation_invariance(self):
        """
        Requirement 2: LCB Translation Shift & Invariance.
        Verify Additive LCB is strictly invariant under arbitrary constant objective shifts.
        """
        base_lcb = LowerConfidenceBound(beta=2.0)
        additive_lcb = AdditiveEpistemicAcquisition(base_acq=base_lcb)

        scores_orig = additive_lcb.compute_additive(
            self.preds, self.unc_tot, self.u_ep, self.y_best, beta_t=1.5
        )

        for offset in [-1e6, -500.0, -10.0, 25.0, 1000.0, 1e6]:
            preds_shifted = self.preds + offset
            y_best_shifted = self.y_best + offset
            scores_shifted = additive_lcb.compute_additive(
                preds_shifted,
                self.unc_tot,
                self.u_ep,
                y_best=y_best_shifted,
                beta_t=1.5,
            )
            np.testing.assert_allclose(
                scores_shifted,
                scores_orig,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"LCB translation invariance violated for offset={offset}",
            )

    def test_strict_pareto_dominance_preservation(self):
        """
        Requirement 2: Strict Pareto Dominance Preservation.
        If candidate A strictly dominates B in both alpha_base and U_E,
        then alpha_add(A) > alpha_add(B) for all beta_t > 0.
        """
        # Candidate 0: preds=0.2 (lower mean), u_ep=1.0 (higher uncertainty)
        # Candidate 1: preds=2.0 (worse mean), u_ep=0.2 (lower uncertainty)
        preds = np.array([0.2, 2.0])
        unc_tot = np.array([0.8, 0.3])
        u_ep = np.array([1.0, 0.2])
        y_best = 1.0

        for acq_name in ["ei", "lcb", "pi"]:
            base_acq = AcquisitionRegistry.get(acq_name)
            additive = AdditiveEpistemicAcquisition(base_acq=base_acq)

            # Check Pareto dominance holds across multiple orders of magnitude of beta_t
            for beta in [1e-3, 0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 1e3]:
                scores = additive.compute_additive(
                    preds, unc_tot, u_ep, y_best=y_best, beta_t=beta
                )
                self.assertGreater(
                    scores[0],
                    scores[1],
                    f"Pareto dominance violated for acq={acq_name} with beta_t={beta}: "
                    f"score[0]={scores[0]}, score[1]={scores[1]}",
                )

    def test_flat_base_acquisition_edge_case(self):
        """
        Requirement 2: Edge Case - Constant Base Acquisition.
        When alpha_base is flat across all candidates, uncertainty completely drives the ranking.
        """
        # High predictions so Expected Improvement is 0.0 everywhere
        preds_high = np.array([500.0, 500.0, 500.0])
        unc_tiny = np.array([1e-6, 1e-6, 1e-6])
        u_ep = np.array([0.1, 0.9, 0.4])  # Candidate 1 has highest uncertainty
        y_best = 0.0

        scores = self.additive_ei.compute_additive(
            preds_high, unc_tiny, u_ep, y_best=y_best, beta_t=1.0
        )

        self.assertTrue(np.all(np.isfinite(scores)))
        # Argmax must be candidate 1
        self.assertEqual(int(np.argmax(scores)), 1)
        # Full ranks must match u_ep ranks exactly
        np.testing.assert_array_equal(np.argsort(scores), np.argsort(u_ep))

    def test_flat_epistemic_uncertainty_edge_case(self):
        """
        Requirement 2: Edge Case - Constant Uncertainty.
        When U_E is flat across all candidates, base acquisition completely drives the ranking.
        """
        preds = np.array([0.1, 0.5, 2.0])
        unc_tot = np.array([0.3, 0.3, 0.3])
        u_ep_flat = np.array([0.5, 0.5, 0.5])  # Uniform epistemic ignorance
        y_best = 1.0

        scores = self.additive_ei.compute_additive(
            preds, unc_tot, u_ep_flat, y_best=y_best, beta_t=1.0
        )

        self.assertTrue(np.all(np.isfinite(scores)))
        raw_base = self.base_ei.compute(preds, unc_tot, y_best)
        # Argmax must be candidate 0 (highest EI)
        self.assertEqual(int(np.argmax(scores)), int(np.argmax(raw_base)))
        np.testing.assert_array_equal(np.argsort(scores), np.argsort(raw_base))

    def test_both_flat_edge_case(self):
        """
        Requirement 2: Edge Case - Both Flat Base and Uncertainty.
        Verify no division by zero or NaN occurs when all inputs are constant.
        """
        preds_const = np.array([10.0, 10.0, 10.0])
        unc_const = np.array([0.1, 0.1, 0.1])
        u_ep_const = np.array([0.5, 0.5, 0.5])
        y_best = 5.0

        scores = self.additive_ei.compute_additive(
            preds_const, unc_const, u_ep_const, y_best=y_best, beta_t=1.0
        )

        self.assertEqual(len(scores), 3)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))
        # All candidates have identical score
        np.testing.assert_allclose(scores, scores[0])

    def test_dimension_and_broadcasting_guard(self):
        """
        Requirement 2: Dimension Handling & Defensive Flattening.
        Verify 1D, 2D (N, 1), and mixed (N, 1) / (N,) inputs return 1D scores of shape (N,),
        preventing inadvertent (N, N) outer-product broadcasting.
        """
        # 1D inputs
        preds_1d = np.array([1.0, 2.0, 0.5])
        unc_1d = np.array([0.2, 0.3, 0.1])
        u_ep_1d = np.array([0.4, 0.1, 0.7])
        scores_1d = self.additive_ei.compute_additive(
            preds_1d, unc_1d, u_ep_1d, y_best=0.5, beta_t=1.0
        )
        self.assertEqual(scores_1d.shape, (3,))

        # 2D column vector inputs (N, 1) defensively flattened to 1D
        preds_2d = preds_1d.reshape(-1, 1)
        unc_2d = unc_1d.reshape(-1, 1)
        u_ep_2d = u_ep_1d.reshape(-1, 1)
        scores_2d = self.additive_ei.compute_additive(
            preds_2d, unc_2d, u_ep_2d, y_best=0.5, beta_t=1.0
        )
        self.assertEqual(scores_2d.shape, (3,))
        np.testing.assert_allclose(scores_1d, scores_2d)

        # Mixed inputs: preds (3, 1), unc (3,), u_ep (3, 1)
        scores_mixed = self.additive_ei.compute_additive(
            preds_2d, unc_1d, u_ep_2d, y_best=0.5, beta_t=1.0
        )
        self.assertEqual(scores_mixed.shape, (3,))
        np.testing.assert_allclose(scores_1d, scores_mixed)

    def test_defensive_1d_flattening_all_base_acquisitions(self):
        """
        Verify ExpectedImprovement, LowerConfidenceBound, and ProbabilityOfImprovement
        defensively flatten (N, 1) and (N,) inputs and return 1D (N,) arrays without (N, N) broadcasting.
        """
        preds_2d = np.array([[1.0], [2.0], [0.5]])
        unc_1d = np.array([0.2, 0.3, 0.1])
        unc_2d = unc_1d.reshape(-1, 1)
        preds_1d = preds_2d.ravel()

        ei = ExpectedImprovement(xi=0.0)
        lcb = LowerConfidenceBound(beta=2.0)
        pi = ProbabilityOfImprovement(xi=0.0)

        for acq in [ei, lcb, pi]:
            # Mixed: (N, 1) preds, (N,) unc
            res1 = acq.compute(preds_2d, unc_1d, y_best=0.5)
            self.assertEqual(res1.shape, (3,))
            # Mixed: (N,) preds, (N, 1) unc
            res2 = acq.compute(preds_1d, unc_2d, y_best=0.5)
            self.assertEqual(res2.shape, (3,))
            # 2D: (N, 1) preds, (N, 1) unc
            res3 = acq.compute(preds_2d, unc_2d, y_best=0.5)
            self.assertEqual(res3.shape, (3,))
            # 1D: (N,) preds, (N,) unc
            res4 = acq.compute(preds_1d, unc_1d, y_best=0.5)
            self.assertEqual(res4.shape, (3,))

            np.testing.assert_allclose(res1, res4)
            np.testing.assert_allclose(res2, res4)
            np.testing.assert_allclose(res3, res4)

    def test_single_candidate_boundary(self):
        """
        Requirement 2: Edge Case - Single Candidate Set (N = 1).
        Verify evaluation on a single point behaves correctly.
        """
        preds_single = np.array([1.0])
        unc_single = np.array([0.5])
        u_ep_single = np.array([0.3])
        scores = self.additive_ei.compute_additive(
            preds_single, unc_single, u_ep_single, y_best=1.0, beta_t=1.0
        )
        self.assertEqual(len(scores), 1)
        self.assertTrue(np.isfinite(scores[0]))
        self.assertTrue(scores[0] >= 0.0)


class TestWarmupCosineSchedulerInvariants(unittest.TestCase):
    """
    Test suite verifying WarmupCosineScheduler properties:
    - Warmup stage plateau (t <= t_warmup: beta_t == beta_max)
    - Cosine decay stage (t_warmup < t < T: smooth monotonic decay)
    - Terminal budget (t >= T: beta_t == beta_min)
    - Boundary and edge case handling
    """

    def test_warmup_phase_exact_plateau(self):
        """
        Requirement 3: Warmup Stage.
        For t <= t_warmup, beta_t == beta_max exactly.
        """
        total_trials = 50
        warmup_ratio = 0.20  # t_warmup = floor(0.20 * 50) = 10
        beta_max = 2.5
        beta_min = 0.1

        scheduler = WarmupCosineScheduler(
            total_trials=total_trials,
            warmup_ratio=warmup_ratio,
            beta_max=beta_max,
            beta_min=beta_min,
        )
        self.assertEqual(scheduler.t_warmup, 10)

        for t in range(11):  # t = 0 .. 10
            beta_t = scheduler.get_beta(t)
            self.assertEqual(
                beta_t,
                beta_max,
                f"Warmup violated at t={t}: expected {beta_max}, got {beta_t}",
            )

    def test_cosine_decay_trajectory_and_monotonicity(self):
        """
        Requirement 3: Cosine Decay Phase.
        For t_warmup < t < T, verify analytical midpoint, continuity, and monotonic non-increasing decay.
        """
        total_trials = 50
        warmup_ratio = 0.20  # t_warmup = 10, remaining = 40
        beta_max = 1.0
        beta_min = 0.0

        scheduler = WarmupCosineScheduler(
            total_trials=total_trials,
            warmup_ratio=warmup_ratio,
            beta_max=beta_max,
            beta_min=beta_min,
        )

        # Midpoint check: t = 10 + 20 = 30 -> progress = 0.5 -> cos(pi/2) = 0 -> beta = 0.5
        beta_mid = scheduler.get_beta(30)
        self.assertAlmostEqual(beta_mid, 0.5, places=6)

        # Monotonic non-increasing decay from t = 10 to t = 50
        prev_beta = scheduler.get_beta(10)
        for t in range(11, 51):
            curr_beta = scheduler.get_beta(t)
            self.assertLessEqual(
                curr_beta,
                prev_beta + 1e-12,
                f"Non-monotonicity detected at step t={t}: prev={prev_beta}, curr={curr_beta}",
            )
            prev_beta = curr_beta

    def test_terminal_budget_endpoint(self):
        """
        Requirement 3: Terminal Budget.
        For t >= T, beta_t == beta_min exactly.
        """
        scheduler = WarmupCosineScheduler(
            total_trials=50,
            warmup_ratio=0.20,
            beta_max=1.0,
            beta_min=0.05,
        )
        # At horizon T
        self.assertAlmostEqual(scheduler.get_beta(50), 0.05, places=6)
        # Beyond horizon T
        self.assertAlmostEqual(scheduler.get_beta(51), 0.05, places=6)
        self.assertAlmostEqual(scheduler.get_beta(100), 0.05, places=6)

    def test_zero_warmup_ratio(self):
        """
        Requirement 3: Edge Case - Zero Warmup (warmup_ratio = 0.0).
        Pure cosine decay begins immediately at t > 0.
        """
        scheduler = WarmupCosineScheduler(
            total_trials=40,
            warmup_ratio=0.0,
            beta_max=1.0,
            beta_min=0.0,
        )
        self.assertEqual(scheduler.t_warmup, 0)
        self.assertAlmostEqual(scheduler.get_beta(0), 1.0, places=6)
        self.assertAlmostEqual(scheduler.get_beta(20), 0.5, places=6)
        self.assertAlmostEqual(scheduler.get_beta(40), 0.0, places=6)

    def test_full_warmup_ratio(self):
        """
        Requirement 3: Edge Case - Full Warmup (warmup_ratio = 1.0).
        Beta remains beta_max for entire budget until terminal horizon.
        """
        scheduler = WarmupCosineScheduler(
            total_trials=30,
            warmup_ratio=1.0,
            beta_max=2.0,
            beta_min=0.0,
        )
        self.assertEqual(scheduler.t_warmup, 30)
        for t in range(31):  # t = 0 .. 30 remain at beta_max
            self.assertEqual(scheduler.get_beta(t), 2.0)
        # Beyond budget (t = 31) transitions to beta_min
        self.assertEqual(scheduler.get_beta(31), 0.0)

    def test_zero_or_single_budget_edge_case(self):
        """
        Requirement 3: Edge Case - T = 0 or T = 1.
        Verify no division by zero; handles cleanly.
        """
        # T = 0 clamped to max(1, 0) = 1
        sched_zero = WarmupCosineScheduler(total_trials=0, beta_max=1.0, beta_min=0.0)
        self.assertEqual(sched_zero.total_trials, 1)
        self.assertAlmostEqual(sched_zero.get_beta(0), 1.0, places=6)
        self.assertAlmostEqual(sched_zero.get_beta(1), 0.0, places=6)

        # T = 1
        sched_one = WarmupCosineScheduler(total_trials=1, beta_max=1.0, beta_min=0.0)
        self.assertAlmostEqual(sched_one.get_beta(0), 1.0, places=6)
        self.assertAlmostEqual(sched_one.get_beta(1), 0.0, places=6)

    def test_negative_step_and_out_of_bounds(self):
        """
        Requirement 3: Boundary Handling.
        Negative steps return beta_max; far future steps return beta_min.
        """
        scheduler = WarmupCosineScheduler(total_trials=50, beta_max=1.5, beta_min=0.2)
        self.assertEqual(scheduler.get_beta(-5), 1.5)
        self.assertEqual(scheduler.get_beta(-1), 1.5)
        self.assertEqual(scheduler.get_beta(500), 0.2)

    def test_custom_beta_bounds_and_types(self):
        """
        Requirement 3: Custom Beta Bounds & Return Type.
        Verify float return type and non-standard beta ranges.
        """
        scheduler = WarmupCosineScheduler(
            total_trials=20,
            beta_max=5.0,
            beta_min=1.0,
        )
        beta_val = scheduler.get_beta(10)
        self.assertIsInstance(beta_val, float)
        self.assertGreaterEqual(beta_val, 1.0)
        self.assertLessEqual(beta_val, 5.0)

        # Flat schedule (beta_max == beta_min)
        sched_flat = WarmupCosineScheduler(
            total_trials=20,
            beta_max=3.0,
            beta_min=3.0,
        )
        for t in [0, 5, 10, 15, 20, 25]:
            self.assertEqual(sched_flat.get_beta(t), 3.0)


class TestEpistemicAcquisitionEndToEndIntegration(unittest.TestCase):
    """
    End-to-End Integration Suite coupling CustomUncertaintyRandomForest,
    DistanceAwareEvidentialExtractor, AdditiveEpistemicAcquisition, and WarmupCosineScheduler.
    """

    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-3.0, 3.0), default=0.0),
            "x2": Float("x2", bounds=(-3.0, 3.0), default=0.0),
        })

        # Synthetic 2D objective (Rosenbrock-like valley)
        def objective(x):
            return 100.0 * (x[:, 1] - x[:, 0] ** 2) ** 2 + (1.0 - x[:, 0]) ** 2

        self.objective = objective

        # Dense initial cluster in [-1, 1]^2
        self.X_train = np.random.uniform(-1.0, 1.0, size=(20, 2))
        self.y_train = self.objective(self.X_train).reshape(-1, 1)

        self.rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=15,
            seed=42,
        )
        self.rf.train(self.X_train, self.y_train)

    def test_e2e_simulated_bo_loop(self):
        """
        Verify end-to-end integration across a simulated multi-step BO budget.
        Ensure acquisition values transition from exploration-heavy to exploitation-focused.
        """
        total_trials = 20
        scheduler = WarmupCosineScheduler(
            total_trials=total_trials,
            warmup_ratio=0.25,  # warmup for t <= 5
            beta_max=2.0,
            beta_min=0.0,
        )
        base_ei = ExpectedImprovement(xi=0.01)
        additive_acq = AdditiveEpistemicAcquisition(base_acq=base_ei)

        # Candidate grid covering both known region [-1, 1] and gap region [1.5, 3.0]
        X_cand = np.array([
            [0.1, 0.1],     # Dense in-distribution point
            [2.5, 2.5],     # Far unexplored gap point (high epistemic uncertainty)
            [0.9, 0.9],     # Near training optimum
        ])

        y_best = float(np.min(self.y_train))
        scores_history = []
        betas_history = []

        for t in range(total_trials):
            beta_t = scheduler.get_beta(t)
            betas_history.append(beta_t)

            mean, var = self.rf._predict(X_cand)
            std_tot = np.sqrt(var).flatten()
            u_ep = self.rf.uq_extractor.extract_epistemic_signal(X_cand)

            scores = additive_acq.compute_additive(
                preds=mean.flatten(),
                unc_tot=std_tot,
                u_epistemic=u_ep,
                y_best=y_best,
                beta_t=beta_t,
            )
            self.assertEqual(scores.shape, (3,))
            self.assertTrue(np.all(np.isfinite(scores)))
            self.assertTrue(np.all(scores >= 0.0))
            scores_history.append(scores)

        # 1. Warmup period (t = 0 to 5): beta_t is maximal (2.0)
        for t in range(6):
            self.assertEqual(betas_history[t], 2.0)

        # 2. Terminal stage (t = 19): beta_t is near zero
        self.assertLess(betas_history[-1], 0.1)

        # 3. Verify that the epistemic exploration bonus for the unexplored gap point (index 1) decays
        # The exploration score component beta_t * norm_ep is strictly higher in early warmup than at the end
        early_bonus = scores_history[0][1] - scores_history[-1][1]
        self.assertGreater(
            early_bonus,
            0.5,
            "Exploration bonus for OOD gap candidate did not decay significantly from early to late stage.",
        )
        self.assertGreater(scores_history[0][1], scores_history[-1][1])

    def test_e2e_ood_exploration_incentive(self):
        """
        Verify that out-of-distribution points are strongly incentivized when beta_t is high,
        and purely exploited according to base objective when beta_t is zero.
        """
        # Candidate A: In-distribution point near optimum (low epistemic uncertainty)
        # Candidate B: Out-of-distribution gap point (high epistemic uncertainty)
        cand_id = np.array([[0.0, 0.0]])
        cand_ood = np.array([[2.8, 2.8]])
        X_eval = np.vstack([cand_id, cand_ood])

        mean, var = self.rf._predict(X_eval)
        u_ep = self.rf.uq_extractor.extract_epistemic_signal(X_eval)
        self.assertGreater(
            u_ep[1],
            u_ep[0],
            "OOD candidate must have higher epistemic uncertainty than ID candidate.",
        )

        base_lcb = LowerConfidenceBound(beta=1.0)
        additive_lcb = AdditiveEpistemicAcquisition(base_acq=base_lcb)

        # High beta (pure exploration boost)
        score_explore = additive_lcb.compute_additive(
            mean.flatten(), np.sqrt(var).flatten(), u_ep, y_best=0.0, beta_t=5.0
        )
        # Zero beta (pure exploitation)
        score_exploit = additive_lcb.compute_additive(
            mean.flatten(), np.sqrt(var).flatten(), u_ep, y_best=0.0, beta_t=0.0
        )

        # Epistemic exploration adds a much larger boost to candidate B than candidate A
        boost_ood = score_explore[1] - score_exploit[1]
        boost_id = score_explore[0] - score_exploit[0]
        self.assertGreater(boost_ood, boost_id)


if __name__ == "__main__":
    unittest.main()

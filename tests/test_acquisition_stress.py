"""
Adversarial Stress Test Suite for Milestone M2:
1. Multidimensional tensor inputs (1D, 2D col, 2D row, single candidate, massive pool N=10,000).
2. Numerical extremes (negative LCB shifts, all-zero arrays, U_E -> 0, extreme beta_t in [0.0, 1000.0]).
3. Strict Pareto dominance preservation across acquisition functions.
4. Custom uncertainty model input shapes, return contracts, and edge cases.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
import numpy as np
from scipy.stats import norm

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ConfigSpace import ConfigurationSpace, Float, Categorical, EqualsCondition
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
from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from ep_extractors import UQExtractorRegistry


class TestAcquisitionTensorShapeHazards(unittest.TestCase):
    """Adversarial stress testing of tensor input shapes across acquisition functions."""

    def setUp(self):
        np.random.seed(42)
        self.N = 15
        self.preds_1d = np.linspace(0.1, 5.0, self.N)
        self.unc_1d = np.linspace(0.01, 1.5, self.N)
        self.y_best = 0.5

    def test_shape_permutations_base_acquisitions(self):
        """Test combinations of 1D, 2D col, 2D row across EI, LCB, PI."""
        acqs = [
            ExpectedImprovement(xi=0.01),
            LowerConfidenceBound(beta=2.0),
            ProbabilityOfImprovement(xi=0.01),
        ]

        shapes = [
            ("1D", self.preds_1d, self.unc_1d),
            ("2D_col", self.preds_1d.reshape(-1, 1), self.unc_1d.reshape(-1, 1)),
            ("2D_row", self.preds_1d.reshape(1, -1), self.unc_1d.reshape(1, -1)),
            ("mixed_col_row", self.preds_1d.reshape(-1, 1), self.unc_1d.reshape(1, -1)),
            ("mixed_row_col", self.preds_1d.reshape(1, -1), self.unc_1d.reshape(-1, 1)),
            ("mixed_1d_col", self.preds_1d, self.unc_1d.reshape(-1, 1)),
            ("mixed_col_1d", self.preds_1d.reshape(-1, 1), self.unc_1d),
        ]

        for acq in acqs:
            baseline = acq.compute(self.preds_1d, self.unc_1d, self.y_best)
            self.assertEqual(baseline.shape, (self.N,))

            for desc, p, u in shapes:
                with self.subTest(acq=acq.__class__.__name__, shape=desc):
                    res = acq.compute(p, u, self.y_best)
                    self.assertIsInstance(res, np.ndarray)
                    self.assertEqual(res.shape, (self.N,))
                    np.testing.assert_allclose(res, baseline, rtol=1e-10, atol=1e-10)

    def test_shape_permutations_additive_acquisition(self):
        """Test combinations of 1D, 2D col, 2D row across AdditiveEpistemicAcquisition."""
        u_ep_1d = np.linspace(0.05, 0.8, self.N)
        base_acqs = [
            ExpectedImprovement(xi=0.01),
            LowerConfidenceBound(beta=2.0),
            ProbabilityOfImprovement(xi=0.01),
        ]

        shape_combos = [
            (self.preds_1d, self.unc_1d, u_ep_1d),
            (self.preds_1d.reshape(-1, 1), self.unc_1d.reshape(-1, 1), u_ep_1d.reshape(-1, 1)),
            (self.preds_1d.reshape(1, -1), self.unc_1d.reshape(1, -1), u_ep_1d.reshape(1, -1)),
            (self.preds_1d.reshape(-1, 1), self.unc_1d.reshape(1, -1), u_ep_1d),
            (self.preds_1d, self.unc_1d.reshape(-1, 1), u_ep_1d.reshape(1, -1)),
            (self.preds_1d.reshape(1, -1), self.unc_1d, u_ep_1d.reshape(-1, 1)),
        ]

        for base in base_acqs:
            add_acq = AdditiveEpistemicAcquisition(base_acq=base)
            baseline = add_acq.compute_additive(self.preds_1d, self.unc_1d, u_ep_1d, self.y_best, beta_t=1.5)
            self.assertEqual(baseline.shape, (self.N,))

            for i, (p, u, ue) in enumerate(shape_combos):
                with self.subTest(base=base.__class__.__name__, combo=i):
                    res = add_acq.compute_additive(p, u, ue, self.y_best, beta_t=1.5)
                    self.assertIsInstance(res, np.ndarray)
                    self.assertEqual(res.shape, (self.N,))
                    np.testing.assert_allclose(res, baseline, rtol=1e-10, atol=1e-10)

    def test_single_candidate_evaluation(self):
        """Single candidate N=1 evaluation for scalars, 1D, and 2D arrays."""
        base_acqs = [
            ExpectedImprovement(xi=0.0),
            LowerConfidenceBound(beta=2.0),
            ProbabilityOfImprovement(xi=0.0),
        ]

        shapes_n1 = [
            (1.0, 0.5, 0.2),  # scalars
            (np.array([1.0]), np.array([0.5]), np.array([0.2])),  # 1D
            (np.array([[1.0]]), np.array([[0.5]]), np.array([[0.2]])),  # 2D (1,1)
        ]

        for base in base_acqs:
            add_acq = AdditiveEpistemicAcquisition(base_acq=base)
            for p, u, ue in shapes_n1:
                with self.subTest(base=base.__class__.__name__, p_type=type(p)):
                    # compute base
                    base_res = base.compute(p, u, y_best=1.0)
                    self.assertEqual(base_res.shape, (1,))
                    self.assertTrue(np.all(np.isfinite(base_res)))

                    # compute additive
                    add_res = add_acq.compute_additive(p, u, ue, y_best=1.0, beta_t=1.0)
                    self.assertEqual(add_res.shape, (1,))
                    self.assertTrue(np.all(np.isfinite(add_res)))

    def test_massive_candidate_pool_scalability(self):
        """Stress-test massive candidate pool (N=10,000) verifying < 1.0s latency and finite values."""
        N_massive = 10000
        preds = np.random.uniform(-50.0, 50.0, size=N_massive)
        unc_tot = np.random.exponential(scale=2.0, size=N_massive) + 1e-4
        u_ep = np.random.exponential(scale=1.0, size=N_massive) + 1e-4
        y_best = 0.0

        for name, acq_cls in [("ei", ExpectedImprovement), ("lcb", LowerConfidenceBound), ("pi", ProbabilityOfImprovement)]:
            base_acq = acq_cls()
            add_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)

            start_t = time.perf_counter()
            scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.5)
            elapsed = time.perf_counter() - start_t

            self.assertEqual(scores.shape, (N_massive,))
            self.assertTrue(np.all(np.isfinite(scores)), f"Non-finite scores in {name}")
            self.assertFalse(np.any(np.isnan(scores)), f"NaN scores in {name}")
            # Ensure sub-second execution (vectorized numpy should take < 50ms)
            self.assertLess(elapsed, 1.0, f"{name} took {elapsed:.4f}s > 1.0s limit")

    def test_empty_candidate_pool(self):
        """Empty candidate pool (N=0) gracefully returns empty 1D array."""
        preds = np.array([])
        unc_tot = np.array([])
        u_ep = np.array([])
        y_best = 0.0

        for acq_cls in [ExpectedImprovement, LowerConfidenceBound, ProbabilityOfImprovement]:
            add_acq = AdditiveEpistemicAcquisition(base_acq=acq_cls())
            scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
            self.assertEqual(scores.shape, (0,))


class TestAcquisitionNumericalExtremes(unittest.TestCase):
    """Stress testing numerical boundaries, shifts, zero arrays, flat surfaces, and extreme parameters."""

    def test_negative_lcb_shifts_and_translation_invariance(self):
        """LCB scores can be heavily negative. Verify translation invariance under arbitrary offsets."""
        base_lcb = LowerConfidenceBound(beta=2.0)
        add_lcb = AdditiveEpistemicAcquisition(base_acq=base_lcb)

        preds = np.array([500.0, 1000.0, 1500.0, 2000.0])  # large positive preds => negative LCB
        unc_tot = np.array([2.0, 5.0, 1.0, 10.0])
        u_ep = np.array([0.5, 1.2, 0.1, 2.5])
        y_best = 500.0

        raw_lcb = base_lcb.compute(preds, unc_tot, y_best)
        self.assertTrue(np.all(raw_lcb < 0), f"Expected all raw LCB < 0, got {raw_lcb}")

        scores_base = add_lcb.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
        self.assertTrue(np.all(scores_base >= 0), "Normalized scores must be non-negative")
        self.assertTrue(np.all(np.isfinite(scores_base)))

        # Extreme positive and negative translations
        for offset in [-1e9, -1e6, -100.0, 100.0, 1e6, 1e9]:
            with self.subTest(offset=offset):
                scores_shifted = add_lcb.compute_additive(
                    preds + offset, unc_tot, u_ep, y_best + offset, beta_t=1.0
                )
                np.testing.assert_allclose(
                    scores_shifted, scores_base, rtol=1e-6, atol=1e-6,
                    err_msg=f"Translation invariance violated for offset {offset}"
                )

    def test_all_zero_inputs(self):
        """
        Verify behavior when all inputs (preds, unc_tot, u_ep, y_best) are strictly zero.
        - For EI and LCB: raw acquisition is 0, so normalized scores are identically 0.
        - For PI with xi > 0: improvement requires diff > xi, so probability is 0.
        - For PI with xi == 0: z = (0 - 0) / sigma = 0, so norm.cdf(0) = 0.5.
          Because all candidates have score 0.5, max_relative normalizes all candidates to 1.0.
        """
        for N in [1, 5, 100]:
            preds = np.zeros(N)
            unc_tot = np.zeros(N)
            u_ep = np.zeros(N)
            y_best = 0.0

            # EI and LCB
            for acq_cls in [ExpectedImprovement, LowerConfidenceBound]:
                add_acq = AdditiveEpistemicAcquisition(base_acq=acq_cls())
                scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
                self.assertEqual(scores.shape, (N,))
                self.assertTrue(np.all(np.isfinite(scores)))
                np.testing.assert_array_equal(scores, np.zeros(N))

            # PI with xi=0.01 (guards against zero-improvement singularity)
            pi_guard = AdditiveEpistemicAcquisition(base_acq=ProbabilityOfImprovement(xi=0.01))
            scores_guard = pi_guard.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
            np.testing.assert_array_equal(scores_guard, np.zeros(N))

            # PI with default xi=0.0 (mathematical CDF limit Phi(0) = 0.5 normalized to 1.0)
            pi_default = AdditiveEpistemicAcquisition(base_acq=ProbabilityOfImprovement(xi=0.0))
            scores_default = pi_default.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
            np.testing.assert_allclose(scores_default, np.ones(N), rtol=1e-6)

    def test_pi_zero_uncertainty_singularity_behavior(self):
        """
        Empirical critique of ProbabilityOfImprovement:
        When unc <= 1e-9 and preds == y_best, z = 0 / 1e-9 = 0, so norm.cdf(0) = 0.5.
        A point with 0 uncertainty and zero improvement gets score 0.5,
        which can exceed points that actually have a positive chance of improvement.
        """
        pi = ProbabilityOfImprovement(xi=0.0)
        # Candidate 0: evaluated point with unc=0, pred=y_best (0 true improvement)
        # Candidate 1: exploratory point with unc=0.5, pred=y_best+0.01
        preds = np.array([1.0, 1.01])
        unc = np.array([0.0, 0.5])
        y_best = 1.0
        scores = pi.compute(preds, unc, y_best)
        # Empirical finding: score[0] == 0.5, score[1] == norm.cdf(-0.02) = 0.492
        self.assertAlmostEqual(scores[0], 0.5, places=5)
        self.assertLess(scores[1], 0.5)

    def test_zero_epistemic_uncertainty_pure_base_fallthrough(self):
        """When U_E = 0, additive acquisition must strictly match normalized base acquisition."""
        preds = np.array([1.0, 2.0, 0.5, 4.0])
        unc_tot = np.array([0.5, 1.0, 0.2, 0.8])
        u_ep = np.zeros(4)
        y_best = 1.0

        for acq_cls in [ExpectedImprovement, LowerConfidenceBound, ProbabilityOfImprovement]:
            base = acq_cls()
            add_acq = AdditiveEpistemicAcquisition(base_acq=base)

            raw_base = base.compute(preds, unc_tot, y_best)
            if isinstance(base, LowerConfidenceBound):
                raw_base = raw_base - np.min(raw_base)
            norm_base = normalize_max_relative(raw_base)

            scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=5.0)
            np.testing.assert_allclose(scores, norm_base, rtol=1e-8, atol=1e-8)

    def test_flat_base_acquisition_pure_epistemic_fallthrough(self):
        """When base acquisition is flat across all candidates, epistemic uncertainty dictates ranking."""
        preds = np.array([2.0, 2.0, 2.0, 2.0])  # completely flat predictions
        unc_tot = np.array([1.0, 1.0, 1.0, 1.0])  # completely flat uncertainty
        u_ep = np.array([0.1, 0.8, 0.3, 0.9])  # distinctive epistemic signals
        y_best = 2.0

        for acq_cls in [ExpectedImprovement, LowerConfidenceBound, ProbabilityOfImprovement]:
            base = acq_cls()
            add_acq = AdditiveEpistemicAcquisition(base_acq=base)
            scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)

            # Ranks must match u_ep ranks
            expected_ranking = np.argsort(u_ep)
            actual_ranking = np.argsort(scores)
            np.testing.assert_array_equal(
                actual_ranking, expected_ranking,
                err_msg=f"Flat base fallthrough failed for {acq_cls.__name__}"
            )

    def test_scale_invariance_ranking_preservation(self):
        """Multiplying base acquisition by 10^6 or 10^-6 preserves relative candidate ranking."""
        preds = np.array([1.0, 2.0, 0.5, 3.0])
        unc_tot = np.array([0.5, 0.8, 0.1, 0.9])
        y_best = 1.0

        base_ei = ExpectedImprovement(xi=0.0)
        raw_base_1 = base_ei.compute(preds, unc_tot, y_best)
        norm_base_1 = normalize_max_relative(raw_base_1)
        rank_1 = np.argsort(norm_base_1)

        for scale in [1e-6, 1e-3, 1e3, 1e6]:
            with self.subTest(scale=scale):
                raw_scaled = raw_base_1 * scale
                norm_scaled = normalize_max_relative(raw_scaled)
                rank_scaled = np.argsort(norm_scaled)
                np.testing.assert_array_equal(
                    rank_scaled, rank_1,
                    err_msg=f"Scaling by {scale} disrupted candidate ranking in normalize_max_relative"
                )

    def test_extreme_beta_t_values(self):
        """Test beta_t = 0.0 (pure exploitation) and beta_t = 1000.0 (pure exploration)."""
        preds = np.array([1.0, 2.0, 0.5])
        unc_tot = np.array([0.5, 1.0, 0.2])
        u_ep = np.array([0.1, 0.8, 0.05])
        y_best = 1.0

        base_ei = ExpectedImprovement()
        add_ei = AdditiveEpistemicAcquisition(base_acq=base_ei)

        # beta_t = 0.0 -> matches norm_base
        scores_beta0 = add_ei.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=0.0)
        norm_base = normalize_max_relative(base_ei.compute(preds, unc_tot, y_best))
        np.testing.assert_allclose(scores_beta0, norm_base, rtol=1e-8, atol=1e-8)

        # beta_t = 1000.0 -> epistemic dominates
        scores_beta1000 = add_ei.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1000.0)
        # Candidate 1 has highest u_ep (0.8), should be chosen overwhelmingly
        self.assertEqual(np.argmax(scores_beta1000), 1)

    def test_negative_beta_t(self):
        """Negative beta_t acts as epistemic penalty (repulsion). Scores remain finite."""
        # Candidates with equal base acquisition
        preds = np.array([1.0, 1.0])
        unc_tot = np.array([0.5, 0.5])
        u_ep = np.array([0.8, 0.1])
        y_best = 1.0

        add_ei = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())
        scores_neg = add_ei.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=-1.0)
        self.assertTrue(np.all(np.isfinite(scores_neg)))
        # With equal base, Candidate 0 has higher epistemic uncertainty, so penalized more by negative beta
        self.assertLess(scores_neg[0], scores_neg[1])

    def test_nan_inf_safety_in_normalizer(self):
        """Test normalize_max_relative under NaN, Inf, -Inf."""
        arr_corrupted = np.array([np.nan, 1.0, np.inf, -np.inf, 2.0])
        normed = normalize_max_relative(arr_corrupted)
        self.assertTrue(np.all(np.isfinite(normed)))
        self.assertFalse(np.any(np.isnan(normed)))
        self.assertEqual(normed[0], 0.0)  # nan mapped to 0

    def test_scheduler_boundaries(self):
        """Test WarmupCosineScheduler edge cases: t<0, t>T, T=1, ratio=0, ratio=1."""
        sched = WarmupCosineScheduler(total_trials=50, warmup_ratio=0.2, beta_max=2.0, beta_min=0.1)
        self.assertEqual(sched.get_beta(-5), 2.0)
        self.assertEqual(sched.get_beta(0), 2.0)
        self.assertEqual(sched.get_beta(10), 2.0)  # t_warmup = 10
        self.assertLess(sched.get_beta(25), 2.0)
        self.assertGreater(sched.get_beta(25), 0.1)
        self.assertEqual(sched.get_beta(50), 0.1)
        self.assertEqual(sched.get_beta(100), 0.1)

        # Edge case: T=1
        sched_1 = WarmupCosineScheduler(total_trials=1, warmup_ratio=0.5, beta_max=1.0, beta_min=0.0)
        self.assertEqual(sched_1.get_beta(0), 1.0)
        self.assertEqual(sched_1.get_beta(1), 0.0)

        # Edge case: warmup_ratio=0.0
        sched_no_warmup = WarmupCosineScheduler(total_trials=10, warmup_ratio=0.0, beta_max=1.0, beta_min=0.0)
        self.assertEqual(sched_no_warmup.get_beta(0), 1.0)
        self.assertLess(sched_no_warmup.get_beta(5), 1.0)


class TestParetoDominancePreservation(unittest.TestCase):
    """Stress testing Pareto dominance preservation across acquisition functions."""

    def test_strict_pareto_dominance_synthetic_pairs(self):
        """
        If candidate A strictly dominates candidate B in both base acquisition and epistemic uncertainty,
        then alpha(A) > alpha(B) MUST hold for all beta_t in [0.01, 100.0].
        """
        np.random.seed(123)
        N_pairs = 100

        for acq_name, acq_cls in [("ei", ExpectedImprovement), ("lcb", LowerConfidenceBound), ("pi", ProbabilityOfImprovement)]:
            base_acq = acq_cls()
            add_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)

            for _ in range(N_pairs):
                # Candidate A: lower pred, higher unc -> dominates B in both base and epistemic
                pred_A = np.random.uniform(0.0, 2.0)
                pred_B = pred_A + np.random.uniform(0.5, 3.0)  # pred_B > pred_A (worse for minimization)

                unc_B = np.random.uniform(0.1, 1.0)
                unc_A = unc_B + np.random.uniform(0.2, 1.5)  # unc_A > unc_B (higher total uncertainty)

                ue_B = np.random.uniform(0.05, 0.5)
                ue_A = ue_B + np.random.uniform(0.1, 0.5)  # ue_A > ue_B (higher epistemic uncertainty)

                preds = np.array([pred_A, pred_B])
                unc_tot = np.array([unc_A, unc_B])
                u_ep = np.array([ue_A, ue_B])
                y_best = 1.0

                # Verify base dominance
                raw_base = base_acq.compute(preds, unc_tot, y_best)
                self.assertGreater(raw_base[0], raw_base[1], f"Base {acq_name} dominance violated at generation")

                # Test across dense grid of beta_t
                for beta_t in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
                    scores = add_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=beta_t)
                    self.assertGreater(
                        scores[0], scores[1],
                        f"Pareto dominance violated for {acq_name} at beta_t={beta_t}: alpha(A)={scores[0]}, alpha(B)={scores[1]}"
                    )

    def test_weak_pareto_dominance(self):
        """
        Weak Pareto dominance:
        Case 1: base(A) > base(B), u_ep(A) == u_ep(B) => alpha(A) > alpha(B) for any beta_t >= 0.
        Case 2: base(A) == base(B), u_ep(A) > u_ep(B) => alpha(A) > alpha(B) for any beta_t > 0.
        """
        add_ei = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())

        # Case 1: base(A) > base(B), equal epistemic
        preds_1 = np.array([0.5, 2.0])
        unc_tot_1 = np.array([1.0, 1.0])
        u_ep_1 = np.array([0.5, 0.5])
        for beta in [0.0, 0.1, 1.0, 10.0]:
            scores = add_ei.compute_additive(preds_1, unc_tot_1, u_ep_1, y_best=1.0, beta_t=beta)
            self.assertGreater(scores[0], scores[1])

        # Case 2: equal base, u_ep(A) > u_ep(B)
        preds_2 = np.array([1.0, 1.0])
        unc_tot_2 = np.array([0.5, 0.5])
        u_ep_2 = np.array([0.8, 0.2])
        for beta in [0.01, 0.1, 1.0, 10.0]:
            scores = add_ei.compute_additive(preds_2, unc_tot_2, u_ep_2, y_best=1.0, beta_t=beta)
            self.assertGreater(scores[0], scores[1])

    def test_tradeoff_inversion_monotonicity(self):
        """
        Non-dominated trade-off:
        Candidate A has high base, low epistemic.
        Candidate B has low base, high epistemic.
        For small beta_t, alpha(A) > alpha(B).
        For large beta_t, alpha(B) > alpha(A).
        Verify that Delta = alpha(B) - alpha(A) is strictly monotonic increasing in beta_t.
        """
        add_ei = AdditiveEpistemicAcquisition(base_acq=ExpectedImprovement())
        preds = np.array([0.0, 1.0])
        unc_tot = np.array([1.0, 0.5])
        u_ep = np.array([0.1, 1.0])
        y_best = 0.5

        betas = np.linspace(0.01, 10.0, 50)
        deltas = []
        for beta in betas:
            scores = add_ei.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=beta)
            deltas.append(scores[1] - scores[0])

        deltas = np.array(deltas)
        diffs = np.diff(deltas)
        self.assertTrue(np.all(diffs > 0), "Delta = alpha(B) - alpha(A) must be strictly monotonic in beta_t")
        self.assertLess(deltas[0], 0.0)
        self.assertGreater(deltas[-1], 0.0)


class TestCustomUncertaintyModelStress(unittest.TestCase):
    """Stress testing CustomUncertaintyRandomForest shapes, inputs, and contracts."""

    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace({"x": Float("x", bounds=(-2.0, 2.0), default=0.0)})
        self.X_train = np.linspace(-2.0, 2.0, 15).reshape(-1, 1)
        self.y_train = np.sin(self.X_train).ravel()

    def test_predict_return_shapes_and_types(self):
        """Verify _predict(X) always returns (mean: (N, 1), var: (N, 1)) for any N."""
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=self.cs,
            n_trees=5,
            seed=42,
        )
        rf.train(self.X_train, self.y_train)

        test_sizes = [1, 5, 20, 1000]
        for n in test_sizes:
            with self.subTest(n=n):
                X_test = np.linspace(-2.0, 2.0, n).reshape(-1, 1)
                mean, var = rf._predict(X_test)

                self.assertEqual(mean.shape, (n, 1), f"Mean shape mismatch for N={n}")
                self.assertEqual(var.shape, (n, 1), f"Var shape mismatch for N={n}")
                self.assertTrue(np.all(np.isfinite(mean)), "Mean contains non-finite values")
                self.assertTrue(np.all(np.isfinite(var)), "Var contains non-finite values")
                self.assertTrue(np.all(var >= 1e-10), "Var violated minimum clip floor 1e-10")

    def test_predict_1d_input_rejection(self):
        """1D array (D,) input must raise ValueError per SMAC3 2D API contract."""
        rf = CustomUncertaintyRandomForest(uncertainty_func="distance_evidential", configspace=self.cs, n_trees=5)
        rf.train(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            rf._predict(np.array([0.5]))

    def test_train_target_formats(self):
        """Verify train accepts 1D y, 2D col y, and Y keyword argument."""
        # 1D y
        rf1 = CustomUncertaintyRandomForest(uncertainty_func="distance_evidential", configspace=self.cs, n_trees=5)
        rf1.train(self.X_train, y=self.y_train)
        self.assertIsNotNone(rf1.uq_extractor)

        # 2D col y
        rf2 = CustomUncertaintyRandomForest(uncertainty_func="distance_evidential", configspace=self.cs, n_trees=5)
        rf2.train(self.X_train, y=self.y_train.reshape(-1, 1))
        self.assertIsNotNone(rf2.uq_extractor)

        # Keyword Y
        rf3 = CustomUncertaintyRandomForest(uncertainty_func="distance_evidential", configspace=self.cs, n_trees=5)
        rf3.train(self.X_train, Y=self.y_train)
        self.assertIsNotNone(rf3.uq_extractor)

        # Neither y nor Y -> ValueError
        rf4 = CustomUncertaintyRandomForest(uncertainty_func="distance_evidential", configspace=self.cs, n_trees=5)
        with self.assertRaises(ValueError):
            rf4.train(self.X_train)

    def test_custom_callable_uncertainty_function(self):
        """Verify custom callable returning 1D, 2D col, or 2D row arrays."""
        # Callable returning 1D array
        func_1d = lambda rf, X, y: np.ones(len(X)) * 0.5
        rf_1d = CustomUncertaintyRandomForest(uncertainty_func=func_1d, configspace=self.cs, n_trees=5)
        rf_1d.train(self.X_train, self.y_train)
        m, v = rf_1d._predict(self.X_train)
        self.assertEqual(v.shape, (len(self.X_train), 1))
        np.testing.assert_allclose(v, 0.25)

        # Callable returning 2D col array
        func_2d_col = lambda rf, X, y: np.ones((len(X), 1)) * 0.4
        rf_2d = CustomUncertaintyRandomForest(uncertainty_func=func_2d_col, configspace=self.cs, n_trees=5)
        rf_2d.train(self.X_train, self.y_train)
        m, v = rf_2d._predict(self.X_train)
        self.assertEqual(v.shape, (len(self.X_train), 1))
        np.testing.assert_allclose(v, 0.16)

        # Callable returning all zeros -> clipped to 1e-10
        func_zero = lambda rf, X, y: np.zeros(len(X))
        rf_zero = CustomUncertaintyRandomForest(uncertainty_func=func_zero, configspace=self.cs, n_trees=5)
        rf_zero.train(self.X_train, self.y_train)
        m, v = rf_zero._predict(self.X_train)
        self.assertEqual(v.shape, (len(self.X_train), 1))
        np.testing.assert_allclose(v, 1e-10)

    def test_invalid_uncertainty_func_type(self):
        """Verify passing an invalid type raises ValueError."""
        rf = CustomUncertaintyRandomForest(uncertainty_func=12345, configspace=self.cs, n_trees=5)
        rf.train(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            rf._predict(self.X_train)

    def test_hierarchical_inactive_imputation(self):
        """Verify CustomUncertaintyRandomForest imputes NaNs in hierarchical spaces."""
        cs_hier = ConfigurationSpace(seed=42)
        model_type = Categorical("model_type", ["rf", "svm"], default="rf")
        rf_trees = Float("rf_trees", (10.0, 100.0), default=50.0)
        svm_c = Float("svm_c", (0.1, 10.0), default=1.0)
        cs_hier.add([model_type, rf_trees, svm_c])
        cs_hier.add(EqualsCondition(rf_trees, model_type, "rf"))
        cs_hier.add(EqualsCondition(svm_c, model_type, "svm"))

        # Synthetic data with NaNs
        X_hier = np.array([
            [0.0, 50.0, np.nan],
            [0.0, 80.0, np.nan],
            [1.0, np.nan, 2.5],
            [1.0, np.nan, 0.5],
        ])
        y_hier = np.array([1.2, 0.8, 2.1, 1.9])

        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs_hier,
            n_trees=5,
            seed=42,
        )
        rf.train(X_hier, y_hier)
        mean, var = rf._predict(X_hier)
        self.assertEqual(mean.shape, (4, 1))
        self.assertEqual(var.shape, (4, 1))
        self.assertTrue(np.all(np.isfinite(mean)))
        self.assertTrue(np.all(np.isfinite(var)))


if __name__ == "__main__":
    unittest.main()

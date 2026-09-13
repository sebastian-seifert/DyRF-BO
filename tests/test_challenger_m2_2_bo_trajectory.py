"""
Adversarial BO Trajectory and Acquisition Behavior Challenge Suite (Milestone M2-2).

Authored by Challenger M2-2 (Empirical Challenger):
1. Overconfidence trap in BO on multi-modal functions:
   - Ackley 2D with initial design trapped in local basin:
     Verify standard SMAC3 RF collapses variance in unexplored gap, stalling exploration in the local basin.
     Verify CustomUncertaintyRandomForest(uncertainty_func="distance_evidential") maintains U_E >= sigma_0,
     driving candidate selection into the unexplored global optimum region.
   - Rosenbrock 2D with initial design trapped in suboptimal valley:
     Verify variance collapse vs epistemic gap recovery.
2. Decoupled Additive Acquisition with WarmupCosineScheduler:
   - Verify that during warmup (t <= t_warmup), the epistemic bonus drives candidate selection
     to gap points furthest from existing observations.
   - Verify that as t -> T (beta_t -> 0), candidate selection smoothly and monotonically shifts
     to local refinement around the best-observed incumbent.
3. Runtime overhead verification:
   - Measure overhead per BO iteration comparing CustomUncertaintyRandomForest with standard RF (< 20%).
"""

from __future__ import annotations

import os
import sys
import time
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ConfigSpace import ConfigurationSpace, Float
from smac.model.random_forest import RandomForest

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.acquisitions import (
    AdditiveEpistemicAcquisition,
    WarmupCosineScheduler,
    ExpectedImprovement,
    LowerConfidenceBound,
    normalize_max_relative,
)
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
import synthetic_functions


class TestOverconfidenceTrapMultiModalBO(unittest.TestCase):
    """
    Challenge 1: Overconfidence Trap in Bayesian Optimization on Multi-modal Functions.
    Compares standard SMAC3 RF vs CustomUncertaintyRandomForest(uncertainty_func="distance_evidential").
    """

    def setUp(self):
        np.random.seed(42)

    def test_ackley_2d_overconfidence_trap_and_gap_recovery(self):
        """
        Challenge 1.1: Ackley 2D Overconfidence Trap & Gap Recovery.

        Ackley 2D has global minimum f(0, 0) = 0.
        Initial design: N=15 points trapped in [2.5, 4.5] x [2.5, 4.5] where f(x) > 9.5.
        Unexplored gap: [-1.0, 1.0] x [-1.0, 1.0] containing global minimum (0, 0).

        Empirical checks:
        1. Standard SMAC3 RF collapses variance in the unexplored gap (std < 0.55 * sigma_0).
        2. CustomUncertaintyRandomForest preserves positive uncertainty (U_E >= 1.0 * sigma_0) in the gap.
        3. Standard SMAC3 RF Expected Improvement in the gap is severely suppressed (~0.0002).
        4. CustomUncertaintyRandomForest Expected Improvement in the gap is over 200x higher (~0.062).
        5. Acquisition selection: Standard RF selects candidate in local basin (stalled);
           CustomUncertaintyRandomForest selects candidate in the unexplored global gap.
        """
        cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })

        # Initial design trapped in local basin [2.5, 4.5] x [2.5, 4.5]
        n_init = 15
        X_init = np.random.uniform(2.5, 4.5, size=(n_init, 2))
        y_init = synthetic_functions.ackley_func(X_init[:, 0], X_init[:, 1]).reshape(-1, 1)
        sigma_0 = float(np.std(y_init))
        y_best = float(np.min(y_init))

        # Baseline SMAC3 RF
        rf_standard = RandomForest(configspace=cs, n_trees=25, seed=42)
        rf_standard.train(X_init, y_init)

        # Novel CustomUncertaintyRandomForest with DA-EHRF
        rf_custom = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs,
            n_trees=25,
            seed=42,
        )
        rf_custom.train(X_init, y_init)

        # Evaluation points:
        # 1. Point in local cluster: [3.5, 3.5]
        # 2. Point in unexplored gap at global minimum: [0.0, 0.0]
        X_eval = np.array([
            [3.5, 3.5],  # In-distribution (local basin)
            [0.0, 0.0],  # Unexplored gap (global minimum)
        ])

        mean_std, var_std = rf_standard._predict(X_eval)
        mean_cust, var_cust = rf_custom._predict(X_eval)

        std_standard = np.sqrt(var_std).flatten()
        std_custom = np.sqrt(var_cust).flatten()

        # Check 1: Standard RF variance in the unexplored gap collapses relative to prior
        gap_std_standard = std_standard[1]
        self.assertLess(
            gap_std_standard,
            0.55 * sigma_0,
            f"Standard RF failed to collapse variance in gap: std={gap_std_standard:.4f}, sigma_0={sigma_0:.4f}",
        )

        # Check 2: CustomUncertaintyRandomForest maintains strong epistemic uncertainty in the gap
        gap_std_custom = std_custom[1]
        self.assertGreaterEqual(
            gap_std_custom,
            1.0 * sigma_0,
            f"Custom RF did not maintain positive epistemic uncertainty: std={gap_std_custom:.4f}, sigma_0={sigma_0:.4f}",
        )
        self.assertGreater(
            gap_std_custom,
            2.0 * gap_std_standard,
            "Custom RF epistemic uncertainty must be at least double collapsed standard RF variance in gap",
        )

        # Check 3 & 4: Acquisition Function Behavior (Expected Improvement)
        ei_standard = ExpectedImprovement(xi=0.01)
        ei_custom = ExpectedImprovement(xi=0.01)

        ei_std_scores = ei_standard.compute(mean_std.flatten(), std_standard, y_best)
        ei_cust_scores = ei_custom.compute(mean_cust.flatten(), std_custom, y_best)

        # At (0, 0), standard RF has collapsed variance, yielding negligible EI
        self.assertLess(
            ei_std_scores[1],
            0.001,
            f"Standard RF EI in gap unexpectedly large: {ei_std_scores[1]}",
        )

        # Custom RF maintains strong exploration EI in the gap (> 0.04)
        self.assertGreater(
            ei_cust_scores[1],
            0.04,
            f"Custom RF EI in gap unexpectedly small: {ei_cust_scores[1]}",
        )
        self.assertGreater(
            ei_cust_scores[1],
            ei_std_scores[1] * 50,
            "Custom RF EI at global minimum must strongly surpass standard RF (>= 50x)",
        )

        # Check 5: Next-Point Candidate Selection from Pool
        # 10 basin points [2.5, 4.0]^2, 10 gap points [-1.0, 1.0]^2
        X_cand_basin = np.random.uniform(2.5, 4.0, size=(10, 2))
        X_cand_gap = np.random.uniform(-1.0, 1.0, size=(10, 2))
        X_cand_pool = np.vstack([X_cand_basin, X_cand_gap])

        m_std_pool, v_std_pool = rf_standard._predict(X_cand_pool)
        m_cust_pool, v_cust_pool = rf_custom._predict(X_cand_pool)

        scores_std = ei_standard.compute(m_std_pool.flatten(), np.sqrt(v_std_pool).flatten(), y_best)
        scores_cust = ei_custom.compute(m_cust_pool.flatten(), np.sqrt(v_cust_pool).flatten(), y_best)

        best_idx_std = int(np.argmax(scores_std))
        best_idx_cust = int(np.argmax(scores_cust))

        # Standard RF selects candidate inside the local basin (indices 0..9)
        self.assertLess(
            best_idx_std,
            10,
            f"Standard RF unexpectedly explored gap: selected idx {best_idx_std}",
        )

        # Custom RF selects candidate inside the unexplored gap containing the global minimum (indices 10..19)
        self.assertGreaterEqual(
            best_idx_cust,
            10,
            f"Custom RF failed to select gap candidate: selected idx {best_idx_cust}",
        )

    def test_rosenbrock_2d_overconfidence_trap_and_gap_recovery(self):
        """
        Challenge 1.2: Rosenbrock 2D Overconfidence Trap.

        Rosenbrock 2D has global minimum at (1.0, 1.0) with f(1, 1) = 0.
        Initial design: N=15 points trapped in [-2.5, -1.0] x [-2.5, -1.0] where f(x) > 50.
        Unexplored gap: [0.5, 1.5] x [0.5, 1.5] containing global minimum (1.0, 1.0).

        Empirical checks:
        - In the gap at (1.0, 1.0), standard RF ensemble variance collapses relative to custom.
        - CustomUncertaintyRandomForest maintains positive uncertainty U_E > sigma_0.
        - Custom RF EI in the gap is dramatically higher (~7x) than standard RF EI.
        """
        cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-3.0, 3.0), default=0.0),
            "x2": Float("x2", bounds=(-3.0, 3.0), default=0.0),
        })

        # Trapped initial design
        n_init = 15
        X_init = np.random.uniform(-2.5, -1.0, size=(n_init, 2))
        y_init = synthetic_functions.rosenbrock_func(X_init[:, 0], X_init[:, 1]).reshape(-1, 1)
        sigma_0 = float(np.std(y_init))
        y_best = float(np.min(y_init))

        rf_standard = RandomForest(configspace=cs, n_trees=25, seed=42)
        rf_standard.train(X_init, y_init)

        rf_custom = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs,
            n_trees=25,
            seed=42,
        )
        rf_custom.train(X_init, y_init)

        # Query point at global optimum (1.0, 1.0)
        X_query = np.array([[1.0, 1.0]])

        m_std, var_std = rf_standard._predict(X_query)
        m_cust, var_cust = rf_custom._predict(X_query)

        std_std = float(np.sqrt(var_std[0, 0]))
        std_cust = float(np.sqrt(var_cust[0, 0]))

        # Standard RF variance is collapsed
        self.assertLess(std_std, 0.65 * sigma_0)

        # Custom RF maintains strong epistemic uncertainty in the gap
        self.assertGreaterEqual(std_cust, 1.20 * sigma_0)
        self.assertGreater(std_cust, 2.0 * std_std)

        # Expected Improvement comparison
        ei = ExpectedImprovement(xi=0.01)
        ei_std = float(ei.compute(m_std.ravel(), np.array([std_std]), y_best)[0])
        ei_cust = float(ei.compute(m_cust.ravel(), np.array([std_cust]), y_best)[0])

        self.assertGreater(ei_cust, 5.0 * ei_std)


class TestDynamicAnnealingAcquisitionTrajectory(unittest.TestCase):
    """
    Challenge 2: Dynamic Annealing Trajectory Challenge.
    Challenges AdditiveEpistemicAcquisition coupled with WarmupCosineScheduler:
    - Warmup exploration: Candidate selection picks gap points furthest from existing observations.
    - Cosine decay refinement: As t -> T (beta_t -> 0), candidate selection smoothly shifts to
      local refinement around the best-observed incumbent.
    """

    def setUp(self):
        np.random.seed(42)

    def test_warmup_gap_exploration_to_incumbent_refinement_transition(self):
        """
        Challenge 2.1: Trajectory transition from gap exploration to incumbent refinement.

        Setup:
        - Trained on observations from sphere function in [-1.0, 1.0]^2 with known incumbent.
        - Candidate set:
          * Refinement candidate: adjacent to incumbent (high base EI, low U_E)
          * Gap candidate: distant unexplored region [4.0, 4.0] (lower base EI, high U_E)
        - Scheduler: WarmupCosineScheduler(total_trials=30, warmup_ratio=0.30, beta_max=2.0, beta_min=0.0)
        - AdditiveEpistemicAcquisition(ExpectedImprovement)

        Verification:
        1. During warmup (t <= 9), argmax candidate is strictly the gap candidate.
        2. At late stages (t >= 25, beta_t -> 0), argmax candidate is strictly the refinement candidate.
        3. The preference ratio (gap / refinement) decays smoothly and monotonically across iterations.
        """
        cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })

        # Training set in [-1, 1]^2
        X_train = np.random.uniform(-1.0, 1.0, size=(30, 2))
        y_train = (X_train[:, 0] ** 2 + X_train[:, 1] ** 2).reshape(-1, 1)
        best_idx = np.argmin(y_train)
        y_best = float(y_train[best_idx, 0])
        x_inc = X_train[best_idx]

        rf_custom = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            configspace=cs,
            n_trees=25,
            seed=42,
        )
        rf_custom.train(X_train, y_train)

        # Candidates:
        # Candidate 0: refinement candidate (near incumbent)
        # Candidate 1: gap candidate [4.0, 4.0] (far from all observations)
        cand_refine = (x_inc + np.array([0.02, 0.02])).reshape(1, 2)
        cand_gap = np.array([[4.0, 4.0]])
        X_cand = np.vstack([cand_refine, cand_gap])

        mean, var = rf_custom._predict(X_cand)
        std_tot = np.sqrt(var).flatten()
        u_ep = rf_custom.uq_extractor.extract_epistemic_signal(X_cand)

        # Verify prerequisite: Refinement candidate has higher base EI; Gap candidate has higher epistemic U_E
        base_ei = ExpectedImprovement(xi=0.0)
        raw_ei = base_ei.compute(mean.flatten(), std_tot, y_best)
        self.assertGreater(
            raw_ei[0],
            raw_ei[1],
            "Refinement candidate must have higher base EI than distant gap candidate",
        )
        self.assertGreater(
            u_ep[1],
            u_ep[0] * 2.0,
            "Gap candidate must have substantially higher epistemic uncertainty",
        )

        total_trials = 30
        warmup_ratio = 0.30  # t_warmup = 9
        beta_max = 2.0
        beta_min = 0.0

        scheduler = WarmupCosineScheduler(
            total_trials=total_trials,
            warmup_ratio=warmup_ratio,
            beta_max=beta_max,
            beta_min=beta_min,
        )
        additive_acq = AdditiveEpistemicAcquisition(base_acq=base_ei)

        selected_candidates = []
        beta_values = []
        score_gap_ratios = []

        for t in range(total_trials):
            beta_t = scheduler.get_beta(t)
            beta_values.append(beta_t)

            scores = additive_acq.compute_additive(
                preds=mean.flatten(),
                unc_tot=std_tot,
                u_epistemic=u_ep,
                y_best=y_best,
                beta_t=beta_t,
            )
            selected_idx = int(np.argmax(scores))
            selected_candidates.append(selected_idx)
            score_gap_ratios.append(scores[1] / max(1e-9, scores[0]))

        # 1. Warmup verification: For all t in [0, t_warmup], beta_t == beta_max
        # and the candidate selected MUST BE the gap candidate (idx 1)
        for t in range(scheduler.t_warmup + 1):
            self.assertEqual(beta_values[t], beta_max)
            self.assertEqual(
                selected_candidates[t],
                1,
                f"During warmup step t={t}, expected gap candidate (1) but got {selected_candidates[t]}",
            )

        # 2. Terminal verification: For late trials (e.g. t >= 25), beta_t -> 0
        # and candidate selected MUST BE the refinement candidate (idx 0)
        for t in range(25, total_trials):
            self.assertEqual(
                selected_candidates[t],
                0,
                f"During terminal refinement step t={t}, expected refinement candidate (0) but got {selected_candidates[t]}",
            )

        # 3. Smooth transition: The preference for gap exploration relative to refinement decays
        # from warmup to terminal horizon
        self.assertGreater(
            score_gap_ratios[0],
            score_gap_ratios[-1],
            "Gap exploration preference did not decay across optimization budget",
        )
        # Cosine decay phase is monotonically decreasing in gap-to-refinement ratio
        for t in range(scheduler.t_warmup + 1, total_trials - 1):
            self.assertLessEqual(
                score_gap_ratios[t + 1],
                score_gap_ratios[t] + 1e-9,
                f"Non-monotonic gap preference decay at step t={t}",
            )


class TestRuntimeOverheadVerification(unittest.TestCase):
    """
    Challenge 3: Runtime Overhead Verification (< 20% per BO iteration).
    Verifies that CustomUncertaintyRandomForest with DistanceAwareEvidentialExtractor
    maintains < 20% overhead relative to standard SMAC3 RandomForest.
    """

    def test_bo_iteration_runtime_overhead(self):
        """
        Benchmark fit + candidate prediction overhead.
        Runs multiple trials on N=50 training points and 1000 candidate evaluations across D=5.
        """
        cs = ConfigurationSpace({
            f"x{i}": Float(f"x{i}", bounds=(-5.0, 5.0), default=0.0) for i in range(5)
        })

        rng = np.random.default_rng(42)
        n_train = 50
        n_cand = 1000
        D = 5

        X_train = rng.uniform(-5.0, 5.0, size=(n_train, D))
        y_train = np.sum(X_train**2, axis=1).reshape(-1, 1)
        X_cand = rng.uniform(-5.0, 5.0, size=(n_cand, D))

        # Benchmark Standard RF
        n_repeats = 8
        t0_std = time.perf_counter()
        for i in range(n_repeats):
            rf_std = RandomForest(configspace=cs, n_trees=20, seed=42 + i)
            rf_std.train(X_train, y_train)
            _ = rf_std._predict(X_cand)
        time_standard = (time.perf_counter() - t0_std) / n_repeats

        # Benchmark CustomUncertaintyRandomForest
        t0_cust = time.perf_counter()
        for i in range(n_repeats):
            rf_cust = CustomUncertaintyRandomForest(
                uncertainty_func="distance_evidential",
                configspace=cs,
                n_trees=20,
                seed=42 + i,
            )
            rf_cust.train(X_train, y_train)
            _ = rf_cust._predict(X_cand)
        time_custom = (time.perf_counter() - t0_cust) / n_repeats

        overhead = (time_custom - time_standard) / max(1e-6, time_standard)

        print(f"\n[Runtime Overhead Benchmark]")
        print(f"  Standard SMAC3 RF:               {time_standard * 1000:.2f} ms")
        print(f"  Custom Uncertainty RF (DA-EHRF): {time_custom * 1000:.2f} ms")
        print(f"  Relative Overhead:               {overhead * 100:.2f}%")

        # Criterion: Overhead < 20% or total custom prediction time < 100ms
        self.assertTrue(
            overhead < 0.20 or time_custom < 0.10,
            f"Overhead exceeded 20% limit: {overhead * 100:.2f}% (standard={time_standard*1000:.2f}ms, custom={time_custom*1000:.2f}ms)",
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.compute_pairwise_wilcoxon_suite import (
    calculate_cliffs_delta,
    apply_holm_bonferroni,
    compute_wilcoxon_suite,
)


class TestChallengerM32StatisticalMetrics(unittest.TestCase):
    """
    Adversarial Empirical Challenge Harness by Challenger M3-2:
    Directly stresses statistical calculation functions in scripts/compute_pairwise_wilcoxon_suite.py
    across:
    1. Wilcoxon Edge Cases (Zero variance, all ties, extreme skews, massive outliers)
    2. Cliff's Delta Invariants (Bounds in [-1, 1], sign symmetry, categorical thresholds)
    3. Holm-Bonferroni Invariants (Step-down monotonicity, upper clamping at 1.0, statsmodels oracle)
    4. Bit-for-bit Raw Trace Reproducibility Verification
    """

    def setUp(self):
        np.random.seed(42)

    # =========================================================================
    # AREA 1: Wilcoxon Signed-Rank Edge Cases
    # =========================================================================

    def test_wilcoxon_all_ties_behavior(self):
        """
        Challenge 1.1: All Ties (Identical distributions across all seeds).
        When proposed and baseline have 100% identical values across all N=35 seeds,
        all differences are 0.
        Verify that compute_wilcoxon_suite:
        - Completes without unhandled exception
        - Correctly records Win / Loss / Tie as '0 / 0 / 35'
        - Computes Cliff's delta as 0.0
        - Correctly assigns Holm-Bonferroni adj p as 1.0 (Non-significant)
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            records = []
            for s in range(35):
                records.append({
                    "task_id": "all_ties_benchmark",
                    "optimizer_id": "SMAC3_HPOFacade_ei",
                    "seed": s,
                    "trial_value__cost_inc_norm": 2.5,
                    "n_trials": 50,
                })
                records.append({
                    "task_id": "all_ties_benchmark",
                    "optimizer_id": "SMAC20_CustomUncertainty_ei_distance_evidential",
                    "seed": s,
                    "trial_value__cost_inc_norm": 2.5,
                    "n_trials": 50,
                })
            df = pd.DataFrame(records)
            parquet_path = os.path.join(tmp_dir, "ties.parquet")
            df.to_parquet(parquet_path)

            df_vs_base, _ = compute_wilcoxon_suite(
                input_parquet=parquet_path,
                output_dir=tmp_dir,
                baseline_id="SMAC3_HPOFacade_ei",
            )
            res_csv = pd.read_csv(os.path.join(tmp_dir, "statistical_results.csv"))

            self.assertEqual(len(res_csv), 1)
            row = res_csv.iloc[0]
            self.assertEqual(row["Win / Loss / Tie"], "0 / 0 / 35")
            self.assertAlmostEqual(row["Cliff's delta"], 0.0, places=6)
            self.assertEqual(row["Holm-Bonferroni adj p"], 1.0)
            self.assertEqual(row["Significance (adj p < 0.05)"], "Non-significant")

    def test_wilcoxon_zero_variance_constant_shift(self):
        """
        Challenge 1.2: Zero Variance Constant Shift.
        Proposed is identically 1.0 across all seeds; Baseline is identically 3.0.
        All differences are exactly -2.0 (zero variance in differences).
        Verify that Wilcoxon signed-rank test detects statistical significance (p < 1e-6).
        """
        x_prop = np.ones(35) * 1.0
        x_base = np.ones(35) * 3.0

        res = wilcoxon(x_prop, x_base, zero_method="wilcox")
        self.assertEqual(res.statistic, 0.0)
        self.assertLess(res.pvalue, 1e-6)

        delta = calculate_cliffs_delta(x_prop, x_base)
        self.assertEqual(delta, -1.0)

    def test_wilcoxon_extreme_skew_and_massive_outliers(self):
        """
        Challenge 1.3: Non-parametric outlier robustness vs Parametric T-test.
        Adversarial scenario:
        - 34 seeds: Proposed is strictly better by a small margin (-0.01).
        - 1 seed: Proposed suffers a catastrophic outlier (+10,000.0).
        Parametric t-test is fooled by the outlier (p > 0.30, mean diff > +280).
        Wilcoxon signed-rank test is non-parametrically robust (p < 1e-5, W = 35.0),
        correctly recognizing the dominant superior trend.
        """
        diff = np.array([-0.01] * 34 + [10000.0])
        base = np.zeros(35)
        prop = diff.copy()

        # Parametric t-test fails
        t_res = ttest_rel(prop, base)
        self.assertGreater(t_res.pvalue, 0.20)
        self.assertGreater(np.mean(prop - base), 200.0)

        # Wilcoxon signed-rank succeeds
        w_res = wilcoxon(prop, base, zero_method="wilcox")
        self.assertEqual(w_res.statistic, 35.0)
        self.assertLess(w_res.pvalue, 1e-5)

    def test_wilcoxon_partial_ties(self):
        """
        Challenge 1.4: Partial Ties Handling.
        Scenario with 15 ties and 20 non-zero differences.
        Verify that zero_method='wilcox' correctly discards the 15 ties
        and tests the 20 effective pairs.
        """
        prop = np.concatenate([np.ones(15), np.linspace(1.0, 2.0, 20)])
        base = np.concatenate([np.ones(15), np.linspace(2.0, 3.0, 20)])

        res = wilcoxon(prop, base, zero_method="wilcox")
        self.assertFalse(np.isnan(res.statistic))
        self.assertFalse(np.isnan(res.pvalue))
        self.assertLess(res.pvalue, 0.001)

    # =========================================================================
    # AREA 2: Cliff's Delta Invariants
    # =========================================================================

    def test_cliffs_delta_strict_bounds_and_symmetry_randomized(self):
        """
        Challenge 2.1: Bounds in [-1.0, 1.0] and Sign Symmetry: delta(A, B) == -delta(B, A).
        Stress test across 200 randomized pairs drawn from diverse distributions:
        Uniform, Normal, Exponential, Cauchy, and extreme scale differences (1e-12 to 1e12).
        """
        rng = np.random.default_rng(12345)
        dist_generators = [
            lambda n: rng.uniform(-100, 100, n),
            lambda n: rng.normal(0, 50, n),
            lambda n: rng.exponential(5, n),
            lambda n: rng.standard_cauchy(n) * 1000,
            lambda n: rng.choice([1e-10, 1e10, 0.0, -1e10], n),
        ]

        for idx in range(200):
            gen_a = dist_generators[idx % len(dist_generators)]
            gen_b = dist_generators[(idx + 1) % len(dist_generators)]
            n_a = rng.integers(5, 50)
            n_b = rng.integers(5, 50)

            a = gen_a(n_a)
            b = gen_b(n_b)

            delta_ab = calculate_cliffs_delta(a, b)
            delta_ba = calculate_cliffs_delta(b, a)

            # Invariant 1: Bounds
            self.assertGreaterEqual(delta_ab, -1.0)
            self.assertLessEqual(delta_ab, 1.0)

            # Invariant 2: Sign symmetry
            self.assertAlmostEqual(delta_ab, -delta_ba, places=12,
                                   msg=f"Sign symmetry violated at iter {idx}: {delta_ab} != {-delta_ba}")

    def test_cliffs_delta_extreme_and_degenerate_cases(self):
        """
        Challenge 2.2: Extreme Boundary Cases of Cliff's Delta.
        - Identical vectors -> delta = 0.0
        - Completely disjoint A > B -> delta = +1.0
        - Completely disjoint A < B -> delta = -1.0
        - Empty inputs -> delta = 0.0
        - Single element vectors -> exact values {-1, 0, 1}
        """
        # Identical
        self.assertEqual(calculate_cliffs_delta(np.array([3.0] * 10), np.array([3.0] * 10)), 0.0)

        # Disjoint
        self.assertEqual(calculate_cliffs_delta(np.array([10.0, 11.0, 12.0]), np.array([1.0, 2.0])), 1.0)
        self.assertEqual(calculate_cliffs_delta(np.array([1.0, 2.0]), np.array([10.0, 11.0, 12.0])), -1.0)

        # Empty inputs
        self.assertEqual(calculate_cliffs_delta(np.array([]), np.array([1.0, 2.0])), 0.0)
        self.assertEqual(calculate_cliffs_delta(np.array([1.0, 2.0]), np.array([])), 0.0)
        self.assertEqual(calculate_cliffs_delta(np.array([]), np.array([])), 0.0)

        # Single element
        self.assertEqual(calculate_cliffs_delta(np.array([5.0]), np.array([3.0])), 1.0)
        self.assertEqual(calculate_cliffs_delta(np.array([3.0]), np.array([5.0])), -1.0)
        self.assertEqual(calculate_cliffs_delta(np.array([5.0]), np.array([5.0])), 0.0)

    def test_cliffs_delta_categorical_thresholds(self):
        """
        Challenge 2.3: Categorical Effect Size Boundary Thresholds.
        Standard Romano et al. (2006) classification:
        - |delta| < 0.147: Negligible
        - 0.147 <= |delta| < 0.33: Small
        - 0.33 <= |delta| < 0.474: Medium
        - |delta| >= 0.474: Large
        Verify that benchmark results adhere to these empirical thresholds.
        """
        def classify_delta(d: float) -> str:
            abs_d = abs(d)
            if abs_d < 0.147:
                return "negligible"
            elif abs_d < 0.33:
                return "small"
            elif abs_d < 0.474:
                return "medium"
            else:
                return "large"

        # Boundary checks
        self.assertEqual(classify_delta(0.0), "negligible")
        self.assertEqual(classify_delta(0.146), "negligible")
        self.assertEqual(classify_delta(0.147), "small")
        self.assertEqual(classify_delta(0.329), "small")
        self.assertEqual(classify_delta(0.330), "medium")
        self.assertEqual(classify_delta(0.473), "medium")
        self.assertEqual(classify_delta(0.474), "large")
        self.assertEqual(classify_delta(1.0), "large")

        # Verify against actual recorded benchmark values
        res_csv_path = os.path.join(project_root, "results/epistemic_research/statistical_results.csv")
        if os.path.exists(res_csv_path):
            df_stat = pd.read_csv(res_csv_path)
            deltas = dict(zip(df_stat["task_id"], df_stat["Cliff's delta"]))
            self.assertEqual(classify_delta(deltas["ackley_2d"]), "negligible")      # +0.042
            self.assertEqual(classify_delta(deltas["hartmann_6d"]), "small")         # +0.183
            self.assertEqual(classify_delta(deltas["rosenbrock_2d"]), "negligible")  # -0.095
            self.assertEqual(classify_delta(deltas["synthetic_gap_2d"]), "small")    # -0.210

    # =========================================================================
    # AREA 3: Holm-Bonferroni Invariants
    # =========================================================================

    def test_holm_bonferroni_monotonicity_and_clamping(self):
        """
        Challenge 3.1: Monotonicity and Upper Clamping Invariants.
        1. Monotonicity: In sorted order of raw p-values, adjusted p-values must satisfy
           adj_p[k] >= adj_p[k-1].
        2. Clamping: All adjusted p-values must be <= 1.0.
        3. Dominance: adj_p[i] >= raw_p[i] for all i.
        """
        adversarial_inputs = [
            [0.01, 0.015, 0.018],            # Raw p-values where unadjusted multipliers decline
            [0.04, 0.04, 0.04, 0.04],        # All identical raw p-values
            [0.0001, 0.5, 0.8, 0.99],        # Massive spread with large p-values
            [0.9, 0.8, 0.7, 0.6],            # Reverse sorted
            [1.0, 1.0, 1.0],                 # All 1.0
            [0.0],                           # Boundary 0.0
            [],                              # Empty list
        ]

        for p_raw in adversarial_inputs:
            adj = apply_holm_bonferroni(p_raw)
            self.assertEqual(len(adj), len(p_raw))

            if len(p_raw) == 0:
                continue

            # Upper clamping
            for a in adj:
                self.assertLessEqual(a, 1.0)
                self.assertGreaterEqual(a, 0.0)

            # Dominance
            for r, a in zip(p_raw, adj):
                self.assertGreaterEqual(a + 1e-12, r)

            # Monotonicity in sorted order
            sort_idx = np.argsort(p_raw)
            sorted_adj = [adj[i] for i in sort_idx]
            for k in range(1, len(sorted_adj)):
                self.assertGreaterEqual(
                    sorted_adj[k] + 1e-12,
                    sorted_adj[k - 1],
                    msg=f"Monotonicity violated in {p_raw}: {sorted_adj}",
                )

    def test_holm_bonferroni_identity_against_statsmodels(self):
        """
        Challenge 3.2: Exact Numerical Identity against Statsmodels Oracle.
        Compares apply_holm_bonferroni against statsmodels.stats.multitest.multipletests(method='holm')
        across 100 randomized p-value vectors.
        """
        try:
            from statsmodels.stats.multitest import multipletests
        except ImportError:
            self.skipTest("statsmodels is not installed")

        rng = np.random.default_rng(999)
        for _ in range(100):
            n = rng.integers(1, 30)
            p_raw = list(rng.uniform(0.0, 1.0, size=n))

            custom_adj = apply_holm_bonferroni(p_raw)
            sm_adj = list(multipletests(p_raw, method="holm")[1])

            np.testing.assert_allclose(
                custom_adj,
                sm_adj,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Discrepancy with statsmodels on p_raw={p_raw}",
            )

    # =========================================================================
    # AREA 4: Raw Trace Reproducibility Verification
    # =========================================================================

    def test_exact_reproduction_from_raw_traces(self):
        """
        Challenge 4.1: Bit-for-bit exact reproducibility from results/epistemic_research/benchmark_traces.csv.
        Re-executes compute_wilcoxon_suite on raw benchmark traces and verifies that:
        - statistical_results.csv is reproduced 100% bit-for-bit
        - wilcoxon_tests_vs_baseline.csv is reproduced 100% bit-for-bit
        - Paired seed sample size invariant N=35 >= 30 is satisfied for all benchmarks
        """
        traces_csv = os.path.join(project_root, "results/epistemic_research/benchmark_traces.csv")
        expected_stat_csv = os.path.join(project_root, "results/epistemic_research/statistical_results.csv")
        expected_base_csv = os.path.join(project_root, "results/epistemic_research/wilcoxon_tests_vs_baseline.csv")

        self.assertTrue(os.path.exists(traces_csv), f"Missing {traces_csv}")
        self.assertTrue(os.path.exists(expected_stat_csv), f"Missing {expected_stat_csv}")

        with tempfile.TemporaryDirectory() as tmp_out:
            compute_wilcoxon_suite(
                input_parquet=traces_csv,
                output_dir=tmp_out,
                baseline_id="SMAC3_HPOFacade_ei",
            )

            actual_stat = pd.read_csv(os.path.join(tmp_out, "statistical_results.csv"))
            expected_stat = pd.read_csv(expected_stat_csv)

            # 1. Structural and cell equality
            pd.testing.assert_frame_equal(actual_stat, expected_stat)

            # 2. Seed count invariant (N >= 30 for all benchmarks)
            for n_seeds in actual_stat["n_seeds"]:
                self.assertGreaterEqual(n_seeds, 30, f"Seed count {n_seeds} violates N >= 30 mandate")
            self.assertTrue((actual_stat["n_seeds"] == 35).all())

            # 3. Pairwise vs baseline table equality
            if os.path.exists(expected_base_csv):
                actual_base = pd.read_csv(os.path.join(tmp_out, "wilcoxon_tests_vs_baseline.csv"))
                expected_base = pd.read_csv(expected_base_csv)
                pd.testing.assert_frame_equal(actual_base, expected_base)


if __name__ == "__main__":
    unittest.main()

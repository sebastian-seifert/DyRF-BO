import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from ep_extractors import UQExtractorRegistry


class TestDistanceEvidentialTreePath(unittest.TestCase):
    """Test suite for Tree-Path Topological Proximity in DA-EHRF extractor."""

    def setUp(self):
        np.random.seed(42)
        # 1D dataset for clear tree topology verification
        self.X_train_1d = np.array([[0.1], [0.3], [0.7], [0.9]])
        self.y_train_1d = np.array([1.0, 1.0, 5.0, 5.0])
        self.rf_1d = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=42)
        self.rf_1d.fit(self.X_train_1d, self.y_train_1d)

    def test_default_parameters(self):
        """Assert default spatial_metric is 'tree_path' and tree_decay_lambda is 3.0."""
        ext = DistanceAwareEvidentialExtractor(self.rf_1d)
        self.assertEqual(ext.spatial_metric, "tree_path")
        self.assertEqual(ext.tree_decay_lambda, 3.0)

    def test_invalid_spatial_metric_raises_error(self):
        """Passing an invalid spatial_metric must raise ValueError."""
        with self.assertRaises(ValueError):
            DistanceAwareEvidentialExtractor(self.rf_1d, spatial_metric="manhattan")

    def test_zero_step_identity(self):
        """Identical training samples have step distance 0 and V_spatial = 0."""
        ext = DistanceAwareEvidentialExtractor(self.rf_1d, spatial_metric="tree_path")
        ext.fit(self.X_train_1d, self.y_train_1d)

        # Candidate points exactly equal to training points
        leaf_matrix_train = ext.model.apply(ext.X_train)
        leaf_matrix_query = ext.model.apply(self.X_train_1d)
        np.testing.assert_array_equal(leaf_matrix_train, leaf_matrix_query)

        # Uncertainty signal on training data
        u_e_train = ext.extract_epistemic_signal(self.X_train_1d)
        self.assertEqual(len(u_e_train), len(self.X_train_1d))
        self.assertTrue(np.all(np.isfinite(u_e_train)))

        # Also check internal calculation: when query == train, d_T,min == 0
        # V_spatial must be 0, so U_E = sqrt(V_ens + V_leaf)
        # Compute V_ens + V_leaf manually to verify exact match
        estimators = ext.model.estimators_
        all_leaf_ids = ext.model.apply(self.X_train_1d)
        n_est = len(estimators)
        n_samples = len(self.X_train_1d)
        tree_means = np.zeros((n_est, n_samples))
        tree_inv_counts = np.zeros((n_est, n_samples))
        for m, est in enumerate(estimators):
            leaf_ids_m = all_leaf_ids[:, m]
            tree_means[m, :] = est.tree_.value[leaf_ids_m, 0, 0]
            counts_m = np.maximum(est.tree_.n_node_samples[leaf_ids_m], 1.0)
            tree_inv_counts[m, :] = 1.0 / counts_m

        v_ens = np.var(tree_means, axis=0, ddof=0)
        v_leaf = (ext.sigma_0 ** 2) * ext.kappa_leaf * np.mean(tree_inv_counts, axis=0)
        u_e_expected_zero_spatial = np.sqrt(np.maximum(v_ens + v_leaf, 0.0))

        np.testing.assert_allclose(
            u_e_train,
            u_e_expected_zero_spatial,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Spatial uncertainty on training data was not exactly zero."
        )

    def test_step_calculation_correctness_known_topology(self):
        """Verifies tree step calculation against exact manual graph distance."""
        # Train a single tree with known binary tree splits
        X_known = np.array([[0.1], [0.3], [0.7], [0.9]])
        y_known = np.array([1.0, 2.0, 3.0, 4.0])
        single_tree = DecisionTreeRegressor(max_depth=2, random_state=42)
        # Points structured to create depth-2 balanced tree:
        # root splits at ~0.5
        # left splits at ~0.2 (leaves: 0.1 and 0.3)
        # right splits at ~0.8 (leaves: 0.7 and 0.9)
        single_tree.fit(X_known, y_known)

        # Mock a forest of 1 tree for exact verification
        class SingleTreeForest:
            def __init__(self, tree):
                self.estimators_ = [tree]
            def apply(self, X):
                return self.estimators_[0].apply(X).reshape(-1, 1)

        st_forest = SingleTreeForest(single_tree)
        ext = DistanceAwareEvidentialExtractor(st_forest, spatial_metric="tree_path")
        ext.fit(X_known, y_known)

        # Node depths and ancestors
        # Leaf for 0.1 and leaf for 0.3 share left child (depth 1) -> distance = 2 + 2 - 2(1) = 2 steps
        # Leaf for 0.1 and leaf for 0.7 share root (depth 0) -> distance = 2 + 2 - 2(0) = 4 steps
        leaves = single_tree.apply(X_known)
        u_01 = leaves[0]
        u_03 = leaves[1]
        u_07 = leaves[2]

        self.assertNotEqual(u_01, u_03)
        self.assertNotEqual(u_01, u_07)

        # Access precomputed path/depth structures
        depths = ext.tree_node_depths[0]
        ancestors = ext.tree_node_ancestors[0]

        # Verify ancestor paths
        self.assertIn(0, ancestors[u_01])
        self.assertIn(0, ancestors[u_07])

        # LCA of u_01 and u_03 has depth 1
        common_01_03 = set(ancestors[u_01]).intersection(set(ancestors[u_03]))
        lca_01_03 = max(common_01_03, key=lambda k: depths[k])
        self.assertEqual(depths[lca_01_03], 1)
        step_dist_01_03 = depths[u_01] + depths[u_03] - 2 * depths[lca_01_03]
        self.assertEqual(step_dist_01_03, 2)

        # LCA of u_01 and u_07 has depth 0 (root)
        common_01_07 = set(ancestors[u_01]).intersection(set(ancestors[u_07]))
        lca_01_07 = max(common_01_07, key=lambda k: depths[k])
        self.assertEqual(depths[lca_01_07], 0)
        step_dist_01_07 = depths[u_01] + depths[u_07] - 2 * depths[lca_01_07]
        self.assertEqual(step_dist_01_07, 4)

    def test_monotonicity_with_tree_divergence(self):
        """Candidate points with larger tree divergence produce larger step distances and higher V_spatial."""
        # Train dataset with two clusters: [0.0, 0.2] and [0.8, 1.0]
        X_train = np.array([[0.05], [0.10], [0.15], [0.85], [0.90], [0.95]])
        y_train = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(X_train, y_train)

        ext = DistanceAwareEvidentialExtractor(rf, spatial_metric="tree_path", c_spatial=2.0)
        ext.fit(X_train, y_train)

        # Near candidate (in same cluster leaf)
        x_near = np.array([[0.08]])
        # Far candidate in unexplored intermediate region
        x_gap = np.array([[0.50]])

        u_near = ext.extract_epistemic_signal(x_near)[0]
        u_gap = ext.extract_epistemic_signal(x_gap)[0]

        self.assertGreater(
            u_gap,
            u_near,
            f"Gap candidate ({u_gap:.4f}) should have higher uncertainty than near candidate ({u_near:.4f})"
        )

    def test_depth_normalization_and_asymptotic_saturation(self):
        """Depth normalization with lambda=3.0: extreme divergence reaches ~(1 - e^-3)*sigma_0^2."""
        # Construct tree with max-depth 1 where training point is in left leaf and query is in right leaf
        X_fit = np.array([[0.1], [0.9]])
        y_fit = np.array([0.0, 10.0])
        single_tree = DecisionTreeRegressor(max_depth=1, random_state=42)
        single_tree.fit(X_fit, y_fit)

        class SingleTreeForest:
            def __init__(self, tree):
                self.estimators_ = [tree]
            def apply(self, X):
                return self.estimators_[0].apply(X).reshape(-1, 1)

        st_forest = SingleTreeForest(single_tree)
        lambda_val = 3.0
        ext = DistanceAwareEvidentialExtractor(
            st_forest,
            spatial_metric="tree_path",
            tree_decay_lambda=lambda_val,
            c_spatial=1.0,
            kappa_leaf=0.0  # Isolate spatial variance component
        )
        # Training dataset contains only the left-leaf sample [0.1]
        ext.fit(X_fit[:1], y_fit[:1])

        # Query point [0.9] lands in right leaf (LCA = root, distance = 1 + 1 = 2)
        # 2 * K_depth = 2 * 1 = 2, so distance ratio = 2 / 2 = 1.0
        u_far = ext.extract_epistemic_signal(np.array([[0.9]]))[0]

        # Theoretical saturation: (1 - exp(-3.0)) * sigma_0^2
        sigma_0 = ext.sigma_0
        expected_v_spatial = (sigma_0 ** 2) * (1.0 - np.exp(-lambda_val))
        expected_saturation_factor = 1.0 - np.exp(-lambda_val)

        self.assertAlmostEqual(expected_saturation_factor, 0.9502, places=3)
        np.testing.assert_allclose(
            u_far ** 2,
            expected_v_spatial,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Spatial variance did not reach exact (1 - exp(-3.0)) * sigma_0^2 at maximum tree divergence."
        )

    def test_backward_compatibility_euclidean(self):
        """spatial_metric='euclidean' produces exact same output as previous implementation."""
        X_train = np.random.uniform(0.0, 1.0, size=(30, 2))
        y_train = X_train[:, 0] + 2.0 * X_train[:, 1]
        rf = RandomForestRegressor(n_estimators=5, random_state=42)
        rf.fit(X_train, y_train)

        # Extractor with explicit spatial_metric="euclidean"
        ext_euc = DistanceAwareEvidentialExtractor(rf, spatial_metric="euclidean", lengthscale=0.2)
        ext_euc.fit(X_train, y_train)

        X_test = np.array([
            [0.2, 0.3],
            [1.5, 1.5],
            [-2.0, 3.0]
        ])

        u_euc = ext_euc.extract_epistemic_signal(X_test)

        # Manually compute spatial variance using Euclidean formula
        X_w = X_test * ext_euc.weights
        min_sq_dists = np.min(
            np.sum((X_w[:, None, :] - ext_euc.X_train_w[None, :, :]) ** 2, axis=2),
            axis=1
        )
        exponent = np.clip(-min_sq_dists / (2.0 * (ext_euc.lengthscale ** 2)), -700.0, 0.0)
        v_spatial_manual = (ext_euc.sigma_0 ** 2) * ext_euc.c_spatial * (1.0 - np.exp(exponent))

        # Check that u_euc squared equals v_ens + v_leaf + v_spatial_manual
        estimators = rf.estimators_
        all_leaf_ids = rf.apply(X_test)
        n_est = len(estimators)
        tree_means = np.zeros((n_est, len(X_test)))
        tree_inv_counts = np.zeros((n_est, len(X_test)))
        for m, est in enumerate(estimators):
            leaf_ids_m = all_leaf_ids[:, m]
            tree_means[m, :] = est.tree_.value[leaf_ids_m, 0, 0]
            counts_m = np.maximum(est.tree_.n_node_samples[leaf_ids_m], 1.0)
            tree_inv_counts[m, :] = 1.0 / counts_m
        v_ens = np.var(tree_means, axis=0, ddof=0)
        v_leaf = (ext_euc.sigma_0 ** 2) * ext_euc.kappa_leaf * np.mean(tree_inv_counts, axis=0)
        u_expected = np.sqrt(v_ens + v_leaf + v_spatial_manual)

        np.testing.assert_allclose(u_euc, u_expected, rtol=1e-5, atol=1e-7)

    def test_high_dimensional_noise_robustness_10d(self):
        """10D problem with 9 uninformative noise features: tree_path preserves OOD distinction."""
        np.random.seed(123)
        n_train = 60
        # Informative feature in dimension 0, noise in dimensions 1..9
        X_train = np.zeros((n_train, 10))
        X_train[:, 0] = np.random.uniform(0.0, 1.0, n_train)
        X_train[:, 1:] = np.random.uniform(-1.0, 1.0, size=(n_train, 9))
        y_train = np.sin(X_train[:, 0] * np.pi)

        rf = RandomForestRegressor(n_estimators=15, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        ext_tree = DistanceAwareEvidentialExtractor(rf, spatial_metric="tree_path", c_spatial=1.5)
        ext_tree.fit(X_train, y_train)

        # In-distribution test points (x0 in [0, 1], random noise)
        X_id = np.zeros((20, 10))
        X_id[:, 0] = np.random.uniform(0.1, 0.9, 20)
        X_id[:, 1:] = np.random.uniform(-1.0, 1.0, size=(20, 9))

        # Out-of-distribution test points (x0 in [2, 3], random noise)
        X_ood = np.zeros((20, 10))
        X_ood[:, 0] = np.random.uniform(2.0, 3.0, 20)
        X_ood[:, 1:] = np.random.uniform(-1.0, 1.0, size=(20, 9))

        u_id = ext_tree.extract_epistemic_signal(X_id)
        u_ood = ext_tree.extract_epistemic_signal(X_ood)

        # Tree path must clearly assign higher epistemic uncertainty to OOD points
        mean_id = np.mean(u_id)
        mean_ood = np.mean(u_ood)
        self.assertGreater(
            mean_ood,
            mean_id * 1.3,
            f"Tree-path OOD ({mean_ood:.4f}) should substantially exceed ID ({mean_id:.4f})"
        )

    def test_edge_cases(self):
        """Edge cases: single sample (N=1), constant target, empty input array."""
        # 1. Single sample N=1
        X_single = np.array([[0.5, 0.5]])
        y_single = np.array([3.0])
        rf_single = RandomForestRegressor(n_estimators=3, random_state=42)
        rf_single.fit(X_single, y_single)

        ext_single = DistanceAwareEvidentialExtractor(rf_single, spatial_metric="tree_path")
        ext_single.fit(X_single, y_single)

        u_single_query = ext_single.extract_epistemic_signal(np.array([[0.5, 0.5], [2.0, 2.0]]))
        self.assertEqual(len(u_single_query), 2)
        self.assertTrue(np.all(np.isfinite(u_single_query)))
        self.assertTrue(np.all(u_single_query >= 0.0))

        # 2. Constant target
        X_const = np.random.uniform(0.0, 1.0, size=(20, 2))
        y_const = np.full(20, 7.0)
        rf_const = RandomForestRegressor(n_estimators=3, random_state=42)
        rf_const.fit(X_const, y_const)

        ext_const = DistanceAwareEvidentialExtractor(rf_const, spatial_metric="tree_path")
        ext_const.fit(X_const, y_const)
        u_const = ext_const.extract_epistemic_signal(np.array([[0.5, 0.5]]))
        self.assertTrue(np.all(np.isfinite(u_const)))
        self.assertTrue(np.all(u_const >= 0.0))

        # 3. Empty input
        u_empty = ext_const.extract_epistemic_signal(np.empty((0, 2)))
        self.assertEqual(len(u_empty), 0)


    def test_zero_oom_memory_streaming_high_dim(self):
        """Asserts fit() and extract_epistemic_signal() do not store huge distance matrices and stream within < 50MB."""
        import tracemalloc

        rng = np.random.default_rng(42)
        n_train = 1000
        n_test = 200
        X_train = rng.uniform(-5.0, 5.0, size=(n_train, 10))
        y_train = rng.normal(0.0, 1.0, size=n_train)
        X_test = rng.uniform(-5.0, 5.0, size=(n_test, 10))

        rf = RandomForestRegressor(n_estimators=10, min_samples_leaf=5, random_state=42)
        rf.fit(X_train, y_train)

        tracemalloc.start()
        ext = DistanceAwareEvidentialExtractor(rf, spatial_metric="tree_path")
        ext.fit(X_train, y_train)

        # Assert no giant dense distance matrices stored on self
        self.assertFalse(
            hasattr(ext, "tree_train_leaf_dists") and len(getattr(ext, "tree_train_leaf_dists", [])) > 0,
            "Extractor must not store dense tree_train_leaf_dists matrices in RAM."
        )
        self.assertFalse(
            hasattr(ext, "tree_leaf_dist_matrices") and len(getattr(ext, "tree_leaf_dist_matrices", [])) > 0,
            "Extractor must not store dense tree_leaf_dist_matrices in RAM."
        )

        # Extract uncertainty on test batch
        u_signal = ext.extract_epistemic_signal(X_test)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(u_signal), n_test)
        self.assertTrue(np.all(np.isfinite(u_signal)))
        self.assertTrue(np.all(u_signal >= 0.0))

        # Peak RAM must be strictly below 50 MB
        peak_mb = peak / (1024 * 1024)
        self.assertLess(
            peak_mb,
            50.0,
            f"Peak memory during fit and extraction ({peak_mb:.2f} MB) exceeded 50 MB threshold."
        )


if __name__ == "__main__":
    unittest.main()


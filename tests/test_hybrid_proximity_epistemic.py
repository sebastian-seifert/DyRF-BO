import os
import sys
import unittest
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Hybrid_Proximity_Epistemic_UQ import HybridProximityEpistemicUQ

class MockTreeData:
    def __init__(self):
        self.node_count = 5
        self.children_left = np.array([1, 3, -1, -1, -1], dtype=np.int32)
        self.children_right = np.array([2, 4, -1, -1, -1], dtype=np.int32)
        self.impurity = np.array([1.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float64)
        self.n_node_samples = np.array([10, 6, 4, 3, 3], dtype=np.int64)
        self.value = np.zeros((5, 1, 1), dtype=np.float64)

class MockTree:
    def __init__(self):
        self.tree_ = MockTreeData()
        self.random_state = 42

class MockForest:
    def __init__(self):
        self.estimators_ = [MockTree()]
        self.oob_score = True
        self.n_estimators = 1
        self.oob_prediction_ = np.array([1.0, 2.0, 3.0])

    def apply(self, X):
        return np.array([[3], [4], [2]], dtype=np.int32)

class TestHybridProximityEpistemic(unittest.TestCase):
    def test_hybrid_proximity_epistemic_math(self):
        # Setup mock forest
        forest = MockForest()
        
        # 3 training samples
        X_train = np.array([[0.1], [0.2], [0.8]])
        y_train = np.array([1.0, 2.0, 3.0])
        
        # Query points
        X_test = np.array([[0.1], [0.2], [0.8]])
        
        # Initialize the hybrid wrapper
        hybrid_uq = HybridProximityEpistemicUQ(
            model=forest,
            X_train=X_train,
            y_train=y_train,
            base_epistemic_method="likelihood",
            proximity_decay_lambda=1.0,
            normalize_by_depth=False,
            lambda_blend=0.4,
            k_neighbors=2
        )
        
        # Mock precomputed train epistemic values
        hybrid_uq.train_epistemic_values = np.array([0.1, 0.5, 2.0], dtype=np.float32)
        
        # Mock proximity model's leaf assignment structures
        hybrid_uq.prox_model.leaf_matrix_train = np.array([[3], [4], [2]], dtype=np.int32)
        hybrid_uq.prox_model.in_bag_leaves = hybrid_uq.prox_model.leaf_matrix_train
        hybrid_uq.prox_model.in_bag_leaves_xp = hybrid_uq.prox_model.xp.asarray(hybrid_uq.prox_model.in_bag_leaves)
        hybrid_uq.prox_model.in_bag_indices = np.array([[1], [1], [1]], dtype=np.float32)
        hybrid_uq.prox_model.in_bag_counts = np.array([[1], [1], [1]], dtype=np.float32)
        hybrid_uq.prox_model.leaf_sizes = [np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)]
        hybrid_uq.prox_model.train_weights = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        hybrid_uq.prox_model.train_weights_xp = hybrid_uq.prox_model.xp.asarray(hybrid_uq.prox_model.train_weights)
        
        # Mock tree distance matrices computed in proximity wrapper
        d_t = np.array([[0.0, 2.0, 3.0],
                        [2.0, 0.0, 3.0],
                        [3.0, 3.0, 0.0]], dtype=np.float32)
        hybrid_uq.prox_model.tree_leaf_distances = [d_t]
        
        id_to_dense = np.full(5, -1, dtype=np.int32)
        id_to_dense[3] = 0
        id_to_dense[4] = 1
        id_to_dense[2] = 2
        hybrid_uq.prox_model.tree_leaf_id_to_dense = [hybrid_uq.prox_model.xp.asarray(id_to_dense)]
        
        # Monkeypatch get_base_epistemic to return mock query values
        def mock_get_base_epistemic(X):
            return np.array([0.1, 0.5, 2.0], dtype=np.float32)
        
        hybrid_uq._get_base_epistemic = mock_get_base_epistemic
        
        uq_vals = hybrid_uq.compute_uq(X_test)
        
        # Assert query 0 matches the blended math exactly (approx 0.1190724)
        np.testing.assert_allclose(uq_vals[0], 0.1190724, rtol=1e-3)
        
        # Test lambda_blend=0.0 boundary behavior
        hybrid_uq.lambda_blend = 0.0
        uq_vals_l0 = hybrid_uq.compute_uq(X_test)
        np.testing.assert_allclose(uq_vals_l0, [0.1, 0.5, 2.0], rtol=1e-3)
        
        # Test lambda_blend=1.0 boundary behavior
        hybrid_uq.lambda_blend = 1.0
        uq_vals_l1 = hybrid_uq.compute_uq(X_test)
        np.testing.assert_allclose(uq_vals_l1[0], 0.147681, rtol=1e-3)

if __name__ == "__main__":
    unittest.main()

import os
import sys
import numpy as np

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

class MockTreeData:
    def __init__(self):
        self.node_count = 5
        self.children_left = np.array([1, 3, -1, -1, -1], dtype=np.int32)
        self.children_right = np.array([2, 4, -1, -1, -1], dtype=np.int32)
        self.impurity = np.array([1.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float64)
        self.n_node_samples = np.array([10, 6, 4, 3, 3], dtype=np.int64)

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

def test_normalized_proximity():
    # Setup mock forest
    forest = MockForest()
    
    # 3 training samples
    X_train = np.array([[0.1], [0.2], [0.8]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    # Initialize wrapper with topological decay lambda = 1.0 and normalize_by_depth = True
    uq_wrapper = GPUProximityRegressionUQ(
        forest, X_train, y_train, 
        device="cpu", 
        topological_decay_lambda=1.0,
        normalize_by_depth=True
    )
    
    # Manually configure in-bag data to bypass actual fit logic
    uq_wrapper.leaf_matrix_train = np.array([[3], [4], [2]], dtype=np.int32)
    uq_wrapper.in_bag_leaves = uq_wrapper.leaf_matrix_train
    uq_wrapper.in_bag_leaves_xp = uq_wrapper.xp.asarray(uq_wrapper.in_bag_leaves)
    
    uq_wrapper.in_bag_indices = np.array([[1], [1], [1]], dtype=np.float32)
    uq_wrapper.in_bag_counts = np.array([[1], [1], [1]], dtype=np.float32)
    
    uq_wrapper.leaf_sizes = [np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)]
    uq_wrapper.train_weights = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    uq_wrapper.train_weights_xp = uq_wrapper.xp.asarray(uq_wrapper.train_weights)
    
    uq_wrapper.oob_residuals = np.array([0.1, 0.5, 2.0], dtype=np.float32)
    uq_wrapper.oob_residuals_xp = uq_wrapper.xp.asarray(uq_wrapper.oob_residuals)
    
    X_test = np.array([[0.1], [0.2], [0.8]])
    
    # 1. Trigger fit preprocessing to check max depth tracking
    uq_wrapper._precompute_tree_paths(0)
    # The max depth of the mock tree is 2 (nodes 3 and 4 are at depth 2)
    expected_max_depth = 2
    actual_max_depth = int(uq_wrapper.tree_max_depths[0])
    assert actual_max_depth == expected_max_depth, f"Expected max depth {expected_max_depth}, got {actual_max_depth}"
    
    # 2. Check leaf distance matrices are computed correctly
    d_t = uq_wrapper.compute_tree_topological_distances([3, 4, 2], [3, 4, 2], 0)
    # Expected distance matrix:
    # d(3,3)=0, d(3,4)=2, d(3,2)=3
    # d(4,3)=2, d(4,4)=0, d(4,2)=3
    # d(2,3)=3, d(2,4)=3, d(2,2)=0
    expected_d_t = np.array([[0.0, 2.0, 3.0],
                             [2.0, 0.0, 3.0],
                             [3.0, 3.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(d_t, expected_d_t)
    
    # 3. Cache the distance matrix manually as done in fit()
    uq_wrapper.tree_leaf_distances = [d_t]
    # Simple identity mapping array for absolute node ID to dense leaf index
    id_to_dense = np.full(5, -1, dtype=np.int32)
    id_to_dense[3] = 0
    id_to_dense[4] = 1
    id_to_dense[2] = 2
    uq_wrapper.tree_leaf_id_to_dense = [uq_wrapper.xp.asarray(id_to_dense)]
    
    # 4. Compute UQ
    uq_vals = uq_wrapper.compute_uq(X_test, n_neighbors="auto", level=0.95)
    print("\nCalculated UQ interval widths (Normalized Weighted Quantiles):", uq_vals)
    
    # For query 0:
    # d_t values: [0.0, 2.0, 3.0]
    # Since normalize_by_depth is True, divisor is 2 * max_depth = 4.
    # decay: [exp(-0.0), exp(-2/4), exp(-3/4)] = [1.0, exp(-0.5), exp(-0.75)]
    # decay = [1.0, 0.6065306597, 0.4723665527]
    # Sum of weights: 1.0 + 0.6065306597 + 0.4723665527 = 2.078897
    # Normalized weights: [0.481024, 0.291756, 0.227220]
    # Cumulative normalized weights: [0.481024, 0.772780, 1.0]
    # Values: [0.1, 0.5, 2.0]
    # Level = 0.95 -> alpha_lwr = 0.025, alpha_upr = 0.975
    # Lwr: first cumulative >= 0.025 is at index 0 (0.481024) -> value = 0.1
    # Upr: cum at index 1 (0.772780) < 0.975; cum at index 2 (1.0) >= 0.975.
    # Interpolation:
    # fraction = (0.975 - 0.772780) / (1.0 - 0.772780) = 0.20222 / 0.22722 = 0.88997
    # value = 0.5 + 1.5 * 0.88997 = 1.834955
    # uq_val = 1.834955 - 0.1 = 1.734955
    np.testing.assert_allclose(uq_vals[0], 1.734955, rtol=1e-3)
    
    print("✓ SUCCESS: Topological Normalized Proximity match expected values!")

if __name__ == "__main__":
    try:
        test_normalized_proximity()
        sys.exit(0)
    except AssertionError as e:
        print("❌ FAILED: Assert mismatch:", e)
        sys.exit(1)
    except Exception as e:
        print("❌ FAILED: Error raised during test:", e)
        sys.exit(1)

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
        # Query points map to leaves 3, 4, 2
        return np.array([[3], [4], [2]], dtype=np.int32)

def test_topological_neighbor_selection():
    # Setup mock forest
    forest = MockForest()
    
    # 3 training samples
    X_train = np.array([[0.1], [0.2], [0.8]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    # Initialize wrapper with topological decay lambda = 1.0
    uq_wrapper = GPUProximityRegressionUQ(
        forest, X_train, y_train, 
        device="cpu", 
        topological_decay_lambda=1.0
    )
    
    # Manually configure in-bag data to bypass actual fit logic
    uq_wrapper.leaf_matrix_train = np.array([[3], [4], [2]], dtype=np.int32)
    uq_wrapper.in_bag_leaves = uq_wrapper.leaf_matrix_train
    uq_wrapper.in_bag_leaves_xp = uq_wrapper.xp.asarray(uq_wrapper.in_bag_leaves)
    
    uq_wrapper.in_bag_indices = np.array([[1], [1], [1]], dtype=np.float32)
    uq_wrapper.in_bag_counts = np.array([[1], [1], [1]], dtype=np.float32)
    
    # leaf_sums for node indices 0 to 4: leaf_sums[3]=1, leaf_sums[4]=1, leaf_sums[2]=1
    uq_wrapper.leaf_sizes = [np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)]
    uq_wrapper.train_weights = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    uq_wrapper.train_weights_xp = uq_wrapper.xp.asarray(uq_wrapper.train_weights)
    
    # Mock residuals
    uq_wrapper.oob_residuals = np.array([0.1, 0.5, 2.0], dtype=np.float32)
    uq_wrapper.oob_residuals_xp = uq_wrapper.xp.asarray(uq_wrapper.oob_residuals)
    
    # Query points: we query X_test which maps to leaves 3, 4, 2
    X_test = np.array([[0.1], [0.2], [0.8]])
    
    # Compute UQ (level=0.95, n_neighbors=2)
    # Expected behavior for Method A:
    # 1. d_t for query 0: d(3,3)=0, d(3,4)=2, d(3,2)=3
    # 2. P_walk: P(0,0) = e^0 = 1.0; P(0,1) = e^-2 = 0.1353; P(0,2) = e^-3 = 0.0498
    # 3. For n_neighbors=2, selects top 2: sample 0 (resid=0.1) and sample 1 (resid=0.5)
    # 4. Quantiles for [0.1, 0.5] at level=0.95:
    #    alpha_lwr = 0.025 -> 0.1 + (0.5 - 0.1) * 0.025 = 0.11
    #    alpha_upr = 0.975 -> 0.1 + (0.5 - 0.1) * 0.975 = 0.49
    #    uq_val = 0.49 - 0.11 = 0.38
    uq_vals = uq_wrapper.compute_uq(X_test, n_neighbors=2, level=0.95)
    
    print("\nCalculated UQ interval widths:", uq_vals)
    
    # uq_val = 0.38 / normal_divisor = 0.38 / 3.919928 = 0.0969405
    from scipy.stats import norm
    normal_divisor = 2.0 * float(norm.ppf((1.0 + 0.95) / 2.0))
    expected_sigma = 0.38 / normal_divisor
    np.testing.assert_allclose(uq_vals[0], expected_sigma, rtol=1e-3)
    print("✓ SUCCESS: Topological Neighbor Selection matches expected values!")

if __name__ == "__main__":
    try:
        test_topological_neighbor_selection()
        sys.exit(0)
    except AssertionError as e:
        print("❌ FAILED: Assert mismatch:", e)
        sys.exit(1)
    except Exception as e:
        print("❌ FAILED: Error raised during test:", e)
        sys.exit(1)

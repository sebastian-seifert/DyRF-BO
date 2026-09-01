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

def test_topological_weighted_quantiles():
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
    
    uq_wrapper.leaf_sizes = [np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)]
    uq_wrapper.train_weights = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    uq_wrapper.train_weights_xp = uq_wrapper.xp.asarray(uq_wrapper.train_weights)
    
    # Mock residuals
    uq_wrapper.oob_residuals = np.array([0.1, 0.5, 2.0], dtype=np.float32)
    uq_wrapper.oob_residuals_xp = uq_wrapper.xp.asarray(uq_wrapper.oob_residuals)
    
    # Query points: we query X_test which maps to leaves 3, 4, 2
    X_test = np.array([[0.1], [0.2], [0.8]])
    
    # Compute UQ (level=0.95, n_neighbors="auto")
    # Expected behavior for Method B (Topological Weighted Quantiles):
    # 1. d_t for query 0: d(3,3)=0, d(3,4)=2, d(3,2)=3
    # 2. P_walk: P(0,0)=1.0, P(0,1)=e^-2=0.135335, P(0,2)=e^-3=0.049787
    # 3. Sum of weights: 1.185122
    # 4. Normalized cumulative weights: [0.843795, 0.957997, 1.0]
    # 5. Weighted Quantiles for [0.1, 0.5, 2.0] at level=0.95 (alpha_lwr=0.025, alpha_upr=0.975):
    #    Lwr: cumulative weight at index 0 (0.843795) >= 0.025 -> index 0 (val = 0.1)
    #    Upr: cumulative weight at index 1 is 0.957997 < 0.975; index 2 is 1.0 >= 0.975.
    #         Interpolate between index 1 (val=0.5, cum=0.957997) and index 2 (val=2.0, cum=1.0):
    #         fraction = (0.975 - 0.957997) / (1.0 - 0.957997) = 0.017003 / 0.042003 = 0.4048
    #         val = 0.5 + 1.5 * 0.4048 = 1.1072
    #         uq_val = 1.1072 - 0.1 = 1.0072
    uq_vals = uq_wrapper.compute_uq(X_test, n_neighbors="auto", level=0.95)
    
    print("\nCalculated UQ interval widths (Weighted Quantiles):", uq_vals)
    
    # uq_val = (1.1072 - 0.1) / normal_divisor = 1.0072 / 3.919928 = 0.256943
    from scipy.stats import norm
    normal_divisor = 2.0 * float(norm.ppf((1.0 + 0.95) / 2.0))
    expected_sigma = 1.0072 / normal_divisor
    np.testing.assert_allclose(uq_vals[0], expected_sigma, rtol=1e-3)
    print("✓ SUCCESS: Topological Weighted Quantiles match expected values!")

if __name__ == "__main__":
    try:
        test_topological_weighted_quantiles()
        sys.exit(0)
    except AssertionError as e:
        print("❌ FAILED: Assert mismatch:", e)
        sys.exit(1)
    except Exception as e:
        print("❌ FAILED: Error raised during test:", e)
        sys.exit(1)

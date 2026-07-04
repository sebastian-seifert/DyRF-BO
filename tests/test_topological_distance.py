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
        # Dummy leaf apply matching shape
        return np.array([[3], [4], [2]], dtype=np.int32)

def test_topological_distances():
    # Setup mock forest
    forest = MockForest()
    
    # Simple training data: 3 points
    X_train = np.array([[0.1], [0.2], [0.8]])
    y_train = np.array([1.0, 2.0, 3.0])
    
    # Initialize wrapper (using CPU backend for simple parity testing)
    uq_wrapper = GPUProximityRegressionUQ(forest, X_train, y_train, device="cpu")
    
    # Mock precomputed leaf matrix train
    # Sample 0 is in leaf 3, Sample 1 is in leaf 4, Sample 2 is in leaf 2
    uq_wrapper.leaf_matrix_train = np.array([[3], [4], [2]], dtype=np.int32)
    uq_wrapper.in_bag_leaves = uq_wrapper.leaf_matrix_train
    uq_wrapper.in_bag_leaves_xp = uq_wrapper.xp.asarray(uq_wrapper.in_bag_leaves)
    
    # Re-run fit step manually for our custom mock structures
    # Precompute path mappings for tree 0
    uq_wrapper._precompute_tree_paths(tree_idx=0)
    
    # Test queries: Leaf 3, 4, and 2
    leaf_test = np.array([3, 4, 2], dtype=np.int32)
    leaf_train = np.array([3, 4, 2], dtype=np.int32)
    
    # Calculate distances
    d_t = uq_wrapper.compute_tree_topological_distances(leaf_test, leaf_train, tree_idx=0)
    
    print("\nCalculated topological distance matrix:\n", d_t)
    
    # Expected distance matrix:
    # d(3, 3)=0, d(3, 4)=2, d(3, 2)=3
    # d(4, 3)=2, d(4, 4)=0, d(4, 2)=3
    # d(2, 3)=3, d(2, 4)=3, d(2, 2)=0
    expected = np.array([
        [0, 2, 3],
        [2, 0, 3],
        [3, 3, 0]
    ], dtype=np.int32)
    
    np.testing.assert_array_equal(d_t, expected)
    print("✓ SUCCESS: Topological distance matrix matches expected values!")

if __name__ == "__main__":
    try:
        test_topological_distances()
        sys.exit(0)
    except AssertionError as e:
        print("❌ FAILED: Assert mismatch:", e)
        sys.exit(1)
    except Exception as e:
        print("❌ FAILED: Error raised during test:", e)
        sys.exit(1)

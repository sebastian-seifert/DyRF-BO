import unittest
import numpy as np
from data_generator import generate_data

class TestManifoldOOD(unittest.TestCase):
    def setUp(self):
        # Setup a simple 1D, 2D, and 3D test function mapping
        self.func_dict = {
            "sin_1d": {
                "func": lambda x: np.sin(x),
                "gap": [4.0, 6.0],
                "range": [0.0, 10.0]
            },
            "sin_cos_2d": {
                "func": lambda x, y: np.sin(x) * np.cos(y),
                "gap": [4.0, 6.0],
                "range": [0.0, 10.0]
            },
            "sin_cos_sin_3d": {
                "func": lambda x, y, z: np.sin(x) * np.cos(y) * np.sin(z),
                "gap": [4.0, 6.0],
                "range": [0.0, 10.0]
            }
        }

    def test_manifold_shapes_1d(self):
        # For ndim = 1, d = 0 (manifold is a point)
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            self.func_dict, "sin_1d", seed=42, points_per_dim=100, ood_type="manifold"
        )
        # Check shapes
        self.assertEqual(X_train.shape[1], 1)
        self.assertEqual(X_test.shape[1], 1)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        self.assertEqual(len(y_true_binary), len(X_test))
        
        # ID test set (y_true_binary == 0) should cluster near c = 5.0
        X_id = X_test[y_true_binary == 0]
        self.assertTrue(np.all(np.abs(X_id - 5.0) < 0.3))
        
        # OOD test set (y_true_binary == 1) should be shifted away
        X_ood = X_test[y_true_binary == 1]
        self.assertTrue(np.all(np.abs(X_ood - 5.0) > 1.5))

    def test_manifold_shapes_2d(self):
        # For ndim = 2, d = 1 (manifold is a curve)
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            self.func_dict, "sin_cos_2d", seed=42, points_per_dim=50, ood_type="manifold"
        )
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(X_test.shape[1], 2)
        
        # Manifold curve is x_2 = f(x_1)
        # Check that ID test points lie close to the manifold curve
        X_id = X_test[y_true_binary == 0]
        
        # Center = 5.0, width = 10.0, A = 2.0, omega = 2pi/10 = pi/5
        # f(z) = 5.0 + 2.0 * sin(pi/5 * z)
        z = X_id[:, 0]
        f_z = 5.0 + 2.0 * np.sin((2.0 * np.pi / 10.0) * z)
        
        # Distance to manifold along y-axis should be small (noise std=0.05)
        self.assertTrue(np.mean(np.abs(X_id[:, 1] - f_z)) < 0.1)
        
        # OOD test points should be shifted away along normal vector
        X_ood = X_test[y_true_binary == 1]
        z_ood = X_ood[:, 0]
        f_z_ood = 5.0 + 2.0 * np.sin((2.0 * np.pi / 10.0) * z_ood)
        
        # Distance to manifold should be significant
        self.assertTrue(np.mean(np.abs(X_ood[:, 1] - f_z_ood)) > 0.8)

    def test_manifold_shapes_3d(self):
        # For ndim = 3, d = 2 (manifold is a surface)
        X_train, y_train, X_test, y_test, y_true_binary = generate_data(
            self.func_dict, "sin_cos_sin_3d", seed=42, ood_type="manifold"
        )
        self.assertEqual(X_train.shape[1], 3)
        self.assertEqual(X_test.shape[1], 3)
        
        X_id = X_test[y_true_binary == 0]
        z = X_id[:, :2]
        # f(z) = 5.0 + (2.0 / 2) * (sin(pi/5 * z_1) + sin(pi/5 * z_2))
        f_z = 5.0 + 1.0 * (np.sin((2.0 * np.pi / 10.0) * z[:, 0]) + np.sin((2.0 * np.pi / 10.0) * z[:, 1]))
        
        self.assertTrue(np.mean(np.abs(X_id[:, 2] - f_z)) < 0.1)

if __name__ == "__main__":
    unittest.main()

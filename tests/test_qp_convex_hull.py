import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.qp_convex_hull import (
    ConvexHullProjectionSolver,
    project_simplex,
)


class TestSimplexProjection:
    """Tests for Euclidean projection onto the standard simplex."""

    def test_1d_simplex_projection(self):
        v = np.array([0.5, 0.8, -0.2, 1.2])
        w = project_simplex(v, z=1.0)
        assert w.shape == (4,)
        assert np.isclose(np.sum(w), 1.0, atol=1e-7)
        assert np.all(w >= -1e-12)

    def test_2d_batch_simplex_projection(self):
        V = np.array([
            [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.34],
            [10.0, -5.0, 2.0],
            [-1.0, -2.0, -3.0],
        ])
        W = project_simplex(V, z=1.0)
        assert W.shape == (4, 3)
        assert np.allclose(np.sum(W, axis=1), 1.0, atol=1e-7)
        assert np.all(W >= -1e-12)

    def test_custom_z_scale(self):
        v = np.array([1.0, 2.0, 3.0])
        z = 3.5
        w = project_simplex(v, z=z)
        assert np.isclose(np.sum(w), z, atol=1e-7)
        assert np.all(w >= -1e-12)

    def test_already_on_simplex(self):
        v = np.array([0.2, 0.5, 0.3])
        w = project_simplex(v, z=1.0)
        assert np.allclose(w, v, atol=1e-7)


class TestConvexHullProjectionSolver:
    """Tests for ConvexHullProjectionSolver functionality and extrapolation distances."""

    def test_lipschitz_precomputation_and_step_size(self):
        np.random.seed(42)
        # Test D < N: _gram should be lazy
        X_train = np.random.randn(20, 4)
        solver = ConvexHullProjectionSolver(X_train, method="fista")
        assert solver._gram_cached is None
        expected_L = float(np.linalg.norm(X_train @ X_train.T, 2))
        assert np.isclose(solver.L, expected_L, rtol=1e-5)
        assert np.isclose(solver.step_size, 1.0 / expected_L, rtol=1e-5)
        assert solver._gram.shape == (20, 20)
        assert solver._gram_cached is not None

        # Test D >= N: _gram computed immediately
        X_train_wide = np.random.randn(5, 10)
        solver_wide = ConvexHullProjectionSolver(X_train_wide, method="fista")
        assert solver_wide._gram_cached is not None
        expected_L_wide = float(np.linalg.norm(X_train_wide @ X_train_wide.T, 2))
        assert np.isclose(solver_wide.L, expected_L_wide, rtol=1e-5)

    def test_direct_training_points(self):
        """Points in X_train must have d_norm < 1e-6 and is_interpolating == True."""
        np.random.seed(42)
        for D in [2, 5, 10]:
            N = 30
            X_train = np.random.randn(N, D)
            solver = ConvexHullProjectionSolver(X_train, method="fista")

            # Query a subset of training points
            queries = X_train[:10]
            result = solver.project(queries, max_iter=100)

            assert np.all(result["d_norm"] < 1e-6), f"Failed for D={D}: max d_norm={result['d_norm'].max()}"
            assert np.all(result["is_interpolating"]), f"Expected all training points to be interpolating for D={D}"
            assert np.all(result["d_raw"] < 1e-6)

    def test_interior_dirichlet_points(self):
        """Convex combinations of training points must lie inside hull with d_norm < 1e-6."""
        np.random.seed(123)
        for D in [2, 4, 8]:
            N = 25
            X_train = np.random.randn(N, D)
            solver = ConvexHullProjectionSolver(X_train, method="fista")

            # Generate random convex combinations
            weights_true = np.random.dirichlet(np.ones(N), size=15)
            X_interior = weights_true @ X_train

            result = solver.project(X_interior, max_iter=100)
            assert np.all(result["d_norm"] < 1e-6), f"Interior check failed for D={D}: max d_norm={result['d_norm'].max()}"
            assert np.all(result["is_interpolating"]), f"Expected is_interpolating == True for all interior points (D={D})"

    def test_exterior_points_exact_analytic_distances(self):
        """Exterior points shifted along coordinate axes must have exact analytic distances."""
        # Define a 2D square with vertices at (+-1, +-1)
        X_train = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ])
        solver = ConvexHullProjectionSolver(X_train, method="fista")

        delta = 2.5
        # Shift along positive x-axis
        x_exterior = np.array([[1.0 + delta, 0.0]])
        result = solver.project(x_exterior, max_iter=100)

        # Expected closest point is (1.0, 0.0)
        expected_x_proj = np.array([[1.0, 0.0]])
        expected_d_raw = delta
        expected_d_norm = delta / np.sqrt(2.0)
        expected_d_rel = delta / (2.0 * np.sqrt(2.0))
        expected_d_inf = delta

        assert np.allclose(result["x_proj"], expected_x_proj, atol=1e-5)
        assert np.isclose(result["d_raw"][0], expected_d_raw, atol=1e-5)
        assert np.isclose(result["d_norm"][0], expected_d_norm, atol=1e-5)
        assert np.isclose(result["d_rel"][0], expected_d_rel, atol=1e-5)
        assert np.isclose(result["d_inf"][0], expected_d_inf, atol=1e-5)
        assert not result["is_interpolating"][0]

    def test_1d_vs_2d_input_shapes(self):
        """1D input (D,) and 2D input (1, D) must yield identical metric results."""
        np.random.seed(99)
        N, D = 15, 6
        X_train = np.random.randn(N, D)
        solver = ConvexHullProjectionSolver(X_train)

        x_query_1d = np.random.randn(D) + 3.0
        x_query_2d = x_query_1d.reshape(1, D)

        res_1d = solver.project(x_query_1d)
        res_2d = solver.project(x_query_2d)

        # Output shapes
        assert res_1d["weights"].shape == (1, N)
        assert res_2d["weights"].shape == (1, N)
        assert res_1d["x_proj"].shape == (1, D)
        assert res_2d["x_proj"].shape == (1, D)
        assert res_1d["d_raw"].shape == (1,)
        assert res_1d["d_norm"].shape == (1,)
        assert res_1d["d_rel"].shape == (1,)
        assert res_1d["d_inf"].shape == (1,)
        assert res_1d["is_interpolating"].shape == (1,)

        # Metric values agreement
        assert np.allclose(res_1d["d_raw"], res_2d["d_raw"], atol=1e-7)
        assert np.allclose(res_1d["d_norm"], res_2d["d_norm"], atol=1e-7)
        assert np.allclose(res_1d["x_proj"], res_2d["x_proj"], atol=1e-7)
        assert np.array_equal(res_1d["is_interpolating"], res_2d["is_interpolating"])

        # Convenience distance method
        dist_1d = solver.distance(x_query_1d)
        dist_2d = solver.distance(x_query_2d)
        assert dist_1d.shape == (1,)
        assert np.allclose(dist_1d, dist_2d, atol=1e-7)
        assert np.allclose(dist_1d, res_1d["d_norm"], atol=1e-7)

    @pytest.mark.parametrize("D", [2, 8, 16, 32])
    def test_fista_slsqp_consistency_across_dimensions(self, D):
        """Consistency between FISTA and SLSQP methods across dimensions D in {2, 8, 16, 32}."""
        np.random.seed(42 + D)
        N = 20
        X_train = np.random.randn(N, D)

        solver_fista = ConvexHullProjectionSolver(X_train, method="fista")
        solver_slsqp = ConvexHullProjectionSolver(X_train, method="slsqp")

        # Mix of interior points and exterior points
        X_int = np.random.dirichlet(np.ones(N), size=3) @ X_train
        X_ext = np.random.randn(3, D) * 3.0 + 5.0
        X_test = np.vstack([X_int, X_ext])

        res_f = solver_fista.project(X_test, max_iter=300, tol=1e-7)
        res_s = solver_slsqp.project(X_test, max_iter=300, tol=1e-7)

        # Normalized distances should match closely
        diff_d_norm = np.abs(res_f["d_norm"] - res_s["d_norm"])
        assert np.all(diff_d_norm < 1e-3), (
            f"FISTA vs SLSQP d_norm mismatch at D={D}: max diff={np.max(diff_d_norm):.2e}"
        )

        # Interpolation flags should match
        assert np.array_equal(res_f["is_interpolating"], res_s["is_interpolating"])

    def test_edge_case_n_greater_than_d(self):
        """Edge case: N >> D (over-determined vertices)."""
        np.random.seed(11)
        N, D = 80, 2
        X_train = np.random.randn(N, D)
        solver = ConvexHullProjectionSolver(X_train)

        # Training point projection
        res_train = solver.project(X_train[:5])
        assert np.all(res_train["d_norm"] < 1e-6)
        assert np.all(res_train["is_interpolating"])

    def test_edge_case_n_less_than_d(self):
        """Edge case: N << D (under-determined, simplex embedded in higher dim)."""
        np.random.seed(22)
        N, D = 4, 16
        X_train = np.random.randn(N, D)
        solver = ConvexHullProjectionSolver(X_train)

        # True convex combination
        weights = np.array([[0.1, 0.4, 0.3, 0.2]])
        x_interior = weights @ X_train

        res = solver.project(x_interior)
        assert res["d_norm"][0] < 1e-6
        assert res["is_interpolating"][0]

    def test_edge_case_collinear_points(self):
        """Edge case: collinear points in 2D."""
        # 4 collinear points on the line y = 2x
        X_train = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ])
        solver = ConvexHullProjectionSolver(X_train)

        # Test point perpendicular to segment [0, (3,6)]
        # Midpoint of segment is (1.5, 3.0). Shift perpendicular vector (-2, 1) * dist
        # Length of (-2, 1) is sqrt(5)
        perp_unit = np.array([-2.0, 1.0]) / np.sqrt(5.0)
        target_dist = 2.0
        x_query = np.array([1.5, 3.0]) + perp_unit * target_dist

        res = solver.project(x_query, max_iter=200)
        assert np.isclose(res["d_raw"][0], target_dist, atol=1e-4)
        assert np.allclose(res["x_proj"][0], np.array([1.5, 3.0]), atol=1e-4)

    def test_edge_case_single_point(self):
        """Edge case: N = 1 (single training point)."""
        X_train = np.array([[2.0, -1.0, 3.0]])
        solver = ConvexHullProjectionSolver(X_train)

        x_query = np.array([[5.0, -1.0, 7.0]])
        res = solver.project(x_query)

        expected_d_raw = np.linalg.norm(x_query[0] - X_train[0])
        assert np.isclose(res["d_raw"][0], expected_d_raw, atol=1e-6)
        assert np.allclose(res["x_proj"], X_train, atol=1e-6)
        assert np.allclose(res["weights"], np.array([[1.0]]), atol=1e-6)

    def test_input_validation(self):
        """Ensure solver validates shapes and parameters properly."""
        # 1D X_train should raise ValueError
        with pytest.raises(ValueError, match="2D array"):
            ConvexHullProjectionSolver(np.array([1.0, 2.0, 3.0]))

        # Empty X_train should raise ValueError
        with pytest.raises(ValueError, match="empty|dimension"):
            ConvexHullProjectionSolver(np.empty((0, 3)))

        # Invalid method should raise ValueError
        with pytest.raises(ValueError, match="Unsupported method"):
            ConvexHullProjectionSolver(np.ones((5, 2)), method="invalid_method")

        solver = ConvexHullProjectionSolver(np.ones((5, 3)))
        # X_test dimension mismatch
        with pytest.raises(ValueError, match="dimension"):
            solver.project(np.ones((2, 4)))

        # 3D X_test should raise ValueError
        with pytest.raises(ValueError, match="1D .* or 2D"):
            solver.project(np.ones((2, 3, 2)))

        # Invalid method in project override
        with pytest.raises(ValueError, match="Unsupported projection method"):
            solver.project(np.ones((2, 3)), method="unknown_solver")

        # Invalid simplex scale z <= 0
        with pytest.raises(ValueError, match="strictly positive"):
            project_simplex(np.array([1.0, 2.0]), z=0.0)

    def test_projection_result_dict_and_attribute_access(self):
        """ProjectionResult must support both dict and attribute access."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        solver = ConvexHullProjectionSolver(X_train)
        res = solver.project(np.array([[0.2, 0.2]]))

        # Dict access
        assert "weights" in res
        assert "x_proj" in res
        assert "d_raw" in res
        assert "d_norm" in res
        assert "d_rel" in res
        assert "d_inf" in res
        assert "is_interpolating" in res

        # Attribute access
        assert np.array_equal(res.weights, res["weights"])
        assert np.array_equal(res.x_proj, res["x_proj"])
        assert np.array_equal(res.d_raw, res["d_raw"])
        assert np.array_equal(res.d_norm, res["d_norm"])
        assert np.array_equal(res.d_rel, res["d_rel"])
        assert np.array_equal(res.d_inf, res["d_inf"])
        assert np.array_equal(res.is_interpolating, res["is_interpolating"])

    def test_slsqp_direct_constructor_mode(self):
        """ConvexHullProjectionSolver constructed with method='slsqp' runs SLSQP by default."""
        X_train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        solver = ConvexHullProjectionSolver(X_train, method="slsqp")
        assert solver.method == "slsqp"

        res = solver.project(np.array([[2.0, 0.0]]))
        assert np.isclose(res.d_raw[0], 1.0, atol=1e-5)
        assert np.allclose(res.x_proj[0], np.array([1.0, 0.0]), atol=1e-5)

    def test_fista_speed_benchmark_d32(self):
        """Benchmark: solver executes queries in D=32 well under 5ms per query."""
        import time

        np.random.seed(42)
        N, D = 112, 32
        X_train = np.random.randn(N, D)
        solver = ConvexHullProjectionSolver(X_train, method="fista")

        M = 100
        X_test = np.random.randn(M, D)

        t0 = time.perf_counter()
        res = solver.project(X_test, max_iter=50)
        elapsed = time.perf_counter() - t0

        ms_per_query = (elapsed / M) * 1000.0
        assert ms_per_query < 5.0, f"Query took {ms_per_query:.2f}ms, expected < 5.0ms"
        assert res.weights.shape == (M, N)


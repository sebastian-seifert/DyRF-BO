"""Unit tests for subdomain training sampler and natural & stratified test samplers."""

import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.qp_convex_hull import (
    ConvexHullProjectionSolver,
    ProjectionResult,
)
from dyrf_bo.extrapolation_uq.samplers import (
    sample_natural_test,
    sample_stratified_test,
    sample_subdomain_training,
)


class TestSubdomainTrainingSampler:
    """Tests for sample_subdomain_training."""

    def test_output_shape_and_bounds_lhs(self):
        for D in [2, 4, 8]:
            N = 50
            half_width = 0.5
            X_train = sample_subdomain_training(
                n_samples=N,
                dimension=D,
                domain_half_width=half_width,
                seed=42,
                method="lhs",
            )
            assert X_train.shape == (N, D)
            assert np.all(X_train >= -half_width)
            assert np.all(X_train <= half_width)

    def test_custom_domain_half_width(self):
        half_width = 0.35
        X = sample_subdomain_training(
            n_samples=40,
            dimension=3,
            domain_half_width=half_width,
            seed=123,
            method="lhs",
        )
        assert np.all(X >= -half_width)
        assert np.all(X <= half_width)

    def test_uniform_random_method(self):
        X = sample_subdomain_training(
            n_samples=60,
            dimension=4,
            domain_half_width=0.6,
            seed=999,
            method="uniform",
        )
        assert X.shape == (60, 4)
        assert np.all(X >= -0.6)
        assert np.all(X <= 0.6)

    def test_seed_reproducibility(self):
        X1 = sample_subdomain_training(30, 3, seed=777)
        X2 = sample_subdomain_training(30, 3, seed=777)
        X3 = sample_subdomain_training(30, 3, seed=888)
        assert np.allclose(X1, X2)
        assert not np.allclose(X1, X3)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="n_samples"):
            sample_subdomain_training(0, 2)
        with pytest.raises(ValueError, match="dimension"):
            sample_subdomain_training(10, 0)
        with pytest.raises(ValueError, match="domain_half_width"):
            sample_subdomain_training(10, 2, domain_half_width=-0.1)
        with pytest.raises(ValueError, match="method"):
            sample_subdomain_training(10, 2, method="invalid_sampler")


class TestNaturalTestSampler:
    """Tests for sample_natural_test."""

    def test_output_types_shapes_and_domain(self):
        np.random.seed(42)
        D = 3
        N = 30
        X_train = sample_subdomain_training(N, D, domain_half_width=0.5, seed=42)
        solver = ConvexHullProjectionSolver(X_train)

        n_test = 100
        X_test, proj_res, p_interp = sample_natural_test(
            n_samples=n_test,
            dimension=D,
            solver=solver,
            seed=42,
        )

        assert X_test.shape == (n_test, D)
        assert np.all(X_test >= -1.0)
        assert np.all(X_test <= 1.0)
        assert isinstance(proj_res, ProjectionResult)
        assert proj_res.d_norm.shape == (n_test,)
        assert isinstance(p_interp, float)
        assert 0.0 <= p_interp <= 1.0
        assert np.isclose(p_interp, np.mean(proj_res.is_interpolating))

    def test_dimension_validation(self):
        X_train = np.random.randn(20, 3)
        solver = ConvexHullProjectionSolver(X_train)
        with pytest.raises(ValueError, match="dimension"):
            sample_natural_test(n_samples=10, dimension=4, solver=solver)

    def test_seed_reproducibility(self):
        X_train = np.random.randn(20, 2)
        solver = ConvexHullProjectionSolver(X_train)
        X1, _, p1 = sample_natural_test(50, 2, solver, seed=123)
        X2, _, p2 = sample_natural_test(50, 2, solver, seed=123)
        assert np.allclose(X1, X2)
        assert np.isclose(p1, p2)


class TestStratifiedTestSampler:
    """Tests for sample_stratified_test with 4 strata (Dirichlet & Ray-Casting)."""

    @pytest.mark.parametrize("D", [2, 4])
    def test_strata_quotas_and_exact_bands(self, D):
        n_per_stratum = 25
        N_train = 40
        X_train = sample_subdomain_training(N_train, D, domain_half_width=0.5, seed=100 + D)
        solver = ConvexHullProjectionSolver(X_train)

        X_test, proj_res, strata_labels = sample_stratified_test(
            n_per_stratum=n_per_stratum,
            dimension=D,
            X_train=X_train,
            solver=solver,
            seed=42 + D,
        )

        total_samples = 4 * n_per_stratum
        assert X_test.shape == (total_samples, D)
        assert strata_labels.shape == (total_samples,)
        assert isinstance(proj_res, ProjectionResult)
        assert proj_res.d_norm.shape == (total_samples,)

        # Domain boundary check [-1.0, 1.0]^D
        assert np.all(X_test >= -1.0 - 1e-12)
        assert np.all(X_test <= 1.0 + 1e-12)

        # Quota checks: exactly n_per_stratum for each stratum
        for s in [0, 1, 2, 3]:
            count = np.sum(strata_labels == s)
            assert count == n_per_stratum, f"Stratum {s} count {count} != {n_per_stratum}"

        # Stratum 0 (Interpolation: Dirichlet combinations): d_norm < 1e-6
        d_norm_0 = proj_res.d_norm[strata_labels == 0]
        assert np.all(d_norm_0 < 1e-6)
        assert np.all(proj_res.is_interpolating[strata_labels == 0])

        # Stratum 1 (Near Extrapolation): 1e-6 <= d_norm <= 0.15
        d_norm_1 = proj_res.d_norm[strata_labels == 1]
        assert np.all(d_norm_1 >= 1e-6)
        assert np.all(d_norm_1 <= 0.15 + 1e-6)

        # Stratum 2 (Medium Extrapolation): 0.15 < d_norm <= 0.40
        d_norm_2 = proj_res.d_norm[strata_labels == 2]
        assert np.all(d_norm_2 > 0.15 - 1e-6)
        assert np.all(d_norm_2 <= 0.40 + 1e-6)

        # Stratum 3 (Far Extrapolation): 0.40 < d_norm <= 0.80
        d_norm_3 = proj_res.d_norm[strata_labels == 3]
        assert np.all(d_norm_3 > 0.40 - 1e-6)
        assert np.all(d_norm_3 <= 0.80 + 1e-6)

    def test_stratified_seed_reproducibility(self):
        D = 3
        X_train = sample_subdomain_training(30, D, seed=42)
        solver = ConvexHullProjectionSolver(X_train)

        X1, _, l1 = sample_stratified_test(15, D, X_train, solver, seed=999)
        X2, _, l2 = sample_stratified_test(15, D, X_train, solver, seed=999)
        assert np.allclose(X1, X2)
        assert np.array_equal(l1, l2)

    def test_dimension_and_input_validations(self):
        X_train = np.random.randn(20, 3)
        solver = ConvexHullProjectionSolver(X_train)

        # Mismatched dimension in arguments
        with pytest.raises(ValueError, match="dimension"):
            sample_stratified_test(10, 4, X_train, solver)

        # Non-positive quota
        with pytest.raises(ValueError, match="n_per_stratum"):
            sample_stratified_test(0, 3, X_train, solver)

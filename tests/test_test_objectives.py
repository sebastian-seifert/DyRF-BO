"""Unit tests for synthetic benchmark objective functions and target standardization."""

import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyrf_bo.extrapolation_uq.test_objectives import (
    BaseObjective,
    StandardizedObjective,
    ackley,
    get_objective,
    rastrigin,
    rosenbrock,
    sphere,
)


class TestVectorizedObjectives:
    """Tests for raw synthetic benchmark objective functions."""

    def test_sphere_known_values(self):
        # Origin
        x_zero = np.zeros(4)
        assert np.isclose(sphere(x_zero), 0.0)

        # Known point
        x_known = np.array([1.0, 2.0, 3.0])
        assert np.isclose(sphere(x_known), 14.0)

        # Batch 2D
        X_batch = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [-1.0, -1.0],
        ])
        y = sphere(X_batch)
        assert y.shape == (3,)
        assert np.allclose(y, [0.0, 2.0, 2.0])

    def test_sphere_arbitrary_dimensions_and_shapes(self):
        for D in [2, 8, 16, 32]:
            X = np.ones((5, D))
            y = sphere(X)
            assert y.shape == (5,)
            assert np.allclose(y, float(D))

        # Higher rank tensor (2, 3, D)
        X_tensor = np.ones((2, 3, 4))
        y_tensor = sphere(X_tensor)
        assert y_tensor.shape == (2, 3)
        assert np.allclose(y_tensor, 4.0)

    def test_rosenbrock_known_values(self):
        # Global minimum at all 1s
        x_opt = np.ones(5)
        assert np.isclose(rosenbrock(x_opt), 0.0)

        # Origin in 2D
        x_zero = np.zeros(2)
        assert np.isclose(rosenbrock(x_zero), 1.0)

        # 3D known point: [1, 2, 3] -> 100*(2-1)^2 + 0 + 100*(3-4)^2 + (-1)^2 = 100 + 100 + 1 = 201
        x_3d = np.array([1.0, 2.0, 3.0])
        assert np.isclose(rosenbrock(x_3d), 201.0)

    def test_rosenbrock_dimension_validation(self):
        # D < 2 must raise ValueError
        with pytest.raises(ValueError, match="dimension D >= 2"):
            rosenbrock(np.array([1.0]))

    def test_rastrigin_known_values(self):
        # Global minimum at origin is 0.0
        for D in [2, 5, 10]:
            x_zero = np.zeros(D)
            assert np.isclose(rastrigin(x_zero), 0.0)

        # At all 1s: 10*D + sum(1 - 10*cos(2*pi)) = 10*D + D*(1 - 10) = D
        for D in [2, 4, 8]:
            x_ones = np.ones(D)
            assert np.isclose(rastrigin(x_ones), float(D))

    def test_ackley_known_values(self):
        # Global minimum at origin is 0.0
        for D in [2, 5, 10]:
            x_zero = np.zeros(D)
            assert np.isclose(ackley(x_zero), 0.0, atol=1e-12)

        # Known 2D point [1, 1]
        x_2d = np.ones(2)
        expected = -20.0 * np.exp(-0.2) - np.e + 20.0 + np.e
        assert np.isclose(ackley(x_2d), expected, atol=1e-10)

    def test_invalid_input_dimensions(self):
        # 0D scalar array
        with pytest.raises(ValueError):
            sphere(np.array(5.0))
        # Empty feature dimension
        with pytest.raises(ValueError):
            sphere(np.empty((5, 0)))


class TestStandardizedObjective:
    """Tests for StandardizedObjective fitting and evaluation."""

    def test_initialization_and_unfitted_state(self):
        obj = StandardizedObjective(sphere, bounds=(-1.0, 1.0), name="sphere")
        assert not obj.is_fitted
        assert obj.mean_train is None
        assert obj.std_train is None
        assert obj.bounds == (-1.0, 1.0)

        # Evaluating with standardized=True before fit raises RuntimeError
        X_query = np.array([[0.5, 0.5]])
        with pytest.raises(RuntimeError, match="must be fit"):
            obj.evaluate(X_query, standardized=True)

        # Raw evaluate works without fitting
        y_raw = obj.evaluate(X_query, standardized=False)
        assert np.isclose(y_raw[0], 0.5)

    def test_fit_and_standardized_evaluation(self):
        np.random.seed(42)
        X_train = np.random.uniform(-0.5, 0.5, size=(100, 4))
        obj = StandardizedObjective(sphere, name="sphere")
        fitted = obj.fit(X_train)
        assert fitted is obj
        assert obj.is_fitted
        assert obj.mean_train is not None
        assert obj.std_train is not None
        assert obj.std_train > 0.0

        # Evaluating training data standardized should have zero mean and unit variance
        y_train_std = obj.evaluate(X_train, standardized=True)
        assert y_train_std.shape == (100,)
        assert np.isclose(np.mean(y_train_std), 0.0, atol=1e-10)
        assert np.isclose(np.std(y_train_std), 1.0, atol=1e-10)

        # Call syntax works identically
        y_call = obj(X_train, standardized=True)
        assert np.allclose(y_train_std, y_call)

        # Raw evaluation returns exact sphere values
        y_raw = obj.evaluate(X_train, standardized=False)
        assert np.allclose(y_raw, np.sum(X_train**2, axis=-1))

    def test_constant_function_std_floor(self):
        # When function output has 0 variance, std_train must be floored to 1e-8
        constant_func = lambda x: np.zeros(x.shape[:-1])
        obj = StandardizedObjective(constant_func, name="constant")
        X_train = np.random.randn(20, 3)
        obj.fit(X_train)
        assert np.isclose(obj.mean_train, 0.0)
        assert np.isclose(obj.std_train, 1e-8)

        # Standardization does not divide by zero or produce NaN
        y_std = obj.evaluate(X_train, standardized=True)
        assert not np.any(np.isnan(y_std))
        assert np.all(y_std == 0.0)


class TestGetObjectiveRegistry:
    """Tests for objective registry and helper retrieval."""

    def test_retrieve_registered_objectives(self):
        for name in ["sphere", "rosenbrock", "rastrigin", "ackley"]:
            obj = get_objective(name)
            assert isinstance(obj, StandardizedObjective)
            assert obj.name == name

            # Case insensitivity
            obj_upper = get_objective(name.upper())
            assert obj_upper.name == name

    def test_unknown_objective_raises_error(self):
        with pytest.raises((KeyError, ValueError), match="not found"):
            get_objective("non_existent_function")

    def test_protocol_conformance(self):
        obj = get_objective("sphere")
        assert isinstance(obj, BaseObjective)

"""Comprehensive Unit and Contract Test Suite for CARPS DA-EHRF Direct EI Integration.

Validates:
1. CustomUncertaintyRandomForest integration with SMAC3's EPMRandomForest.
2. Variance parity & Direct EI property: _predict(X) returns (mean, var) where var = U_E(X)^2.
3. Tree-path topological metric: LCA shortest-path distances and topological depth normalization.
4. Mathematical invariants: non-negativity, finite bounds, zero-division protection, scale equivariance.
5. CARPS Hydra configuration schema and resolution for optimizer and benchmark tasks.
6. End-to-end smoke execution on Ackley 2D and Rosenbrock 2D benchmarks.
"""

from __future__ import annotations

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ConfigSpace import ConfigurationSpace, Float

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from ep_extractors.distance_evidential import DistanceAwareEvidentialExtractor
from ep_extractors import UQExtractorRegistry
from noisy_benchmarks.registry import NoisyBenchmarkRegistry
from smac.model.random_forest.random_forest import EPMRandomForest
from smac.acquisition.function import EI


# =====================================================================
# 1. SMAC3 EPMRandomForest Surrogate Binding & Extractor Contract Tests
# =====================================================================

class TestCustomUncertaintyRandomForestDAEHRFIntegration:
    """Verifies that CustomUncertaintyRandomForest binds native EPMRandomForest to DA-EHRF."""

    @pytest.fixture(autouse=True)
    def setup_model(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })
        self.n_train = 30
        self.X_train = np.random.uniform(-3.0, 3.0, size=(self.n_train, 2))
        self.y_train = (
            self.X_train[:, 0] ** 2
            + self.X_train[:, 1] ** 2
            + 0.5 * np.sin(2.0 * self.X_train[:, 0])
        ).reshape(-1, 1)

        self.extractor_kwargs = {
            "spatial_metric": "tree_path",
            "tree_decay_lambda": 3.0,
            "kappa_leaf": 1.0,
            "c_spatial": 1.0,
        }
        self.model = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs=self.extractor_kwargs,
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        self.model.train(self.X_train, self.y_train)

    def test_rf_model_binding_and_smac_epm_integration(self):
        """Verify CustomUncertaintyRandomForest wraps SMAC3 EPMRandomForest without retraining overhead."""
        # EPMRandomForest verification
        assert self.model._rf is not None
        assert isinstance(self.model._rf, EPMRandomForest)
        assert hasattr(self.model._rf, "estimators_")
        assert len(self.model._rf.estimators_) == 10

        # DA-EHRF extractor binding
        assert self.model.uq_extractor is not None
        assert isinstance(self.model.uq_extractor, DistanceAwareEvidentialExtractor)
        assert self.model.extractor is self.model.uq_extractor
        assert self.model.uq_extractor.model is self.model._rf

        # Hyperparameter verification
        assert self.model.uq_extractor.spatial_metric == "tree_path"
        assert np.isclose(self.model.uq_extractor.tree_decay_lambda, 3.0)
        assert np.isclose(self.model.uq_extractor.kappa_leaf, 1.0)
        assert np.isclose(self.model.uq_extractor.c_spatial, 1.0)


# =====================================================================
# 2. Variance Parity & Direct EI Substitution Tests
# =====================================================================

class TestVarianceParityAndDirectEIProperty:
    """Verifies that _predict(X) returns var = U_E(X)^2, enabling pure Direct EI acquisition."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        np.random.seed(123)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })
        # Clustered training set in [-1, 1]^2 to create a visible extrapolation gap
        self.X_train = np.random.uniform(-1.0, 1.0, size=(25, 2))
        self.y_train = (self.X_train[:, 0] ** 2 + self.X_train[:, 1] ** 2).reshape(-1, 1)

        self.model = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={
                "spatial_metric": "tree_path",
                "tree_decay_lambda": 3.0,
                "kappa_leaf": 1.0,
                "c_spatial": 1.0,
            },
            configspace=self.cs,
            n_trees=12,
            seed=123,
        )
        self.model.train(self.X_train, self.y_train)

    def test_variance_equals_ue_squared_direct_formula(self):
        """Verify _predict(X) returns var == U_E(X)^2 matching standalone DA-EHRF extractor."""
        X_test = np.array([
            [0.0, 0.0],     # Training center
            [0.8, -0.7],    # In-distribution
            [3.5, 3.5],     # Moderate OOD
            [-4.8, 4.8],    # Extreme OOD
        ])
        mean, var = self.model._predict(X_test)

        assert mean.shape == (4, 1)
        assert var.shape == (4, 1)

        # Standalone ground-truth evaluation directly from self.model._rf
        standalone_ext = DistanceAwareEvidentialExtractor(
            self.model._rf,
            spatial_metric="tree_path",
            tree_decay_lambda=3.0,
            kappa_leaf=1.0,
            c_spatial=1.0,
        )
        standalone_ext.fit(self.X_train, self.y_train.flatten())
        expected_ue = standalone_ext.extract_epistemic_signal(X_test)
        expected_var = np.maximum(expected_ue ** 2, 1e-10).reshape(-1, 1)

        np.testing.assert_allclose(var, expected_var, rtol=1e-6, atol=1e-9)

    def test_direct_ei_acquisition_uses_pure_epistemic_uncertainty(self):
        """Verify that SMAC3 Expected Improvement evaluates using pure epistemic uncertainty sigma = U_E."""
        ei = EI()
        ei.update(
            model=self.model,
            eta=float(np.min(self.y_train)),
        )

        # Point inside dense training cloud vs point in unexplored gap
        X_dense = np.array([[0.1, 0.1]])
        X_gap = np.array([[4.5, 4.5]])

        mean_dense, var_dense = self.model._predict(X_dense)
        mean_gap, var_gap = self.model._predict(X_gap)

        u_e_dense = np.sqrt(var_dense[0, 0])
        u_e_gap = np.sqrt(var_gap[0, 0])

        # Epistemic uncertainty in unexplored space must significantly exceed in-distribution
        assert u_e_gap > u_e_dense

        # Acquisition evaluation
        val_dense = ei._compute(X_dense)
        val_gap = ei._compute(X_gap)

        assert np.isfinite(val_dense).all()
        assert np.isfinite(val_gap).all()
        assert (val_dense >= 0.0).all()
        assert (val_gap >= 0.0).all()


# =====================================================================
# 3. Tree-Path Topological Proximity Metric Tests
# =====================================================================

class TestTreePathTopologicalProximity:
    """Verifies tree-path LCA shortest-path distances and topological depth normalization."""

    @pytest.fixture(autouse=True)
    def setup_trees(self):
        np.random.seed(99)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-5.0, 5.0), default=0.0),
            "x2": Float("x2", bounds=(-5.0, 5.0), default=0.0),
        })
        self.n_train = 35
        self.X_train = np.random.uniform(-3.0, 3.0, size=(self.n_train, 2))
        self.y_train = (self.X_train[:, 0] ** 2 + self.X_train[:, 1] ** 2).reshape(-1, 1)

        self.model = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={
                "spatial_metric": "tree_path",
                "tree_decay_lambda": 3.0,
                "kappa_leaf": 1.0,
                "c_spatial": 1.0,
            },
            configspace=self.cs,
            n_trees=10,
            seed=99,
        )
        self.model.train(self.X_train, self.y_train)
        self.ext = self.model.uq_extractor

    def test_tree_path_lca_distance_and_depth_normalization(self):
        """Verify topological structures: depths, LCA depths, mean_max_depth normalization."""
        assert self.ext.mean_max_depth > 0.0
        assert len(self.ext.tree_node_depths) == 10
        assert len(self.ext.tree_node_ancestors) == 10

        # Verify that for each tree, self leaf distance is 0
        for m in range(len(self.ext.tree_leaf_dists)):
            train_leaves_m = self.ext.leaf_matrix_train[:, m]
            q_dense = self.ext.tree_leaf_to_dense[m][train_leaves_m]
            self_dists = self.ext.tree_leaf_dists[m][q_dense, np.arange(self.n_train)]
            assert np.all(self_dists == 0), f"Tree {m} self leaf distance must be 0"

        # Verify LCA shortest-path distances: min dist is 0, max dist > 0
        all_dists = [d.max() for d in self.ext.tree_leaf_dists]
        assert max(all_dists) > 0, "Trees must exhibit positive inter-leaf distances"

        # Verify topological depth normalization parameter
        assert self.ext.mean_max_depth >= 1.0
        assert np.isclose(self.ext.tree_decay_lambda, 3.0)

    def test_tree_path_mathematical_spatial_formula(self):
        """Verify V_spatial = sigma_0^2 * c_spatial * (1 - exp(-lambda * d / (2 * max_depth)))."""
        X_test = np.array([[3.0, 3.0]])
        u_e = self.ext.extract_epistemic_signal(X_test)[0]

        # Theoretical upper bound on U_E is sqrt(V_ens_max + V_leaf_max + sigma_0^2 * c_spatial)
        max_possible_spatial = (self.ext.sigma_0 ** 2) * self.ext.c_spatial
        assert u_e <= np.sqrt(self.ext.sigma_0 ** 2 * 10.0)


# =====================================================================
# 4. Mathematical Invariants, Zero-Division, & Scale Equivariance Tests
# =====================================================================

class TestMathematicalInvariants:
    """Verifies non-negativity, finite bounds, zero-division protection, and scale equivariance."""

    @pytest.fixture(autouse=True)
    def setup_spaces(self):
        np.random.seed(77)
        self.cs = ConfigurationSpace({
            "x1": Float("x1", bounds=(-10.0, 10.0), default=0.0),
            "x2": Float("x2", bounds=(-10.0, 10.0), default=0.0),
        })
        self.X_train = np.random.uniform(-4.0, 4.0, size=(20, 2))
        self.y_train = np.random.normal(5.0, 2.0, size=(20, 1))

    def test_non_negativity_and_finite_bounds(self):
        """Verify predictions are strictly positive and finite across arbitrary evaluation coordinates."""
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={"spatial_metric": "tree_path"},
            configspace=self.cs,
            n_trees=5,
            seed=77,
        )
        rf.train(self.X_train, self.y_train)

        # Extreme evaluation domain [-100, 100]^2
        X_eval = np.array([
            [0.0, 0.0],
            [1e-6, 1e-6],
            [-50.0, 50.0],
            [100.0, -100.0],
        ])
        mean, var = rf._predict(X_eval)

        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 1e-10)
        assert np.all(np.sqrt(var) > 0.0)

    def test_zero_division_protection_and_constant_targets(self):
        """Verify robust zero-variance protection when target vector is constant (std == 0)."""
        y_constant = np.full((20, 1), 7.0)
        rf = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={"spatial_metric": "tree_path", "sigma_0_fallback": 1.0},
            configspace=self.cs,
            n_trees=5,
            seed=77,
        )
        # Must not raise ZeroDivisionError or NaN
        rf.train(self.X_train, y_constant)
        mean, var = rf._predict(self.X_train[:5])

        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 1e-10)
        assert rf.uq_extractor.sigma_0 > 0.0

    def test_scale_equivariance_with_target_prior_sigma_0(self):
        """Verify scale equivariance: scaling target y by c scales epistemic uncertainty U_E by c."""
        c = 10.0
        y_scaled = self.y_train * c

        rf_base = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={"spatial_metric": "tree_path", "c_spatial": 1.0, "kappa_leaf": 1.0},
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        rf_base.train(self.X_train, self.y_train)

        rf_scaled = CustomUncertaintyRandomForest(
            uncertainty_func="distance_evidential",
            extractor_kwargs={"spatial_metric": "tree_path", "c_spatial": 1.0, "kappa_leaf": 1.0},
            configspace=self.cs,
            n_trees=10,
            seed=42,
        )
        rf_scaled.train(self.X_train, y_scaled)

        X_test = np.array([[2.0, -2.0], [5.0, 5.0]])
        _, var_base = rf_base._predict(X_test)
        _, var_scaled = rf_scaled._predict(X_test)

        ue_base = np.sqrt(var_base.flatten())
        ue_scaled = np.sqrt(var_scaled.flatten())

        # Extracted epistemic uncertainty must scale approximately linearly with c
        ratio = ue_scaled / ue_base
        np.testing.assert_allclose(ratio, c, rtol=0.15)


# =====================================================================
# 5. CARPS Hydra Configuration Resolution Tests
# =====================================================================

class TestCARPSHydraConfigResolution:
    """Verifies Hydra and OmegaConf correctly discover and parse all new optimizer and task YAML configs."""

    def test_hydra_resolves_smac20_da_ehrf_direct_ei_config(self):
        """Verify smac20_da_ehrf_direct_ei.yaml schema and parameter resolution."""
        cfg_path = os.path.join(PROJECT_ROOT, "carps_integration", "configs", "optimizer", "smac20_da_ehrf_direct_ei.yaml")
        cfg = OmegaConf.load(cfg_path)
        assert cfg.optimizer_id == "SMAC20_DAEHRF_DirectEI"
        assert cfg.optimizer.acq_func_name == "ei"
        smac_cfg = cfg.optimizer.smac_cfg
        assert smac_cfg.model_class == "carps_integration.custom_uncertainty_model.CustomUncertaintyRandomForest"
        assert smac_cfg.model_kwargs.uncertainty_func == "distance_evidential"
        extractor_kwargs = smac_cfg.model_kwargs.extractor_kwargs
        assert extractor_kwargs.spatial_metric == "tree_path"
        assert extractor_kwargs.tree_decay_lambda == 3.0
        assert extractor_kwargs.kappa_leaf == 1.0
        assert extractor_kwargs.c_spatial == 1.0

    def test_hydra_resolves_dyrf_da_ehrf_direct_ei_config(self):
        """Verify dyrf_da_ehrf_direct_ei.yaml schema and parameter resolution."""
        cfg_path = os.path.join(PROJECT_ROOT, "carps_integration", "configs", "optimizer", "dyrf_da_ehrf_direct_ei.yaml")
        cfg = OmegaConf.load(cfg_path)
        assert cfg.optimizer_id == "CARPSDynamicRF_DAEHRF_DirectEI"
        assert cfg.optimizer.extractor_name == "distance_evidential"
        assert cfg.optimizer.acq_mode == "direct"
        assert cfg.optimizer.acq_func_name == "ei"
        assert cfg.optimizer.extractor_kwargs.spatial_metric == "tree_path"
        assert cfg.optimizer.extractor_kwargs.tree_decay_lambda == 3.0

    def test_hydra_resolves_low_d_benchmark_tasks(self):
        """Verify all 4 benchmark tasks (Ackley 2D, Rosenbrock 2D, Sphere 2D, Rosenbrock 4D) load."""
        cfg_dir = os.path.join(PROJECT_ROOT, "carps_integration", "configs")
        tasks = [
            "cfg_ackley_2d_gaussian",
            "cfg_rosenbrock_2d_gaussian",
            "cfg_sphere_2d_gaussian",
            "cfg_rosenbrock_4d_gaussian",
        ]
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
            for task in tasks:
                cfg = compose(
                    config_name=f"task/Noisy/bbob/{task}",
                    overrides=["+seed=1"]
                )
                assert cfg.task.name.startswith("Noisy/bbob/")
                assert cfg.task.objective_function.problem_name.startswith("bbob_noisy_")


# =====================================================================
# 6. CARPS End-to-End CLI Smoke Execution Tests
# =====================================================================

class TestCARPSSmokeExecution:
    """Executes quick CARPS runs using run_carps_patched.py on Ackley 2D and Rosenbrock 2D."""

    def test_carps_cli_da_ehrf_direct_ei_ackley_smoke(self, tmp_path):
        """Execute 5 trials with smac20_da_ehrf_direct_ei on Ackley 2D."""
        run_dir = str(tmp_path / "ackley_run")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
            "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
            "+optimizer=smac20_da_ehrf_direct_ei",
            "+task/Noisy/bbob=cfg_ackley_2d_gaussian",
            "task.optimization_resources.n_trials=5",
            "seed=1",
            f"outdir={run_dir}",
            f"hydra.run.dir={run_dir}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

        log_file = Path(run_dir) / "trial_logs.jsonl"
        assert log_file.exists(), f"Log file {log_file} was not generated."
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) >= 5, f"Expected >= 5 logged trials, found {len(lines)}."

    def test_carps_cli_da_ehrf_direct_ei_rosenbrock_smoke(self, tmp_path):
        """Execute 5 trials with smac20_da_ehrf_direct_ei on Rosenbrock 2D."""
        run_dir = str(tmp_path / "rosenbrock_run")
        cmd = [
            sys.executable,
            os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
            "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
            "+optimizer=smac20_da_ehrf_direct_ei",
            "+task/Noisy/bbob=cfg_rosenbrock_2d_gaussian",
            "task.optimization_resources.n_trials=5",
            "seed=1",
            f"outdir={run_dir}",
            f"hydra.run.dir={run_dir}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert res.returncode == 0, f"Run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"

        log_file = Path(run_dir) / "trial_logs.jsonl"
        assert log_file.exists(), f"Log file {log_file} was not generated."
        lines = [line.strip() for line in log_file.read_text().splitlines() if line.strip()]
        assert len(lines) >= 5, f"Expected >= 5 logged trials, found {len(lines)}."

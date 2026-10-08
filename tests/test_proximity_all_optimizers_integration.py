from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import yaml
from ConfigSpace import ConfigurationSpace, Float
from smac.scenario import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade

from carps_integration.custom_uncertainty_model import CustomUncertaintyRandomForest
from carps_integration.proximity_lcb import ProximityLowerBoundAcquisition
import scripts.run_carps_patched
from carps.optimizers.smac20 import SMAC3Optimizer
from omegaconf import OmegaConf


CONFIGS_DIR = Path("carps_integration/configs/optimizer")

ALL_8_CONFIG_SPECS = {
    "A": {
        "file": "smac20_proximity_lcb.yaml",
        "optimizer_id": "SMAC20_ProximityLCB",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_b",
        "method": None,
        "is_continuous": True,
        "k": 25,
        "k_warmup": 25,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"decay_lambda": 1.345},
    },
    "A_CV": {
        "file": "smac20_proximity_lcb_cv.yaml",
        "optimizer_id": "SMAC20_ProximityLCB_CV",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "standard_proximity_cv",
        "method": None,
        "is_continuous": False,
        "k": 25,
        "k_warmup": 25,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": None,
    },
    "B": {
        "file": "smac20_proximity_b_lcb.yaml",
        "optimizer_id": "SMAC20_ProximityB_LCB",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_b",
        "method": "b",
        "is_continuous": True,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"decay_lambda": 1.345},
    },
    "B_CV": {
        "file": "smac20_proximity_b_lcb_cv.yaml",
        "optimizer_id": "SMAC20_ProximityB_LCB_CV",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_b_cv",
        "method": "b",
        "is_continuous": True,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"decay_lambda": 1.345},
    },
    "AC": {
        "file": "smac20_proximity_ac_lcb.yaml",
        "optimizer_id": "SMAC20_ProximityAC_LCB",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_ac",
        "method": "ac",
        "is_continuous": False,
        "k": 25,
        "k_warmup": 25,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"alpha": 1.0},
    },
    "AC_CV": {
        "file": "smac20_proximity_ac_lcb_cv.yaml",
        "optimizer_id": "SMAC20_ProximityAC_LCB_CV",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_ac_cv",
        "method": "ac",
        "is_continuous": False,
        "k": 25,
        "k_warmup": 25,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"alpha": 1.0},
    },
    "BC": {
        "file": "smac20_proximity_bc_lcb.yaml",
        "optimizer_id": "SMAC20_ProximityBC_LCB",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_bc",
        "method": "bc",
        "is_continuous": True,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"decay_lambda": 1.345, "alpha": 1.0},
    },
    "BC_CV": {
        "file": "smac20_proximity_bc_lcb_cv.yaml",
        "optimizer_id": "SMAC20_ProximityBC_LCB_CV",
        "acq_func_name": "proximity_lcb",
        "uncertainty_func": "proximity_bc_cv",
        "method": "bc",
        "is_continuous": True,
        "eps": 0.16,
        "level": 0.95,
        "extractor_kwargs": {"decay_lambda": 1.345, "alpha": 1.0},
    },
}


class MockSurrogateWithTracking:
    """Mock surrogate tracking calls to predict_standard_rf vs predict_with_intervals."""

    def __init__(self, n_train: int = 5, uncertainty_func: str = "proximity_b"):
        self.n_train = n_train
        self.last_X = np.zeros((n_train, 2))
        self.uncertainty_func = uncertainty_func
        self.standard_rf_calls = 0
        self.predict_with_intervals_calls = 0
        self.last_intervals_kwargs = {}

    def predict_standard_rf(self, X: np.ndarray):
        self.standard_rf_calls += 1
        n = len(X)
        return np.full(n, 2.0), np.full(n, 1.0)

    def predict_with_intervals(self, X: np.ndarray, **kwargs):
        self.predict_with_intervals_calls += 1
        self.last_intervals_kwargs = kwargs
        n = len(X)
        return np.full(n, 1.5), np.full(n, 2.0), np.full(n, 2.5), np.full(n, 0.5)


class TestProximityAllOptimizersIntegration(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.cs = ConfigurationSpace(seed=42)
        self.cs.add([
            Float("x0", (-2.0, 2.0), default=0.0),
            Float("x1", (-2.0, 2.0), default=0.0),
        ])

    def test_all_8_yaml_configs_syntax_and_schema(self):
        """
        Validate that all 8 YAML optimizer configurations exist, parse cleanly,
        and conform to the CARP-S optimizer schema.
        """
        self.assertEqual(len(ALL_8_CONFIG_SPECS), 8)

        for cfg_key, spec in ALL_8_CONFIG_SPECS.items():
            with self.subTest(config=cfg_key):
                file_path = CONFIGS_DIR / spec["file"]
                self.assertTrue(file_path.exists(), f"Configuration file missing: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                # 1. Root & Defaults validation
                self.assertIn("defaults", cfg, f"Missing defaults in {spec['file']}")
                self.assertIn("/optimizer/smac20/base", cfg["defaults"])
                self.assertEqual(cfg["optimizer_id"], spec["optimizer_id"])
                self.assertEqual(cfg["optimizer_container_id"], spec["optimizer_id"])

                # 2. Optimizer section
                self.assertIn("optimizer", cfg)
                opt = cfg["optimizer"]
                self.assertEqual(opt["acq_func_name"], spec["acq_func_name"])

                # 3. Acquisition function kwargs
                acq_kwargs = opt["acq_func_kwargs"]
                self.assertAlmostEqual(acq_kwargs["eps"], spec["eps"], places=4)
                self.assertAlmostEqual(acq_kwargs["level"], spec["level"], places=4)
                if spec["method"] is not None:
                    self.assertEqual(acq_kwargs["method"], spec["method"])
                if "k" in spec:
                    self.assertEqual(acq_kwargs["k"], spec["k"])
                    self.assertEqual(acq_kwargs["k_warmup"], spec["k_warmup"])

                # 4. SMAC configuration section
                smac_cfg = opt["smac_cfg"]
                self.assertEqual(smac_cfg["smac_class"], "smac.facade.HyperparameterOptimizationFacade")
                self.assertEqual(
                    smac_cfg["model_class"],
                    "carps_integration.custom_uncertainty_model.CustomUncertaintyRandomForest",
                )

                # 5. Model kwargs
                model_kwargs = smac_cfg["model_kwargs"]
                self.assertEqual(model_kwargs["uncertainty_func"], spec["uncertainty_func"])

                # 6. Extractor kwargs
                if spec["extractor_kwargs"] is not None:
                    self.assertIn("extractor_kwargs", model_kwargs)
                    ext_kwargs = model_kwargs["extractor_kwargs"]
                    for k_exp, v_exp in spec["extractor_kwargs"].items():
                        self.assertIn(k_exp, ext_kwargs)
                        self.assertAlmostEqual(ext_kwargs[k_exp], v_exp, places=4)

    def test_closed_loop_bo_stepping_all_8_methods(self):
        """
        Execute real closed-loop Bayesian Optimization loop across all 8 configurations
        on a synthetic 2D objective function. Verifies that optimization completes without
        exceptions, NaNs, or crashes.
        """
        def objective(config, seed=0) -> float:
            x0 = float(config["x0"])
            x1 = float(config["x1"])
            return float((x0 - 0.2) ** 2 + (x1 + 0.3) ** 2 + 0.1 * np.sin(5.0 * x0))

        for cfg_key, spec in ALL_8_CONFIG_SPECS.items():
            with self.subTest(config=cfg_key):
                file_path = CONFIGS_DIR / spec["file"]
                self.assertTrue(file_path.exists(), f"Configuration file missing: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                opt_section = cfg["optimizer"]
                acq_kwargs = dict(opt_section["acq_func_kwargs"])
                smac_cfg_dict = opt_section["smac_cfg"]
                model_kwargs = dict(smac_cfg_dict["model_kwargs"])

                with tempfile.TemporaryDirectory() as td:
                    scenario = Scenario(
                        configspace=self.cs,
                        n_trials=5,
                        seed=42,
                        output_directory=Path(td),
                    )

                    model = CustomUncertaintyRandomForest(
                        configspace=self.cs,
                        seed=scenario.seed,
                        n_trees=10,
                        uncertainty_func=model_kwargs["uncertainty_func"],
                        extractor_kwargs=model_kwargs.get("extractor_kwargs", None),
                    )

                    acq_func = ProximityLowerBoundAcquisition(**acq_kwargs)

                    smac = HPOFacade(
                        scenario=scenario,
                        target_function=objective,
                        model=model,
                        acquisition_function=acq_func,
                        overwrite=True,
                    )

                    incumbent = smac.optimize()

                    # Assertions on incumbent and runhistory
                    self.assertIsNotNone(incumbent, f"Incumbent should not be None for {cfg_key}")
                    self.assertGreaterEqual(len(smac.runhistory), 5, f"At least 5 trials run for {cfg_key}")
                    costs = [tv.cost for tv in smac.runhistory.values()]
                    self.assertTrue(
                        all(np.isfinite(c) for c in costs),
                        f"All evaluated costs must be finite for {cfg_key}, got {costs}",
                    )
                    self.assertFalse(any(np.isnan(c) for c in costs), f"No NaN costs for {cfg_key}")

    def test_run_carps_patched_integration_all_8_methods(self):
        """
        Verify that scripts.run_carps_patched.patched_smac3_setup_optimizer instantiates
        SMAC3Optimizer cleanly for all 8 configurations, passing method from acq_func_kwargs
        to ProximityLowerBoundAcquisition.
        """
        def dummy_target(config, seed=0) -> float:
            return 0.0

        for cfg_key, spec in ALL_8_CONFIG_SPECS.items():
            with self.subTest(config=cfg_key):
                file_path = CONFIGS_DIR / spec["file"]
                self.assertTrue(file_path.exists(), f"Configuration file missing: {file_path}")

                with open(file_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)

                opt_section = cfg["optimizer"]

                with tempfile.TemporaryDirectory() as td:
                    smac_cfg_dict = dict(opt_section["smac_cfg"])
                    smac_cfg_dict["scenario"] = {
                        "n_trials": 3,
                        "seed": 42,
                        "output_directory": td,
                    }
                    smac_cfg_dict["smac_kwargs"] = {"overwrite": True}

                    opt_inst = SMAC3Optimizer.__new__(SMAC3Optimizer)
                    opt_inst.configspace = self.cs
                    opt_inst.target_function = dummy_target
                    opt_inst.acq_func_name = opt_section["acq_func_name"]
                    opt_inst.acq_func_kwargs = opt_section.get("acq_func_kwargs", {})
                    opt_inst.smac_cfg = OmegaConf.create(smac_cfg_dict)

                    solver = scripts.run_carps_patched.patched_smac3_setup_optimizer(opt_inst)
                    opt_inst._solver = solver

                    self.assertIsNotNone(solver)
                    acq = solver._acquisition_function
                    self.assertIsInstance(acq, ProximityLowerBoundAcquisition)

                    # Check that method from acq_func_kwargs was correctly passed into the acquisition
                    if spec["method"] is not None:
                        self.assertEqual(acq._method, spec["method"])

    def test_b_and_bc_immediate_active_acquisition(self):
        """
        Verify that Proximity B and BC (both OOB and CV) activate proximity intervals
        immediately after initial design (N=5) without waiting for N >= 25,
        while discrete Methods A and AC remain in RF warmup (predict_standard_rf) when N=5.
        """
        X_eval = np.array([[0.0, 0.0], [1.0, -1.0]])

        # 1. Continuous methods (B, B_CV, BC, BC_CV) must bypass warmup at N=5
        continuous_keys = ["B", "B_CV", "BC", "BC_CV"]
        for key in continuous_keys:
            spec = ALL_8_CONFIG_SPECS[key]
            with self.subTest(continuous_key=key):
                mock_model = MockSurrogateWithTracking(n_train=5, uncertainty_func=spec["uncertainty_func"])
                acq = ProximityLowerBoundAcquisition(
                    method=spec["method"],
                    eps=spec["eps"],
                    level=spec["level"],
                )
                acq.update(model=mock_model, num_data=5)

                self.assertTrue(acq._is_continuous_method())

                scores = acq._compute(X_eval)
                self.assertEqual(scores.shape, (len(X_eval), 1))
                self.assertTrue(np.all(np.isfinite(scores)))

                # Proximity intervals must be active immediately (0 standard RF calls, 1 intervals call)
                self.assertEqual(
                    mock_model.standard_rf_calls, 0,
                    f"Method {key} at N=5 must not call predict_standard_rf",
                )
                self.assertEqual(
                    mock_model.predict_with_intervals_calls, 1,
                    f"Method {key} at N=5 must call predict_with_intervals immediately",
                )

        # 2. Discrete methods (A_CV, AC, AC_CV) must use standard RF warmup when N=5 <= 25
        discrete_keys = ["A_CV", "AC", "AC_CV"]
        for key in discrete_keys:
            spec = ALL_8_CONFIG_SPECS[key]
            with self.subTest(discrete_key=key):
                mock_model = MockSurrogateWithTracking(n_train=5, uncertainty_func=spec["uncertainty_func"])
                acq = ProximityLowerBoundAcquisition(
                    method=spec["method"],
                    k=spec.get("k", 25),
                    k_warmup=spec.get("k_warmup", 25),
                    eps=spec["eps"],
                    level=spec["level"],
                )
                acq.update(model=mock_model, num_data=5)

                self.assertFalse(acq._is_continuous_method())

                scores = acq._compute(X_eval)
                self.assertEqual(scores.shape, (len(X_eval), 1))
                self.assertTrue(np.all(np.isfinite(scores)))

                # Phase 1 warmup must be active (1 standard RF call, 0 intervals calls)
                self.assertEqual(
                    mock_model.standard_rf_calls, 1,
                    f"Discrete method {key} at N=5 must call predict_standard_rf (warmup)",
                )
                self.assertEqual(
                    mock_model.predict_with_intervals_calls, 0,
                    f"Discrete method {key} at N=5 must not call predict_with_intervals",
                )


if __name__ == "__main__":
    unittest.main()

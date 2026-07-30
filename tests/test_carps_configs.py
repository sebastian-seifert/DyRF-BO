import os
import sys
import unittest
from omegaconf import OmegaConf

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCARPSConfigs(unittest.TestCase):
    def test_hydra_config_loading(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "carps_integration", "configs", "optimizer", "dyrf_epistemic_hpobench.yaml"
        )
        
        # Verify file exists
        self.assertTrue(os.path.exists(config_path), f"Config file not found at: {config_path}")
        
        # Load and parse config
        cfg = OmegaConf.load(config_path)
        
        # Verify keys
        self.assertEqual(cfg.optimizer_id, "CARPSDynamicRF")
        self.assertEqual(cfg.optimizer._target_, "carps_integration.optimizer.CARPSDynamicRFOptimizer")
        self.assertEqual(cfg.optimizer.extractor_name, "standard_disagreement")
        self.assertEqual(cfg.optimizer.kappa, 1.96)
        self.assertEqual(cfg.optimizer.min_samples_leaf_base, 2)

    def test_hydra_acq_function_config(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "carps_integration", "configs", "optimizer", "dyrf_epistemic_ei.yaml"
        )
        cfg = OmegaConf.load(config_path)
        self.assertIn("acq_func_name", cfg.optimizer)
        self.assertIn("acq_func_kwargs", cfg.optimizer)

    def test_run_carps_patched_overrides_yahpo_dir(self):
        # Import the script to apply the override
        import scripts.run_carps_patched
        import carps.objective_functions.yahpo
        
        # Verify that YAHPO_TASK_DATA_DIR was programmatically redirected if on cluster
        if os.path.exists("/bigwork/nhwpseis/benchmarks"):
            self.assertEqual(
                str(carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR),
                "/bigwork/nhwpseis/benchmarks/yahpo-data"
            )
        else:
            self.assertIsNotNone(carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR)

    def test_log_python_env_creates_directory_if_missing(self):
        import tempfile
        import shutil
        import scripts.run_carps_patched
        from carps.utils.loggingutils import log_python_env
        
        # Create a temporary directory path that does NOT exist
        temp_dir = tempfile.mkdtemp()
        nested_dir = os.path.join(temp_dir, "nested", "path", "to", "logs")
        log_file = os.path.join(nested_dir, "env_info.txt")
        
        try:
            # Call log_python_env - this should succeed and create the directories
            log_python_env(log_file=log_file)
            
            # Verify the file was created
            self.assertTrue(os.path.isfile(log_file))
        finally:
            shutil.rmtree(temp_dir)

    def test_run_carps_patched_safeguards_missing_optimizer_id(self):
        import subprocess
        
        # Run scripts/run_carps_patched.py with NO overrides to trigger execution crash
        run_cmd = [sys.executable, "scripts/run_carps_patched.py"]
        result = subprocess.run(run_cmd, capture_output=True, text=True)
        
        # The exit code should be non-zero (since configuration is incomplete)
        self.assertNotEqual(result.returncode, 0)
        
        # Verify that Hydra's exception formatter did NOT crash, meaning the real traceback was printed.
        # When the formatter crashes, it prints "An error occurred during Hydra's exception formatting" 
        # and has "full_key: hydra.run.dir" due to path interpolation failure.
        combined_output = result.stdout + result.stderr
        self.assertNotIn("An error occurred during Hydra's exception formatting", combined_output)
        self.assertNotIn("full_key: hydra.run.dir", combined_output)
        
        # Verify that the normal config missing value error is raised at runtime
        self.assertIn("Missing mandatory value", combined_output)

    def test_smac3_acquisition_function_override_ei_pi_lcb(self):
        """Verify that SMAC3Optimizer instantiates PI/LCB/EI acquisition functions cleanly when acq_func_name is passed."""
        import scripts.run_carps_patched
        from carps.optimizers.smac20 import SMAC3Optimizer
        import smac.acquisition.function as acq_module
        from omegaconf import OmegaConf

        acq_classes = {
            "ei": acq_module.EI,
            "pi": acq_module.PI,
            "lcb": acq_module.LCB,
        }

        for acq_name, expected_cls in acq_classes.items():
            with self.subTest(acq_func=acq_name):
                smac_cfg = OmegaConf.create({
                    "smac_class": "smac.facade.hyperparameter_optimization_facade.HyperparameterOptimizationFacade",
                    "scenario": {
                        "n_trials": 5,
                        "seed": 1,
                    },
                    "smac_kwargs": {}
                })
                
                from ConfigSpace import ConfigurationSpace, Float
                cs = ConfigurationSpace()
                cs.add(Float("x", (0.0, 1.0)))

                optimizer_inst = SMAC3Optimizer.__new__(SMAC3Optimizer)
                optimizer_inst.configspace = cs
                optimizer_inst.target_function = lambda config, seed=0: 0.0
                optimizer_inst.acq_func_name = acq_name
                optimizer_inst.smac_cfg = smac_cfg
                
                # Execute patched _setup_optimizer logic
                solver = scripts.run_carps_patched.patched_smac3_setup_optimizer(optimizer_inst)
                acq_fn = solver._acquisition_function
                self.assertIsInstance(acq_fn, expected_cls, f"acq_func_name='{acq_name}' should instantiate {expected_cls}")

if __name__ == "__main__":
    unittest.main()

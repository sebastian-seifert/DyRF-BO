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

    def test_run_carps_patched_overrides_yahpo_dir(self):
        # Import the script to apply the override
        import scripts.run_carps_patched
        import carps.objective_functions.yahpo
        
        # Verify that YAHPO_TASK_DATA_DIR was programmatically redirected
        self.assertEqual(
            str(carps.objective_functions.yahpo.YAHPO_TASK_DATA_DIR),
            "/bigwork/nhwpseis/benchmarks/yahpo-data"
        )

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
        
        # The exit code should be non-zero (since task and optimizer are missing)
        self.assertNotEqual(result.returncode, 0)
        
        # The stderr/stdout should report the missing 'task' or 'optimizer', NOT 'optimizer_id'
        combined_output = result.stdout + result.stderr
        self.assertNotIn("Missing mandatory value: optimizer_id", combined_output)

if __name__ == "__main__":
    unittest.main()

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

if __name__ == "__main__":
    unittest.main()

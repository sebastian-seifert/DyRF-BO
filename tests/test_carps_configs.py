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
        self.assertEqual(cfg.optimizer.n_base, 100)

if __name__ == "__main__":
    unittest.main()

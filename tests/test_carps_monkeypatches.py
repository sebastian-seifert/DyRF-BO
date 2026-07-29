import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_carps_patched

class TestCARPSMonkeypatches(unittest.TestCase):
    def test_file_logger_unlink_resilience(self):
        import carps.loggers.file_logger
        from carps.loggers.file_logger import FileLogger

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create dummy trial_logs.jsonl and env_info.txt
            dummy_logs = temp_dir / "trial_logs.jsonl"
            dummy_env = temp_dir / "env_info.txt"
            dummy_logs.write_text('{"test": 1}\n')
            dummy_env.write_text('Python 3.14\n')

            # Create a non-existent file entry or delete a file in the middle of traversal simulation
            # FileLogger with overwrite=True should clear old files without raising FileNotFoundError
            logger_inst = FileLogger.__new__(FileLogger)
            FileLogger.__init__(logger_inst, overwrite=True, directory=temp_dir)

            self.assertEqual(logger_inst.directory, temp_dir)
            self.assertTrue((temp_dir / "env_info.txt").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_yahpo_gym_config_redirection(self):
        import yahpo_gym
        expected_path = Path("/bigwork/nhwpseis/benchmarks/yahpo-data")
        self.assertEqual(Path(yahpo_gym.local_config.data_path), expected_path)

    def test_configspace_sort_hyperparameters_compatibility(self):
        import ConfigSpace
        from ConfigSpace import ConfigurationSpace, Float
        cs = ConfigurationSpace()
        cs.add(Float("x", (0.0, 1.0)))
        
        self.assertTrue(hasattr(ConfigSpace.ConfigurationSpace, "_sort_hyperparameters"))
        hps = cs._sort_hyperparameters()
        self.assertEqual(len(hps), 1)

    def test_smac3_optimizer_acq_func_name_patch(self):
        from unittest.mock import MagicMock
        import carps.optimizers.smac20
        from carps.optimizers.smac20 import SMAC3Optimizer
        from omegaconf import OmegaConf

        mock_task = MagicMock()
        mock_task.optimization_resources.time_budget = None
        mock_smac_cfg = OmegaConf.create({"smac_class": "smac.facade.HyperparameterOptimizationFacade", "scenario": {}})
        
        # Initializing SMAC3Optimizer with extra kwargs (like acq_func_name) should not raise TypeError
        try:
            opt = SMAC3Optimizer(task=mock_task, smac_cfg=mock_smac_cfg, acq_func_name="lcb")
            self.assertEqual(getattr(opt, "acq_func_name", None), "lcb")
        except TypeError as e:
            self.fail(f"SMAC3Optimizer raised TypeError with acq_func_name: {e}")

    def test_omegaconf_missing_mandatory_value_fallback(self):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({"benchmark_id": "???", "conda_env_name": "carps_${benchmark_id}_container"})
        res = OmegaConf.select(cfg, "conda_env_name")
        self.assertIsNotNone(res)

if __name__ == "__main__":
    unittest.main()


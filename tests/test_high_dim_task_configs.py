import os
import sys
import unittest
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHighDimTaskConfigs(unittest.TestCase):
    def test_new_rbv2_super_configs_exist_and_load(self):
        new_instances = [
            "1040", "1049", "1050", "1056", "1067",
            "1068", "1111", "1220", "1461", "1462", "1464"
        ]
        carps_yahpo_so_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".venv/lib/python3.14/site-packages/carps/configs/task/YAHPO/SO"
        )
        for inst in new_instances:
            cfg_path = os.path.join(carps_yahpo_so_dir, f"cfg_rbv2_super_{inst}.yaml")
            self.assertTrue(os.path.exists(cfg_path), f"Config file missing: {cfg_path}")
            cfg = OmegaConf.load(cfg_path)
            self.assertEqual(cfg.task.objective_function.bench, "rbv2_super")
            self.assertEqual(str(cfg.task.objective_function.instance), inst)

    def test_hpobench_nas_configs_exist_and_load(self):
        hpobench_nas_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            ".venv/lib/python3.14/site-packages/carps/configs/task/HPOBench/blackbox/tabular/nas"
        )
        nas_configs = [
            "cfg_Cifar10ValidNasBench201Benchmark.yaml",
            "cfg_Cifar100NasBench201Benchmark.yaml",
            "cfg_ImageNetNasBench201Benchmark.yaml"
        ]
        for cfg_name in nas_configs:
            cfg_path = os.path.join(hpobench_nas_dir, cfg_name)
            self.assertTrue(os.path.exists(cfg_path), f"NAS Config file missing: {cfg_path}")
            cfg = OmegaConf.load(cfg_path)
            self.assertIn("NasBench201Benchmark", cfg.task.name)

if __name__ == "__main__":
    unittest.main()

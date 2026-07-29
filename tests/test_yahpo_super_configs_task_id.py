import os
import glob
import yaml
import unittest

class TestYahpoSuperConfigsTaskId(unittest.TestCase):
    def test_yahpo_super_configs_task_id_matching(self):
        config_dirs = [
            "carps_integration/configs/task/YAHPO/SO",
            ".venv/lib/python3.14/site-packages/carps/configs/task/YAHPO/SO"
        ]
        
        for cdir in config_dirs:
            if not os.path.exists(cdir):
                continue
            pattern = os.path.join(cdir, "cfg_rbv2_super_*.yaml")
            files = glob.glob(pattern)
            self.assertGreater(len(files), 0)
            
            for filepath in files:
                filename = os.path.basename(filepath)
                # e.g. cfg_rbv2_super_1040.yaml -> 1040
                expected_task_id = filename.replace("cfg_rbv2_super_", "").replace(".yaml", "")
                
                with open(filepath, "r") as f:
                    content = f.read()
                
                # Parse yaml
                data = yaml.safe_load(content)
                hyperparameters = data["task"]["input_space"]["configuration_space"]["d"]["hyperparameters"]
                task_id_hp = [hp for hp in hyperparameters if hp.get("name") == "task_id"]
                self.assertEqual(len(task_id_hp), 1, f"Missing task_id in {filepath}")
                val = str(task_id_hp[0]["value"])
                self.assertEqual(val, expected_task_id, f"In {filepath}: task_id value '{val}' does not match expected '{expected_task_id}'")

if __name__ == "__main__":
    unittest.main()

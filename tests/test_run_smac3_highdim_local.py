import os
import json
import unittest
import tempfile
import shutil

from scripts.run_smac3_highdim_local import extract_telemetry_from_carps_run

class TestRunSMAC3HighdimLocal(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.test_dir, "runs/SMAC3-HPOFacade/YAHPO/yahpo/so/rbv2_super/1040/None/1")
        os.makedirs(self.run_dir, exist_ok=True)

        # Create dummy trial_logs.jsonl
        self.trial_logs_file = os.path.join(self.run_dir, "trial_logs.jsonl")
        dummy_trials = [
            {"trial_idx": 1, "cost": -0.85, "config": {"lr": 0.01}},
            {"trial_idx": 2, "cost": -0.92, "config": {"lr": 0.001}},
        ]
        with open(self.trial_logs_file, "w") as f:
            for t in dummy_trials:
                f.write(json.dumps(t) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_telemetry_from_carps_run(self):
        out_telemetry_file = os.path.join(self.test_dir, "telemetry_smac3_cfg_rbv2_super_1040_seed1.json")
        res = extract_telemetry_from_carps_run(
            carps_run_dir=self.run_dir,
            task_name="cfg_rbv2_super_1040",
            seed=1,
            output_path=out_telemetry_file
        )
        self.assertTrue(os.path.exists(out_telemetry_file))
        self.assertEqual(res["task_name"], "cfg_rbv2_super_1040")
        self.assertEqual(res["seed"], 1)
        self.assertEqual(res["extractor_name"], "smac3_bo")
        self.assertEqual(len(res["trials"]), 2)
        self.assertEqual(res["trials"][1]["cost"], -0.92)

if __name__ == "__main__":
    unittest.main()

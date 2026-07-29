import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_smac3_highdim_array_tasks import generate_smac3_highdim_array_tasks

class TestGenerateSMAC3HighDimArrayTasks(unittest.TestCase):
    def setUp(self):
        self.output_file = "results/test_smac3_highdim_array_tasks.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_generate_smac3_highdim_array_tasks(self):
        lines = generate_smac3_highdim_array_tasks(output_path=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

        # 18 high-dim tasks * 5 seeds = 90 tasks
        self.assertEqual(len(lines), 90)

        for line in lines:
            self.assertIn("+optimizer/smac20=hpo", line)
            self.assertIn("optimizer_id=SMAC3-HPOFacade", line)
            self.assertIn("optimizer_container_id=SMAC3", line)
            self.assertNotIn("acq_func_name", line)
            self.assertNotIn("acq_uncertainty_type", line)
            self.assertNotIn("telemetry_path", line)

if __name__ == "__main__":
    unittest.main()

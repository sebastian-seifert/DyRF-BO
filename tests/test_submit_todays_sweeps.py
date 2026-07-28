import os
import unittest

class TestSubmitTodaysSweeps(unittest.TestCase):
    def test_todays_sweeps_script_content(self):
        script_path = "submit_todays_sweeps.sh"
        self.assertTrue(os.path.exists(script_path))
        with open(script_path, "r") as f:
            content = f.read()
        
        self.assertIn("submit_sweep1_empty.sh", content)
        self.assertIn("submit_sweep2_linear_sparse.sh", content)
        self.assertNotIn("submit_sweep6_carps_epistemic_ei.sh", content)

if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_smoke_epistemic_full_acq_tasks import generate_smoke_epistemic_full_acq_tasks

class TestSmokeEpistemicFullAcq(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "test_smoke_tasks.txt")

    def test_smoke_task_generation(self):
        lines = generate_smoke_epistemic_full_acq_tasks(output_path=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        self.assertGreater(len(lines), 0)

        # Check for task prefix format +task/YAHPO
        self.assertTrue(any("+task/YAHPO" in line for line in lines))

        # Check for all three acquisitions in the smoke tasks
        ei_found = any("acq_func_name=ei" in line for line in lines)
        pi_found = any("acq_func_name=pi" in line for line in lines)
        lcb_found = any("acq_func_name=lcb" in line for line in lines)

        self.assertTrue(ei_found, "EI acquisition missing from smoke tasks")
        self.assertTrue(pi_found, "PI acquisition missing from smoke tasks")
        self.assertTrue(lcb_found, "LCB acquisition missing from smoke tasks")

        # Check for DyRF epistemic and SMAC3 baseline presence
        custom_found = any("+optimizer=smac20_custom_uncertainty" in line for line in lines)
        smac_found = any("+optimizer/smac20=hpo" in line for line in lines)
        self.assertTrue(custom_found, "Custom uncertainty optimizer missing from smoke tasks")
        self.assertTrue(smac_found, "SMAC3 baseline optimizer missing from smoke tasks")

if __name__ == "__main__":
    unittest.main()

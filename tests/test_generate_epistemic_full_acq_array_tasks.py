import os
import sys
import unittest
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_epistemic_full_acq_array_tasks import generate_full_acq_array_tasks

class TestGenerateEpistemicFullAcqArrayTasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "test_full_acq_tasks.txt")

    def test_task_generation_counts_and_acquisitions(self):
        lines = generate_full_acq_array_tasks(output_path=self.output_file)
        # 27 benchmarks * 8 approaches * 3 acqs * 5 seeds = 3240 DyRF tasks
        # 27 benchmarks * 1 baseline * 3 acqs * 5 seeds = 405 SMAC3 tasks
        # Total = 3645
        self.assertEqual(len(lines), 3645)
        self.assertTrue(os.path.exists(self.output_file))

        # Check acquisition breakdown
        ei_count = sum(1 for line in lines if "acq_func_name=ei" in line or "smac3_ei" in line or "results/epistemic_acq/ei/" in line)
        pi_count = sum(1 for line in lines if "acq_func_name=pi" in line or "smac3_pi" in line or "results/epistemic_acq/pi/" in line)
        lcb_count = sum(1 for line in lines if "acq_func_name=lcb" in line or "smac3_lcb" in line or "results/epistemic_acq/lcb/" in line)

        self.assertEqual(ei_count, 1215)
        self.assertEqual(pi_count, 1215)
        self.assertEqual(lcb_count, 1215)

        # Check SMAC3 baseline tasks count (405 total, 135 per acq)
        smac_lines = [line for line in lines if "smac20" in line or "SMAC3" in line]
        self.assertEqual(len(smac_lines), 405)

if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_epistemic_full_acq_sparse4_array_tasks import generate_full_acq_sparse4_array_tasks

class TestGenerateEpistemicFullAcqSparse4ArrayTasks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "test_full_acq_sparse4_tasks.txt")

    def test_task_generation_counts_and_acquisitions(self):
        lines = generate_full_acq_sparse4_array_tasks(output_path=self.output_file)
        # 4 benchmarks * 8 approaches * 3 acqs * 5 seeds = 480 DyRF tasks
        # 4 benchmarks * 1 baseline * 3 acqs * 5 seeds = 60 SMAC3 tasks
        # Total = 540
        self.assertEqual(len(lines), 540)
        self.assertTrue(os.path.exists(self.output_file))

        # Check acquisition breakdown
        ei_count = sum(1 for line in lines if "acq_func_name=ei" in line or "/ei/" in line)
        pi_count = sum(1 for line in lines if "acq_func_name=pi" in line or "/pi/" in line)
        lcb_count = sum(1 for line in lines if "acq_func_name=lcb" in line or "/lcb/" in line)

        self.assertEqual(ei_count, 180)
        self.assertEqual(pi_count, 180)
        self.assertEqual(lcb_count, 180)

        # Check SMAC3 baseline tasks count (60 total, 20 per acq)
        smac_lines = [line for line in lines if "+optimizer/smac20=hpo" in line]
        self.assertEqual(len(smac_lines), 60)

        # Check Custom Uncertainty tasks count (480 total, 160 per acq)
        custom_lines = [line for line in lines if "+optimizer=smac20_custom_uncertainty" in line and "++optimizer.smac_cfg.model_kwargs.uncertainty_func=" in line]
        self.assertEqual(len(custom_lines), 480)

if __name__ == "__main__":
    unittest.main()

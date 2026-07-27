import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_epistemic_acq_array_tasks import generate_array_tasks

class TestGenerateEpistemicAcqArrayTasks(unittest.TestCase):
    def setUp(self):
        self.output_file = "results/test_epistemic_acq_array_tasks.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_generate_array_tasks_file(self):
        lines = generate_array_tasks(output_path=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        # 3 acqs * 40 tasks * 8 extractors * 5 seeds (4800) + 1 baseline * 40 tasks * 5 seeds (200) = 5000 lines
        self.assertEqual(len(lines), 5000)

        # Check for epistemic acquisition lines across ei, pi, lcb
        for acq in ["ei", "pi", "lcb"]:
            epistemic_lines = [l for l in lines if f"optimizer.acq_func_name={acq}" in l and "optimizer.acq_uncertainty_type=epistemic" in l]
            self.assertEqual(len(epistemic_lines), 1600, f"Acquisition {acq} should have 1600 DyRF epistemic task lines")

        smac_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]
        self.assertEqual(len(smac_lines), 200)


        # Ensure all 8 registered extractors are covered across 40 tasks and 5 seeds
        from ep_extractors import UQExtractorRegistry
        for extractor in UQExtractorRegistry.list_registered():
            extractor_lines = [l for l in lines if f"optimizer.extractor_name={extractor} " in l]
            self.assertEqual(len(extractor_lines), 3 * 40 * 5, f"Extractor {extractor} should have 600 task lines across 3 acq functions")

if __name__ == "__main__":
    unittest.main()


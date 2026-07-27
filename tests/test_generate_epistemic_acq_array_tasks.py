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
        # 3 acqs * 26 tasks * 8 extractors * 5 seeds (3120) + 3 acqs * 26 tasks * 5 seeds (390) = 3510 lines
        self.assertEqual(len(lines), 3510)

        # Check for epistemic acquisition lines across ei, pi, lcb
        for acq in ["ei", "pi", "lcb"]:
            acq_lines = [l for l in lines if f"optimizer.acq_func_name={acq}" in l]
            self.assertEqual(len(acq_lines), 1170, f"Acquisition {acq} should have 1170 task lines")

            epistemic_lines = [l for l in acq_lines if "optimizer.acq_uncertainty_type=epistemic" in l]
            smac_lines = [l for l in acq_lines if "+optimizer/smac20=hpo" in l]
            self.assertEqual(len(epistemic_lines), 1040)
            self.assertEqual(len(smac_lines), 130)

        # Ensure all 8 registered extractors are covered
        from ep_extractors import UQExtractorRegistry
        for extractor in UQExtractorRegistry.list_registered():
            extractor_lines = [l for l in lines if f"optimizer.extractor_name={extractor} " in l]
            self.assertEqual(len(extractor_lines), 3 * 26 * 5, f"Extractor {extractor} should have 390 task lines across 3 acq functions")

if __name__ == "__main__":
    unittest.main()

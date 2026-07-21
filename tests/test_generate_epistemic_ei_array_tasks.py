import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_epistemic_ei_array_tasks import generate_array_tasks

class TestGenerateEpistemicEIArrayTasks(unittest.TestCase):
    def setUp(self):
        self.output_file = "results/test_epistemic_ei_array_tasks.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_generate_array_tasks_file(self):
        lines = generate_array_tasks(output_path=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        self.assertGreater(len(lines), 0)

        # Check for epistemic acquisition lines
        epistemic_lines = [l for l in lines if "optimizer.acq_uncertainty_type=epistemic" in l]
        total_lines = [l for l in lines if "optimizer.acq_uncertainty_type=total" in l]
        smac_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]

        self.assertGreater(len(epistemic_lines), 0)
        self.assertGreater(len(total_lines), 0)
        self.assertGreater(len(smac_lines), 0)

        # Ensure all 8 registered extractors are covered in epistemic lines
        from ep_extractors import UQExtractorRegistry
        for extractor in UQExtractorRegistry.list_registered():
            extractor_lines = [l for l in epistemic_lines if f"optimizer.extractor_name={extractor}" in l]
            self.assertGreater(len(extractor_lines), 0, f"Extractor {extractor} missing in array tasks")

if __name__ == "__main__":
    unittest.main()

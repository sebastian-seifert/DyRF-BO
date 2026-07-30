import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_epistemic_ei_highdim_array_tasks import generate_highdim_array_tasks

class TestGenerateEpistemicEIHighDimArrayTasks(unittest.TestCase):
    def setUp(self):
        self.output_file = "results/test_epistemic_ei_highdim_array_tasks.txt"
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_generate_highdim_array_tasks_file(self):
        lines = generate_highdim_array_tasks(output_path=self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        # 1 acq (EI) * 18 high-dim tasks * 8 extractors * 5 seeds = 720 DyRF tasks
        # 1 baseline * 18 high-dim tasks * 5 seeds = 90 SMAC3 tasks
        # Total = 720 + 90 = 810 tasks
        self.assertEqual(len(lines), 810)

        custom_lines = [l for l in lines if "+optimizer=smac20_custom_uncertainty" in l and "++optimizer.smac_cfg.model_kwargs.uncertainty_func=" in l]
        smac_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]

        self.assertEqual(len(custom_lines), 720)
        self.assertEqual(len(smac_lines), 90)

        from ep_extractors import UQExtractorRegistry
        for extractor in UQExtractorRegistry.list_registered():
            extractor_lines = [l for l in custom_lines if f"++optimizer.smac_cfg.model_kwargs.uncertainty_func={extractor} " in l]
            self.assertEqual(len(extractor_lines), 18 * 5, f"Extractor {extractor} should have 90 task lines in High-Dim EI sweep")

if __name__ == "__main__":
    unittest.main()

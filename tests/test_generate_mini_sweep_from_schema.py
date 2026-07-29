import os
import sys
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_mini_sweep_from_schema import create_mini_sweep

class TestGenerateMiniSweepFromSchema(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_mini_sweep(self):
        tasks_file, config_file = create_mini_sweep(
            output_dir=self.test_dir,
            functions=["sin", "damped_osc"],
            rf_configs=["A"],
            seeds=[1]
        )
        self.assertTrue(os.path.exists(tasks_file))
        self.assertTrue(os.path.exists(config_file))

        with open(tasks_file, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # 2 functions * 2 lines per function (1 base, 1 prox) = 4 lines
        self.assertEqual(len(lines), 4)
        self.assertIn("--function damped_osc", lines[0])

if __name__ == "__main__":
    unittest.main()

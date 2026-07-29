import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.benchmark_registry import BenchmarkRegistry

class TestBenchmarkRegistry(unittest.TestCase):
    def test_task_counts_and_categories(self):
        tasks = BenchmarkRegistry.get_all_tasks()
        self.assertEqual(len(tasks), 37)

        cats = BenchmarkRegistry.get_tasks_by_category()
        self.assertIn("Low-Dim (<=6D)", cats)
        self.assertIn("Mid-Dim (7-20D)", cats)
        self.assertIn("High-Dim & NAS (>20D)", cats)

        self.assertEqual(len(cats["Low-Dim (<=6D)"]), 10)
        self.assertEqual(len(cats["Mid-Dim (7-20D)"]), 9)
        self.assertEqual(len(cats["High-Dim & NAS (>20D)"]), 18)

        # Check total across categories matches get_all_tasks
        total_cat_tasks = sum(len(t) for t in cats.values())
        self.assertEqual(total_cat_tasks, 37)

    def test_get_task_category(self):
        self.assertEqual(
            BenchmarkRegistry.get_task_category("+task/YAHPO/SO=cfg_nb301_CIFAR10"),
            "High-Dim & NAS (>20D)"
        )
        self.assertEqual(
            BenchmarkRegistry.get_task_category("+task/HPOBench/blackbox/tabular/ml=cfg_ml_svm_3"),
            "Low-Dim (<=6D)"
        )

    def test_get_balanced_tasks(self):
        balanced_tasks = BenchmarkRegistry.get_balanced_tasks()
        self.assertEqual(len(balanced_tasks), 27)

        by_cat = BenchmarkRegistry.get_balanced_tasks_by_category()
        self.assertEqual(len(by_cat["Low-Dim (<=6D)"]), 9)
        self.assertEqual(len(by_cat["Mid-Dim (7-20D)"]), 9)
        self.assertEqual(len(by_cat["High-Dim & NAS (>20D)"]), 9)

if __name__ == "__main__":
    unittest.main()


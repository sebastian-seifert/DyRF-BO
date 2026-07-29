import os
import sys
import unittest
import tempfile
import shutil

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_local_multiprocessing_sweep import load_tasks_from_sweep_dir, execute_single_task

class TestLocalMultiprocessingSweep(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.sweep_dir = os.path.join(self.test_dir, "sweep_1_empty")
        os.makedirs(self.sweep_dir, exist_ok=True)
        self.tasks_file = os.path.join(self.sweep_dir, "tasks.txt")
        
        self.dummy_tasks = [
            "--function ackley_1d --rf_config A --seed 1 --gap_type empty --ood_type hypercube --approaches Standard --output_dir " + os.path.join(self.sweep_dir, "raw"),
            "--function ackley_2d --rf_config B --seed 2 --gap_type empty --ood_type hypercube --approaches Chen --output_dir " + os.path.join(self.sweep_dir, "raw")
        ]
        with open(self.tasks_file, "w") as f:
            for line in self.dummy_tasks:
                f.write(line + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_tasks_from_sweep_dir(self):
        tasks = load_tasks_from_sweep_dir(self.sweep_dir)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0][0], 1)
        self.assertIn("--function ackley_1d", tasks[0][1])
        self.assertEqual(tasks[1][0], 2)
        self.assertIn("--function ackley_2d", tasks[1][1])

    def test_execute_single_task_dry_run(self):
        log_dir = os.path.join(self.sweep_dir, "local_logs")
        res = execute_single_task(
            task_info=(1, self.dummy_tasks[0]),
            sweep_dir=self.sweep_dir,
            log_dir=log_dir,
            dry_run=True
        )
        self.assertEqual(res["status"], "success_dry_run")
        self.assertEqual(res["task_idx"], 1)

    def test_get_default_python_exec(self):
        from scripts.run_local_multiprocessing_sweep import get_default_python_exec
        py_exec = get_default_python_exec()
        self.assertTrue(os.path.exists(py_exec))


if __name__ == "__main__":
    unittest.main()

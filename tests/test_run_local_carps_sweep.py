import os
import sys
import unittest
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_local_carps_sweep import execute_single_carps_task, load_tasks_from_file

class TestRunLocalCARPSSweep(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.task_file = os.path.join(self.temp_dir, "test_tasks.txt")
        with open(self.task_file, "w") as f:
            f.write("+task/YAHPO/SO=cfg_rbv2_glmnet_375 seed=1\n")
            f.write("+task/YAHPO/SO=cfg_rbv2_glmnet_375 seed=2\n")

    def test_load_tasks(self):
        tasks = load_tasks_from_file(self.task_file)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0][0], 1)
        self.assertIn("seed=1", tasks[0][1])

    def test_execute_dry_run(self):
        tasks = load_tasks_from_file(self.task_file)
        log_dir = os.path.join(self.temp_dir, "logs")
        res = execute_single_carps_task(tasks[0], log_dir=log_dir, dry_run=True)
        self.assertEqual(res["status"], "success_dry_run")

if __name__ == "__main__":
    unittest.main()

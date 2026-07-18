import os
import unittest
import sys
from pathlib import Path

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).parent.parent))

class TestGenerateArrayTasks(unittest.TestCase):
    def setUp(self):
        self.task_file = Path("results/array_tasks.txt")
        # Store backup if exists
        self.backup_exists = self.task_file.exists()
        if self.backup_exists:
            self.backup_content = self.task_file.read_text()

    def tearDown(self):
        # Restore backup if exists
        if self.backup_exists:
            self.task_file.parent.mkdir(parents=True, exist_ok=True)
            self.task_file.write_text(self.backup_content)

    def test_generated_tasks_format(self):
        # Import/execute scripts/generate_array_tasks.py
        import scripts.generate_array_tasks
        
        # Verify array_tasks.txt was generated
        self.assertTrue(self.task_file.exists(), "results/array_tasks.txt was not generated.")
        
        lines = self.task_file.read_text().splitlines()
        self.assertGreater(len(lines), 0, "Generated tasks file is empty.")
        
        for idx, line in enumerate(lines):
            # Each command line must have optimizer_id and optimizer_container_id
            self.assertIn("optimizer_id=", line, f"Line {idx+1} missing 'optimizer_id': {line}")
            self.assertIn("optimizer_container_id=", line, f"Line {idx+1} missing 'optimizer_container_id': {line}")
            
            # Baseline smac20 runs should use +optimizer/smac20=hpo, NOT optimizer=smac20/hpo or +optimizer=smac20/hpo
            if "smac20" in line:
                self.assertIn("+optimizer/smac20=hpo", line, f"Line {idx+1} should use '+optimizer/smac20=hpo': {line}")
                self.assertNotIn("optimizer=smac20/hpo", line, f"Line {idx+1} should not use 'optimizer=smac20/hpo': {line}")
                self.assertIn("optimizer_id=SMAC3-HPOFacade", line, f"Line {idx+1} missing SMAC3-HPOFacade ID: {line}")
                self.assertIn("optimizer_container_id=SMAC3", line, f"Line {idx+1} missing SMAC3 container ID: {line}")

    def test_sbatch_array_limit(self):
        sbatch_path = Path("scripts/submit_hpobench_array.sbatch")
        self.assertTrue(sbatch_path.exists())
        
        content = sbatch_path.read_text()
        for line in content.splitlines():
            if "#SBATCH --array=" in line:
                # Extract range (e.g. 1-260%15 -> 1-260)
                array_spec = line.split("=")[-1].split("%")[0]
                start, end = map(int, array_spec.split("-"))
                task_count = end - start + 1
                self.assertLessEqual(task_count, 300, f"SBATCH array size {task_count} exceeds LUIS cluster limit of 300.")

if __name__ == "__main__":
    unittest.main()

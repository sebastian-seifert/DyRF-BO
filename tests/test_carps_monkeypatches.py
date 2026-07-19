import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_carps_patched

class TestCARPSMonkeypatches(unittest.TestCase):
    def test_file_logger_unlink_resilience(self):
        import carps.loggers.file_logger
        from carps.loggers.file_logger import FileLogger

        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create dummy trial_logs.jsonl and env_info.txt
            dummy_logs = temp_dir / "trial_logs.jsonl"
            dummy_env = temp_dir / "env_info.txt"
            dummy_logs.write_text('{"test": 1}\n')
            dummy_env.write_text('Python 3.14\n')

            # Create a non-existent file entry or delete a file in the middle of traversal simulation
            # FileLogger with overwrite=True should clear old files without raising FileNotFoundError
            logger_inst = FileLogger.__new__(FileLogger)
            FileLogger.__init__(logger_inst, overwrite=True, directory=temp_dir)

            self.assertEqual(logger_inst.directory, temp_dir)
            self.assertTrue((temp_dir / "env_info.txt").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_yahpo_gym_config_redirection(self):
        import yahpo_gym
        expected_path = Path("/bigwork/nhwpseis/benchmarks/yahpo-data")
        self.assertEqual(Path(yahpo_gym.local_config.data_path), expected_path)

if __name__ == "__main__":
    unittest.main()

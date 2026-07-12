import os
import sys
import unittest
import subprocess

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestCLIFunctionFilter(unittest.TestCase):
    def test_single_function_filter_execution(self):
        """Verify that Uncertainty_Quantification.py successfully filters and runs a single function via CLI."""
        python_exec = sys.executable
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Uncertainty_Quantification.py")
        
        # Call the script to run only the 'sin' function for 1 run (seed) using Standard approach
        cmd = [
            python_exec,
            script_path,
            "--function", "sin",
            "--n_runs", "1",
            "--approaches", "Standard",
            "--output_dir", "results/test_filter_run",
            "--debug_timing"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Verify success exit code
        self.assertEqual(
            result.returncode, 0,
            f"CLI run failed with exit code {result.returncode}.\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
        
        # Check stdout to verify that exactly 1 function was reported in [SETUP SUMMARY]
        self.assertIn("Functions: 1 total", result.stdout)
        self.assertIn("Starting: Function=sin", result.stdout)

if __name__ == "__main__":
    unittest.main()

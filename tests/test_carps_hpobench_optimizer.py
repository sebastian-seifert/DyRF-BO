import os
import sys
import unittest
import numpy as np
import json
from ConfigSpace import ConfigurationSpace, Float

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carps.utils.task import Task, InputSpace, OutputSpace, OptimizationResources, TaskMetadata
from carps.objective_functions.dummy_problem import DummyObjectiveFunction
from carps_integration.optimizer import CARPSDynamicRFOptimizer

class TestCARPSDynamicRFOptimizer(unittest.TestCase):
    def setUp(self):
        # Create a simple config space
        self.cs = ConfigurationSpace(space={"x": Float("x", bounds=(-2.0, 2.0))})
        # Create a dummy objective function returning 1.5
        self.obj_func = DummyObjectiveFunction(return_value=1.5, configuration_space=self.cs)
        
        self.task = Task(
            name="dummy_hpobench_test_task",
            objective_function=self.obj_func,
            input_space=InputSpace(configuration_space=self.cs),
            output_space=OutputSpace(),
            optimization_resources=OptimizationResources(n_trials=12),
            metadata=TaskMetadata()
        )
        
        self.telemetry_file = "test_telemetry.json"
        if os.path.exists(self.telemetry_file):
            os.remove(self.telemetry_file)

    def tearDown(self):
        if os.path.exists(self.telemetry_file):
            os.remove(self.telemetry_file)

    def test_optimizer_run(self):
        # Instantiate optimizer with standard_disagreement and 3 warmstart trials
        optimizer = CARPSDynamicRFOptimizer(
            task=self.task,
            extractor_name="standard_disagreement",
            n_init=3,
            kappa=1.96,
            telemetry_path=self.telemetry_file,
            window_size=3,
            n_base=15,
            n_min=5,
            n_max=30
        )
        
        # Run optimization
        incumbent = optimizer.run()
        
        # Verify incumbent structure
        self.assertIsNotNone(incumbent)
        self.assertEqual(optimizer.trial_counter, 12)
        
        # Verify telemetry file was written
        self.assertTrue(os.path.exists(self.telemetry_file), "Telemetry file was not created.")
        
        with open(self.telemetry_file, "r") as f:
            data = json.load(f)
            
        self.assertIn("trials", data)
        self.assertEqual(len(data["trials"]), 12)
        
        # Check first trial (should be warmstart, so no fitted surrogate yet)
        first_trial = data["trials"][0]
        self.assertEqual(first_trial["trial_idx"], 0)
        self.assertEqual(first_trial["surrogate_n_estimators"], 15)  # initial base value
        
        # Check last trial (should have fitted surrogate and adapted parameters)
        last_trial = data["trials"][-1]
        self.assertEqual(last_trial["trial_idx"], 11)
        self.assertIn("surrogate_n_estimators", last_trial)
        self.assertIn("surrogate_max_depth", last_trial)
        
if __name__ == "__main__":
    unittest.main()

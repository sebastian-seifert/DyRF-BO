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
            min_samples_leaf_base=2,
            min_samples_leaf_min=1,
            min_samples_leaf_max=5,
            alpha=1.0,
            max_features_base=0.5,
            max_features_min=0.1,
            max_features_max=0.8,
            eta=0.5
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
        self.assertEqual(first_trial["surrogate_min_samples_leaf"], 2)  # initial base value
        self.assertEqual(first_trial["surrogate_max_features"], 0.5)  # initial base value
        
        # Check last trial (should have fitted surrogate and adapted parameters)
        last_trial = data["trials"][-1]
        self.assertEqual(last_trial["trial_idx"], 11)
        self.assertIn("surrogate_min_samples_leaf", last_trial)
        self.assertIn("surrogate_max_features", last_trial)

    def test_optimizer_acq_functions(self):
        for acq_name, acq_kwargs in [("ei", {"xi": 0.01}), ("lcb", {"beta": 2.0}), ("pi", {"xi": 0.05})]:
            optimizer = CARPSDynamicRFOptimizer(
                task=self.task,
                extractor_name="standard_disagreement",
                acq_func_name=acq_name,
                acq_func_kwargs=acq_kwargs,
                n_init=3,
                telemetry_path=self.telemetry_file
            )
            incumbent = optimizer.run()
            self.assertIsNotNone(incumbent)
            self.assertEqual(optimizer.acq_func_name, acq_name)
        
if __name__ == "__main__":
    unittest.main()

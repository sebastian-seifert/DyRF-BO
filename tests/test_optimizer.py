import os
import sys
import unittest
import numpy as np
from ConfigSpace import ConfigurationSpace, Categorical, Float, EqualsCondition

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carps.utils.task import Task
from carps.utils.trials import TrialInfo, TrialValue
from carps_integration.optimizer import CARPSDynamicRFOptimizer

from carps.utils.task import OptimizationResources

class MockObjectiveFunction:
    def __init__(self, cs):
        self.configspace = cs

class MockTask:
    def __init__(self, cs, name="mock_task"):
        self.name = name
        self.objective_function = MockObjectiveFunction(cs)
        self.optimization_resources = OptimizationResources(n_trials=50, time_budget=None)


class TestCARPSDynamicRFOptimizer(unittest.TestCase):
    def setUp(self):
        # Create a hierarchical ConfigSpace (with conditional hyperparameters that yield NaNs)
        self.cs = ConfigurationSpace()
        model = Categorical("model", ["rf", "svm"])
        rf_depth = Float("rf_depth", (1.0, 10.0), default=5.0)
        svm_gamma = Float("svm_gamma", (0.01, 1.0), default=0.1)
        self.cs.add_hyperparameters([model, rf_depth, svm_gamma])
        self.cs.add_conditions([
            EqualsCondition(rf_depth, model, "rf"),
            EqualsCondition(svm_gamma, model, "svm")
        ])
        self.task = MockTask(self.cs)

    def test_nan_imputation_hierarchical_spaces(self):
        opt = CARPSDynamicRFOptimizer(task=self.task, n_init=2, telemetry_path=None)
        
        # Populate history with configurations containing NaNs due to conditions
        for _ in range(2):
            trial_info = opt.ask()
            trial_val = TrialValue(cost=1.0, virtual_time=0.1)
            opt.tell(trial_info, trial_val)

        # After 2 initial trials, tell() fits the surrogate model.
        # Check that asking for candidate 3 does not raise ValueError due to NaNs
        next_trial = opt.ask()
        self.assertIsNotNone(next_trial.config)

    def test_inf_cost_fallback(self):
        opt = CARPSDynamicRFOptimizer(task=self.task, n_init=2, telemetry_path=None)
        
        # Populate history with failed runs (cost = inf)
        for _ in range(2):
            trial_info = opt.ask()
            trial_val = TrialValue(cost=float("inf"), virtual_time=0.1)
            opt.tell(trial_info, trial_val)

        # Asking for next trial should fallback to random candidate selection without crashing
        next_trial = opt.ask()
        self.assertIsNotNone(next_trial.config)

if __name__ == "__main__":
    unittest.main()

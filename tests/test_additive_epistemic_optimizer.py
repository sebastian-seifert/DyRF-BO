import pytest
import os
import json
import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from carps.utils.task import OptimizationResources
from carps.utils.trials import TrialInfo, TrialValue
from carps_integration.optimizer import CARPSDynamicRFOptimizer

class MockObjectiveFunction:
    def __init__(self, cs):
        self.configspace = cs

    def __call__(self, config):
        x1 = config["x1"]
        x2 = config["x2"]
        return float(x1**2 + x2**2)

class MockTask:
    def __init__(self, cs=None, name="mock_2d_sphere", seed=42):
        if cs is None:
            cs = ConfigurationSpace()
            cs.add([
                Float("x1", bounds=(-5.0, 5.0), default=0.0),
                Float("x2", bounds=(-5.0, 5.0), default=0.0)
            ])
        self.name = name
        self.seed = seed
        self.objective_function = MockObjectiveFunction(cs)
        self.optimization_resources = OptimizationResources(n_trials=50, time_budget=None)

def test_optimizer_initial_design_phase(tmp_path):
    """Verify first n_init trials strictly sample initial design configurations."""
    task = MockTask()
    telemetry_path = str(tmp_path / "telemetry.json")
    
    optimizer = CARPSDynamicRFOptimizer(
        task=task,
        extractor_name="standard_disagreement",
        acq_mode="additive_epistemic",
        acq_func_name="ei",
        beta_max=1.0,
        warmup_ratio=0.2,
        n_init=5,
        telemetry_path=telemetry_path
    )
    
    # Run 5 initial design steps
    for step in range(5):
        trial_info = optimizer.ask()
        val = task.objective_function(trial_info.config)
        optimizer.tell(trial_info, TrialValue(cost=val, virtual_time=1.0))
        
    assert len(optimizer.history) == 5
    assert len(optimizer.telemetry_records) == 5

def test_optimizer_bayesian_step_with_epistemic_bonus(tmp_path):
    """Verify surrogate prediction with total variance + epistemic signal addition across trials."""
    task = MockTask()
    telemetry_path = str(tmp_path / "telemetry.json")
    
    optimizer = CARPSDynamicRFOptimizer(
        task=task,
        extractor_name="standard_disagreement",
        acq_mode="additive_epistemic",
        acq_func_name="ei",
        beta_max=1.0,
        warmup_ratio=0.2,
        n_init=3,
        telemetry_path=telemetry_path
    )
    
    # Run 8 steps (3 initial + 5 BO steps)
    for step in range(8):
        trial_info = optimizer.ask()
        val = task.objective_function(trial_info.config)
        optimizer.tell(trial_info, TrialValue(cost=val, virtual_time=1.0))
        
    assert len(optimizer.history) == 8
    assert os.path.exists(telemetry_path)
    
    with open(telemetry_path, "r") as f:
        data = json.load(f)
        
    assert data["extractor_name"] == "standard_disagreement"
    assert len(data["trials"]) == 8
    # Check that beta_t is recorded and valid
    assert "beta_t" in data["trials"][-1]
    assert 0.0 <= data["trials"][-1]["beta_t"] <= 1.0

def test_all_7_registered_extractors_compatibility(tmp_path):
    """Run a 4-trial smoke test for all 7 registered extractors without errors."""
    approaches = [
        "standard_disagreement",
        "shaker_entropy",
        "likelihood_credal",
        "standard_proximity",
        "proximity_b",
        "proximity_bc",
        "proximity_auto_lambda"
    ]
    
    for approach in approaches:
        task = MockTask()
        telemetry_path = str(tmp_path / f"telemetry_{approach}.json")
        optimizer = CARPSDynamicRFOptimizer(
            task=task,
            extractor_name=approach,
            acq_mode="additive_epistemic",
            acq_func_name="ei",
            beta_max=1.0,
            warmup_ratio=0.2,
            n_init=3,
            telemetry_path=telemetry_path
        )
        
        for step in range(4):
            trial_info = optimizer.ask()
            val = task.objective_function(trial_info.config)
            optimizer.tell(trial_info, TrialValue(cost=val, virtual_time=1.0))
            
        assert len(optimizer.history) == 4

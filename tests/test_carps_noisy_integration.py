"""Extensive Integration Test Suite for CARP-S Noisy & Heteroscedastic Task Execution."""

import os
import sys
import tempfile
import subprocess
import numpy as np
import pytest
from ConfigSpace import Configuration

# Ensure DyRF-BO root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from carps.utils.trials import TrialInfo, TrialValue
from carps_integration.noisy_objective import CARPSNoisyObjectiveFunction
from noisy_benchmarks.registry import NoisyBenchmarkRegistry


# =====================================================================
# 1. CARPSNoisyObjectiveFunction Unit Tests
# =====================================================================

def test_carps_noisy_objective_instantiation():
    """Test instantiating CARPSNoisyObjectiveFunction and properties."""
    obj_fn = CARPSNoisyObjectiveFunction(problem_name="hetgp_branin_2d", seed=42)
    assert obj_fn.configspace is not None
    assert len(obj_fn.configspace.values()) == 2
    assert obj_fn.f_min is not None
    assert np.isclose(obj_fn.f_min, 0.397887, atol=1e-3)


def test_carps_noisy_objective_evaluation():
    """Test _evaluate populates TrialValue and additional_info ground truth."""
    obj_fn = CARPSNoisyObjectiveFunction(problem_name="hetgp_yuan_wahba_1d", seed=10)
    cs = obj_fn.configspace
    cfg = cs.sample_configuration()
    
    trial_info = TrialInfo(config=cfg, seed=10)
    trial_val = obj_fn.evaluate(trial_info)
    
    assert isinstance(trial_val, TrialValue)
    assert isinstance(trial_val.cost, float)
    assert trial_val.time > 0
    assert "y_true" in trial_val.additional_info
    assert "sigma_true" in trial_val.additional_info
    assert "instantaneous_regret" in trial_val.additional_info
    assert "noise_residual" in trial_val.additional_info


# =====================================================================
# 2. End-to-End CARP-S Hydra CLI Execution Tests
# =====================================================================

@pytest.mark.parametrize(
    "task_arg,optimizer_args",
    [
        (
            "+task/Noisy/hetgp=cfg_branin_2d",
            ["+optimizer/smac20=hpo", "++optimizer.acq_func_name=ei"]
        ),
        (
            "+task/Noisy/hetgp=cfg_yuan_wahba_1d",
            [
                "+optimizer=smac20_custom_uncertainty",
                "++optimizer.acq_func_name=ei",
                "++optimizer.smac_cfg.model_kwargs.uncertainty_func=proximity_bc"
            ]
        ),
        (
            "+task/Noisy/bbob=cfg_sphere_2d_gaussian",
            [
                "+optimizer=smac20_custom_uncertainty",
                "++optimizer.acq_func_name=ei",
                "++optimizer.smac_cfg.model_kwargs.uncertainty_func=shaker_entropy"
            ]
        ),
        (
            "+task/Noisy/bbob=cfg_rosenbrock_2d_cauchy",
            [
                "+optimizer=dyrf_additive_epistemic_ei",
                "++optimizer.extractor_name=likelihood_credal",
                "++optimizer.beta_max=1.0",
                "++optimizer.warmup_ratio=0.2"
            ]
        ),
    ]
)
def test_carps_cli_noisy_run_e2e(task_arg, optimizer_args, tmp_path):
    """Executes a full CARP-S run using scripts/run_carps_patched.py on noisy tasks."""
    run_dir = str(tmp_path / "carps_test_run")
    telemetry_file = str(tmp_path / "telemetry.json")
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "run_carps_patched.py"),
        "--config-dir", os.path.join(PROJECT_ROOT, "carps_integration", "configs"),
        *optimizer_args,
        task_arg,
        "task.optimization_resources.n_trials=5",
        "seed=1",
        f"outdir={run_dir}",
        f"hydra.run.dir={run_dir}",
        f"++optimizer.telemetry_path={telemetry_file}",
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    
    # Assert successful execution (exit code 0)
    assert res.returncode == 0, f"CARP-S run failed with stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    
    # Assert run directory and trial logs exist
    assert os.path.exists(run_dir)
    log_file = os.path.join(run_dir, "trial_logs.jsonl")
    assert os.path.exists(log_file), f"Log file {log_file} was not generated."
    
    # Verify trial count in log
    with open(log_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) >= 5, f"Expected at least 5 logged trials, found {len(lines)}."

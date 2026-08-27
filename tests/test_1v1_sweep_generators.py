import pytest
import os
from scripts.generate_1v1_sweep_tasks import (
    generate_all_1v1_sweeps,
    generate_single_1v1_sweep,
    SWEEP_CONFIGS
)

def test_sweep_configs_definitions():
    """Verify exact optimizer configurations for the 3 1v1 sweeps."""
    assert "disagreement" in SWEEP_CONFIGS
    assert "proximity" in SWEEP_CONFIGS
    assert "credal" in SWEEP_CONFIGS
    
    # Sweep 1: Direct Disagreement vs SMAC3 Baseline
    assert SWEEP_CONFIGS["disagreement"]["optimizer_id"] == "SMAC20_CustomUncertainty_ei_standard_disagreement"
    assert SWEEP_CONFIGS["disagreement"]["paradigm"] == "direct"
    assert SWEEP_CONFIGS["disagreement"]["baseline_id"] == "SMAC3_HPOFacade_ei"
    
    # Sweep 2: Direct Proximity vs SMAC3 Baseline
    assert SWEEP_CONFIGS["proximity"]["optimizer_id"] == "SMAC20_CustomUncertainty_ei_standard_proximity"
    assert SWEEP_CONFIGS["proximity"]["paradigm"] == "direct"
    assert SWEEP_CONFIGS["proximity"]["baseline_id"] == "SMAC3_HPOFacade_ei"
    
    # Sweep 3: Additive Credal vs SMAC3 Baseline
    assert SWEEP_CONFIGS["credal"]["optimizer_id"] == "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal"
    assert SWEEP_CONFIGS["credal"]["paradigm"] == "additive"
    assert SWEEP_CONFIGS["credal"]["baseline_id"] == "SMAC3_HPOFacade_ei"

def test_generate_single_1v1_sweep_task_count(tmp_path):
    """Verify task count for 30 seeds on 20 dev tasks is exactly 1,200 tasks."""
    out_file = str(tmp_path / "tasks_disagreement.txt")
    runs_dir = str(tmp_path / "runs_disagreement")
    
    tasks = generate_single_1v1_sweep(
        sweep_name="disagreement",
        output_file=out_file,
        runs_dir=runs_dir,
        n_seeds=30
    )
    
    # 2 optimizers * 20 tasks * 30 seeds = 1,200 tasks
    assert len(tasks) == 1200
    assert os.path.exists(out_file)
    
    with open(out_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 1200
    
    # Verify exact contents
    for line in lines:
        assert "log_dir=" not in line
        assert "telemetry_path=" in line
        assert ("optimizer_id=SMAC20_CustomUncertainty_ei_standard_disagreement" in line or
                "optimizer_id=SMAC3_HPOFacade_ei" in line)

def test_generate_all_1v1_sweeps_end_to_end(tmp_path):
    """Verify end-to-end generation of all 3 sweep task files."""
    base_dir = str(tmp_path / "sweeps")
    file_map = generate_all_1v1_sweeps(base_dir=base_dir, n_seeds=30)
    
    assert len(file_map) == 3
    for name in ["disagreement", "proximity", "credal"]:
        assert name in file_map
        task_path = file_map[name]
        assert os.path.exists(task_path)
        with open(task_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1200

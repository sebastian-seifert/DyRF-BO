import pytest
import os
from scripts.generate_ei_comparison_sweep_tasks import generate_ei_comparison_sweep_tasks
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

def test_task_count_exact_1500(tmp_path):
    """Verify 700 direct replacement + 700 additive hybrid + 100 SMAC3 baseline = 1500 tasks."""
    output_path = str(tmp_path / "ei_comparison_tasks.txt")
    lines = generate_ei_comparison_sweep_tasks(output_path=output_path)
    
    assert len(lines) == 1500
    
    direct_lines = [l for l in lines if "+optimizer=smac20_custom_uncertainty" in l]
    additive_lines = [l for l in lines if "+optimizer=dyrf_additive_epistemic_ei" in l]
    baseline_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]
    
    assert len(direct_lines) == 700
    assert len(additive_lines) == 700
    assert len(baseline_lines) == 100

def test_strictly_ei_acquisitions_only(tmp_path):
    """Verify ONLY Expected Improvement (EI) is generated, with zero PI or LCB."""
    output_path = str(tmp_path / "ei_comparison_tasks.txt")
    lines = generate_ei_comparison_sweep_tasks(output_path=output_path)
    
    for idx, line in enumerate(lines):
        assert "acq_func_name=pi" not in line.lower()
        assert "acq_func_name=lcb" not in line.lower()
        assert "_pi_" not in line
        assert "_lcb_" not in line

def test_zero_chen_occurrences(tmp_path):
    """Verify 0 instances of 'chen' exist in the generated task lines."""
    output_path = str(tmp_path / "ei_comparison_tasks.txt")
    lines = generate_ei_comparison_sweep_tasks(output_path=output_path)
    
    for idx, line in enumerate(lines):
        assert "chen" not in line.lower(), f"Found 'chen' in task line {idx}: {line}"

def test_all_20_dev_tasks_covered(tmp_path):
    """Verify all 20 dev tasks have 75 runs each (35 direct + 35 additive + 5 baseline)."""
    output_path = str(tmp_path / "ei_comparison_tasks.txt")
    lines = generate_ei_comparison_sweep_tasks(output_path=output_path)
    
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    for task in dev_tasks:
        task_name = task.split("/")[-1]
        matching = [l for l in lines if task_name in l]
        assert len(matching) == 75, f"Task {task_name} has {len(matching)} lines, expected 75"

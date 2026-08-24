import pytest
import os
import re
from scripts.generate_bbsubset_additive_hybrid_tasks import generate_bbsubset_additive_hybrid_tasks
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

def test_task_count_exact_2400(tmp_path):
    """Verify 2,100 additive hybrid + 300 SMAC3 baseline tasks = 2,400 tasks."""
    output_path = str(tmp_path / "bbsubset_additive_hybrid_tasks.txt")
    lines = generate_bbsubset_additive_hybrid_tasks(output_path=output_path, beta_max=1.0)
    
    assert len(lines) == 2400
    
    # Check split: 2100 custom additive + 300 smac3 baseline
    additive_lines = [l for l in lines if "+optimizer=dyrf_additive_epistemic" in l]
    baseline_lines = [l for l in lines if "+optimizer/smac20=hpo" in l]
    
    assert len(additive_lines) == 2100
    assert len(baseline_lines) == 300

def test_zero_chen_occurrences(tmp_path):
    """Verify 0 instances of 'chen' or 'chen_variance' exist in the generated task lines."""
    output_path = str(tmp_path / "bbsubset_additive_hybrid_tasks.txt")
    lines = generate_bbsubset_additive_hybrid_tasks(output_path=output_path, beta_max=1.0)
    
    for idx, line in enumerate(lines):
        assert "chen" not in line.lower(), f"Found 'chen' in task line {idx}: {line}"

def test_all_20_bbsubset_dev_tasks_present(tmp_path):
    """Verify all 20 official dev tasks are covered across EI, PI, and LCB."""
    output_path = str(tmp_path / "bbsubset_additive_hybrid_tasks.txt")
    lines = generate_bbsubset_additive_hybrid_tasks(output_path=output_path, beta_max=1.0)
    
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    assert len(dev_tasks) == 20
    
    for task in dev_tasks:
        task_name = task.split("/")[-1]
        matching_lines = [l for l in lines if task_name in l]
        # Each task has: (3 acqs * 7 approaches * 5 seeds) + (3 acqs * 1 baseline * 5 seeds) = 105 + 15 = 120 lines
        assert len(matching_lines) == 120, f"Task {task_name} has {len(matching_lines)} lines, expected 120"

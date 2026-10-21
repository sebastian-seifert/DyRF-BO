"""TDD Tests for CARP-S BBSubset Dev Big Comparison Task Generator.

Verifies:
1. Exact total task count is 6,600 (20 tasks * 11 optimizers * 30 seeds) or 5,940 (18 tasks * 11 optimizers * 30 seeds).
2. Exactly 2 split parts.
3. Part 1 contains 6 optimizers:
   - SMAC3_HPOFacade_lcb
   - SMAC3_HPOFacade_ei
   - SMAC20_Entropy_LCB
   - SMAC20_ProximityA_LCB
   - SMAC20_ProximityA_LCB_CV
   - SMAC20_ProximityAC_LCB
4. Part 2 contains 5 optimizers:
   - SMAC20_ProximityAC_LCB_CV
   - SMAC20_ProximityB_LCB
   - SMAC20_ProximityB_LCB_CV
   - SMAC20_ProximityBC_LCB
   - SMAC20_ProximityBC_LCB_CV
5. Telemetry paths and seeds are strictly paired across all optimizers.
6. Dev tasks point to dev benchmark tasks (+task=subselection/blackbox/dev/...) and NOT test tasks.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_bbsubset_dev_big_comparison_tasks import (
    generate_bbsubset_dev_big_comparison_tasks,
    OPTIMIZERS_PART1,
    OPTIMIZERS_PART2,
    ALL_OPTIMIZERS,
)
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry


def test_registry_dev_tasks_count():
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    assert len(dev_tasks) == 20
    for task in dev_tasks:
        assert "/dev/" in task
        assert "/test/" not in task


def test_optimizer_splits_balanced():
    total_opts = len(ALL_OPTIMIZERS)
    assert total_opts == 11
    assert len(OPTIMIZERS_PART1) + len(OPTIMIZERS_PART2) == 11
    ids_p1 = {opt["id"] for opt in OPTIMIZERS_PART1}
    ids_p2 = {opt["id"] for opt in OPTIMIZERS_PART2}
    assert ids_p1.isdisjoint(ids_p2)


def test_generate_tasks_dev_all():
    p1_lines, p2_lines = generate_bbsubset_dev_big_comparison_tasks(
        seeds=30,
        trials=100,
        output_dir=None,
        exclude_nas=False,
    )
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()

    expected_p1 = len(dev_tasks) * len(OPTIMIZERS_PART1) * 30
    expected_p2 = len(dev_tasks) * len(OPTIMIZERS_PART2) * 30

    assert len(p1_lines) == expected_p1
    assert len(p2_lines) == expected_p2
    assert len(p1_lines) + len(p2_lines) == 6600

    # Assert proper flags in generated commands
    sample_cmd = p1_lines[0]
    assert "--config-dir carps_integration/configs" in sample_cmd
    assert "+task=subselection/blackbox/dev/" in sample_cmd
    assert "task.optimization_resources.n_trials=100" in sample_cmd
    assert "baserundir=runs/sweep_bbsubset_dev_big_comparison" in sample_cmd
    assert "++optimizer.telemetry_path=" in sample_cmd


def test_generate_tasks_exclude_nas():
    p1_lines, p2_lines = generate_bbsubset_dev_big_comparison_tasks(
        seeds=30,
        trials=100,
        output_dir=None,
        exclude_nas=True,
    )
    working_tasks = CarpsBBSubsetRegistry.get_working_dev_tasks(exclude_nas=True)
    assert len(working_tasks) == 18

    expected_p1 = len(working_tasks) * len(OPTIMIZERS_PART1) * 30
    expected_p2 = len(working_tasks) * len(OPTIMIZERS_PART2) * 30

    assert len(p1_lines) == expected_p1
    assert len(p2_lines) == expected_p2
    assert len(p1_lines) + len(p2_lines) == 5940

    for line in p1_lines + p2_lines:
        assert "tabular_nas" not in line

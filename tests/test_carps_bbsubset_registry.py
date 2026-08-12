import os
import pytest
from scripts.carps_bbsubset_registry import CarpsBBSubsetRegistry

def test_get_dev_tasks_count():
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    assert isinstance(dev_tasks, list)
    assert len(dev_tasks) == 20, f"Expected 20 dev tasks, found {len(dev_tasks)}"

def test_dev_task_yaml_files_exist():
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_dir = os.path.join(base_dir, "carps_integration", "configs")

    for task_str in dev_tasks:
        # Task string format: +task=subselection/blackbox/dev/subset_xyz
        assert task_str.startswith("+task=subselection/blackbox/dev/"), f"Invalid task format: {task_str}"
        task_rel = task_str.split("+task=")[-1] + ".yaml"
        yaml_path = os.path.join(config_dir, "task", task_rel)
        assert os.path.exists(yaml_path), f"Missing task YAML file: {yaml_path}"

def test_no_test_set_exposure():
    # Ensure CarpsBBSubsetRegistry does not expose any test set tasks
    dev_tasks = CarpsBBSubsetRegistry.get_dev_tasks()
    for task_str in dev_tasks:
        assert "/test/" not in task_str, f"Test set task exposed in dev registry: {task_str}"

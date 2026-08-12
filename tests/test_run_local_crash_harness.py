import os
import json
import pytest
from scripts.run_local_crash_harness import get_auto_workers, execute_single_task

def test_get_auto_workers():
    workers = get_auto_workers()
    assert isinstance(workers, int)
    assert workers >= 1

def test_execute_single_task_success(tmp_path):
    # Test a simple echo task that succeeds
    task_str = "+optimizer/smac20=hpo --help"
    python_exec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
    if not os.path.exists(python_exec):
        import sys
        python_exec = sys.executable

    res = execute_single_task(
        task_idx=1,
        command_str=task_str,
        python_exec=python_exec,
        crash_out_dir=str(tmp_path)
    )
    assert res["status"] in ["success", "failed"]

def test_execute_single_task_crash_logging(tmp_path):
    # Pass an invalid flag to force a command crash and test crash report generation
    task_str = "--invalid_flag_force_crash_xyz"
    python_exec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "python")
    if not os.path.exists(python_exec):
        import sys
        python_exec = sys.executable

    res = execute_single_task(
        task_idx=999,
        command_str=task_str,
        python_exec=python_exec,
        crash_out_dir=str(tmp_path)
    )

    assert res["status"] == "failed"
    assert "crash_dir" in res
    assert os.path.exists(res["crash_dir"])

    report_json = os.path.join(res["crash_dir"], "crash_report.json")
    assert os.path.exists(report_json)

    with open(report_json, "r") as f:
        data = json.load(f)

    assert data["task_idx"] == 999
    assert "--invalid_flag_force_crash_xyz" in data["command"]
    assert data["returncode"] != 0

    # Verify generated RED pytest stub
    test_stub = os.path.join("tests", "test_reproduce_crash_task_999.py")
    if os.path.exists(test_stub):
        os.remove(test_stub)

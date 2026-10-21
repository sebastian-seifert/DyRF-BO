"""TDD Tests for CARP-S BBSubset Dev Big Comparison SLURM Scripts.

Verifies:
1. Shell and sbatch scripts exist and have proper executable permissions.
2. Shell syntax checks pass (bash -n).
3. Dry-run execution generates tasks and produces compliant chunked submission commands without contacting SLURM daemon.
"""

import os
import stat
import subprocess
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_scripts_exist_and_executable():
    sh_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_all.sh")
    sbatch_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_array.sbatch")

    assert os.path.exists(sh_script), f"{sh_script} does not exist"
    assert os.path.exists(sbatch_script), f"{sbatch_script} does not exist"

    mode = os.stat(sh_script).st_mode
    assert bool(mode & stat.S_IXUSR), f"{sh_script} is not executable"


def test_shell_syntax():
    sh_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_all.sh")
    sbatch_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_array.sbatch")

    res_sh = subprocess.run(["bash", "-n", sh_script], capture_output=True, text=True)
    assert res_sh.returncode == 0, f"Syntax error in {sh_script}: {res_sh.stderr}"

    res_sbatch = subprocess.run(["bash", "-n", sbatch_script], capture_output=True, text=True)
    assert res_sbatch.returncode == 0, f"Syntax error in {sbatch_script}: {res_sbatch.stderr}"


def test_dry_run_execution():
    sh_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_all.sh")

    result = subprocess.run(
        ["bash", sh_script, "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry run failed: {result.stderr}"
    assert "Dry run completed successfully" in result.stdout
    assert "[DRY-RUN] sbatch" in result.stdout
    assert "bb_dev_part" in result.stdout


def test_dry_run_part_flag_separate_arg():
    sh_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_all.sh")

    result = subprocess.run(
        ["bash", sh_script, "--dry-run", "--part", "1"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry run failed: {result.stderr}"
    assert "Target Execution: 1" in result.stdout
    assert "--- Submitting Part 1" in result.stdout
    assert "--- Submitting Part 2" not in result.stdout


def test_dry_run_exclude_nas_regeneration():
    sh_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_all.sh")

    # Run with --exclude-nas
    result = subprocess.run(
        ["bash", sh_script, "--dry-run", "--exclude-nas"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry run failed: {result.stderr}"
    assert "Tasks in Part 1 (Baselines + Entropy + Prox A & AC): 3240" in result.stdout
    assert "Tasks in Part 2 (Prox B & BC):                       2700" in result.stdout

    # Then run without --exclude-nas to verify it regenerates the full 20 tasks
    result_full = subprocess.run(
        ["bash", sh_script, "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result_full.returncode == 0, f"Dry run failed: {result_full.stderr}"
    assert "Tasks in Part 1 (Baselines + Entropy + Prox A & AC): 3600" in result_full.stdout
    assert "Tasks in Part 2 (Prox B & BC):                       3000" in result_full.stdout


def test_sbatch_matches_bbob_dim2_conda_setup():
    sbatch_script = os.path.join(PROJECT_ROOT, "scripts", "submit_bbsubset_dev_big_comparison_array.sbatch")
    with open(sbatch_script, "r") as f:
        content = f.read()

    # Check for robust profile.d/conda.sh checks
    assert "miniforge3/etc/profile.d/conda.sh" in content
    assert "miniconda3/etc/profile.d/conda.sh" in content
    assert "/bigwork/nhwpseis/miniforge3/etc/profile.d/conda.sh" in content

    # Check for explicit PYTHON_BIN resolution matching bbob_dim2
    assert 'PYTHON_BIN="/bigwork/nhwpseis/.conda/envs/dyrf/bin/python"' in content
    assert 'export PATH="$(dirname "$PYTHON_BIN"):$PATH"' in content
    assert '"$PYTHON_BIN" scripts/run_carps_patched.py $TASK_ARGS' in content



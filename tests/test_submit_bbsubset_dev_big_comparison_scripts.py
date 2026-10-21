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

"""Unit tests for Extrapolation UQ cluster submission scripts and local multi-process runner."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_extrapolation_sweep_local import (
    build_parser,
    compute_chunks,
    execute_command,
    main as local_runner_main,
    run_local_sweep,
)


SCRIPTS_DIR = REPO_ROOT / "scripts"
SBATCH_SCRIPT = SCRIPTS_DIR / "submit_extrapolation_sweep_array.sbatch"
SUBMIT_ALL_SCRIPT = SCRIPTS_DIR / "submit_extrapolation_sweep_all.sh"
SUBMIT_SLURM_SCRIPT = SCRIPTS_DIR / "submit_extrapolation_sweep_slurm.sh"
LOCAL_RUNNER_SCRIPT = SCRIPTS_DIR / "run_extrapolation_sweep_local.py"

TARGET_SCRIPTS = [
    SBATCH_SCRIPT,
    SUBMIT_ALL_SCRIPT,
    SUBMIT_SLURM_SCRIPT,
    LOCAL_RUNNER_SCRIPT,
]


class TestScriptFilesIntegrity:
    """Verify script files exist and have appropriate executable permissions."""

    @pytest.mark.parametrize("script_path", TARGET_SCRIPTS)
    def test_script_exists(self, script_path: Path):
        assert script_path.exists(), f"Target script does not exist: {script_path}"
        assert script_path.is_file(), f"Target script is not a file: {script_path}"

    @pytest.mark.parametrize("script_path", TARGET_SCRIPTS)
    def test_script_is_executable(self, script_path: Path):
        assert os.access(script_path, os.X_OK), (
            f"Script {script_path.name} is missing executable bit (chmod +x needed)."
        )

    @pytest.mark.parametrize(
        "script_path",
        [SBATCH_SCRIPT, SUBMIT_ALL_SCRIPT, SUBMIT_SLURM_SCRIPT],
    )
    def test_bash_syntax_check(self, script_path: Path):
        """Ensure shell scripts pass syntax validation via bash -n."""
        proc = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Bash syntax error in {script_path.name}:\n{proc.stderr}"


class TestSbatchScriptDirectives:
    """Verify LUIS SLURM sbatch resource directives and environment setup."""

    def test_sbatch_directives_and_thread_pinning(self):
        content = SBATCH_SCRIPT.read_text(encoding="utf-8")

        # Partition and job name
        assert "#SBATCH -p ai" in content
        assert "#SBATCH --job-name=extrap_uq" in content

        # Log paths
        assert "#SBATCH --output=results/extrapolation_uq/logs/array_%A_%a.log" in content
        assert "#SBATCH --error=results/extrapolation_uq/logs/array_%A_%a.err" in content

        # Resource limits
        assert "#SBATCH --cpus-per-task=1" in content
        assert "#SBATCH --mem=4G" in content
        assert "#SBATCH --time=00:30:00" in content

        # Thread pinning (prevent multi-thread BLAS contention)
        assert "export OPENBLAS_NUM_THREADS=1" in content
        assert "export MKL_NUM_THREADS=1" in content
        assert "export OMP_NUM_THREADS=1" in content
        assert "export NUMEXPR_NUM_THREADS=1" in content

        # Python / Conda / Directory setup
        assert 'cd "$SLURM_SUBMIT_DIR"' in content
        assert "conda activate dyrf" in content
        assert "results/extrapolation_sweep_tasks.txt" in content
        assert "SLURM_ARRAY_TASK_ID" in content
        assert 'sed -n "${SLURM_ARRAY_TASK_ID}p"' in content or 'sed -n "$SLURM_ARRAY_TASK_ID' in content


class TestSubmitSlurmSafetyWrapper:
    """Verify safety wrapper blocks direct submission under Slurm allocation."""

    def test_blocks_execution_when_slurm_job_id_is_set(self):
        env = os.environ.copy()
        env["SLURM_JOB_ID"] = "999999"
        proc = subprocess.run(
            ["bash", str(SUBMIT_SLURM_SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "login node" in proc.stderr.lower() or "login node" in proc.stdout.lower() or "sbatch" in proc.stderr.lower()

    def test_delegates_to_submit_all_on_login_node(self, tmp_path):
        """When not in slurm allocation, delegates to submit_all script."""
        log_file = tmp_path / "sbatch_calls.log"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        mock_sbatch = bin_dir / "sbatch"
        mock_sbatch.write_text(f"""#!/bin/bash
echo "$@" >> "{log_file}"
echo 123456
""")
        mock_sbatch.chmod(mock_sbatch.stat().st_mode | stat.S_IEXEC)

        env = os.environ.copy()
        env.pop("SLURM_JOB_ID", None)
        env["PATH"] = f"{bin_dir}:{env['PATH']}"

        # Temporary task file isolation/cleanup in results directory
        dummy_task_file = REPO_ROOT / "results" / "extrapolation_sweep_tasks.txt"
        backup_content = None
        if dummy_task_file.exists():
            backup_content = dummy_task_file.read_text(encoding="utf-8")

        dummy_task_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Two tasks so sbatch receives --array=1-2%25
            dummy_task_file.write_text("cmd 1\ncmd 2\n")

            proc = subprocess.run(
                ["bash", str(SUBMIT_SLURM_SCRIPT)],
                env=env,
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert proc.returncode == 0, f"Script failed:\n{proc.stderr}\n{proc.stdout}"

            calls = log_file.read_text().splitlines()
            assert len(calls) == 1
            assert "--array=1-2%25" in calls[0]
        finally:
            if backup_content is not None:
                dummy_task_file.write_text(backup_content)
            elif dummy_task_file.exists():
                dummy_task_file.unlink()


class TestChunkingLogic:
    """Verify chunking arithmetic complies with LUIS cluster limits."""

    @pytest.mark.parametrize(
        "total_tasks, chunk_size, expected_chunks",
        [
            (1920, 200, [(1, 200), (201, 400), (401, 600), (601, 800), (801, 1000),
                         (1001, 1200), (1201, 1400), (1401, 1600), (1601, 1800), (1801, 1920)]),
            (4, 200, [(1, 4)]),
            (200, 200, [(1, 200)]),
            (201, 200, [(1, 200), (201, 201)]),
            (1, 200, [(1, 1)]),
            (50, 200, [(1, 50)]),
        ],
    )
    def test_compute_chunks(self, total_tasks: int, chunk_size: int, expected_chunks: list[tuple[int, int]]):
        chunks = compute_chunks(total_tasks, chunk_size=chunk_size)
        assert chunks == expected_chunks

        # Verify all chunk sizes are <= 300 (LUIS array limit)
        for start, end in chunks:
            size = end - start + 1
            assert size <= 300
            assert size <= chunk_size

    def test_compute_chunks_invalid_tasks(self):
        assert compute_chunks(0, chunk_size=200) == []
        assert compute_chunks(-5, chunk_size=200) == []

    def test_submit_all_script_chunk_execution(self, tmp_path):
        """Test submit_extrapolation_sweep_all.sh with mock sbatch."""
        # Create a mock sbatch in PATH that logs each call
        log_file = tmp_path / "sbatch_calls.log"
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        mock_sbatch = bin_dir / "sbatch"
        mock_sbatch.write_text(f"""#!/bin/bash
echo "$@" >> "{log_file}"
echo 98765
""")
        mock_sbatch.chmod(mock_sbatch.stat().st_mode | stat.S_IEXEC)

        # Create dummy task file
        dummy_task_file = REPO_ROOT / "results" / "extrapolation_sweep_tasks.txt"
        backup_content = None
        if dummy_task_file.exists():
            backup_content = dummy_task_file.read_text(encoding="utf-8")

        dummy_task_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Write 450 tasks
            dummy_task_file.write_text("\n".join([f"echo task {i}" for i in range(1, 451)]) + "\n")

            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            proc = subprocess.run(
                ["bash", str(SUBMIT_ALL_SCRIPT)],
                env=env,
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
            )
            assert proc.returncode == 0, f"Script failed:\n{proc.stderr}\n{proc.stdout}"

            # Verify sbatch was called 3 times: 1-200, 201-400, 401-450
            calls = log_file.read_text().splitlines()
            assert len(calls) == 3
            assert "--array=1-200%25" in calls[0]
            assert "--array=201-400%25" in calls[1]
            assert "--array=401-450%25" in calls[2]
        finally:
            if backup_content is not None:
                dummy_task_file.write_text(backup_content)
            elif dummy_task_file.exists():
                dummy_task_file.unlink()


class TestLocalSweepRunner:
    """Verify run_extrapolation_sweep_local.py multi-processing and CLI behavior."""

    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.task_file == "results/extrapolation_sweep_tasks.txt"
        assert args.workers == max(1, (os.cpu_count() or 1) - 1)
        assert args.limit is None
        assert args.dry_run is False

    def test_parser_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--task-file", "my_tasks.txt",
            "--workers", "4",
            "--limit", "10",
            "--dry-run",
        ])
        assert args.task_file == "my_tasks.txt"
        assert args.workers == 4
        assert args.limit == 10
        assert args.dry_run is True

    def test_dry_run_execution(self, tmp_path):
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("echo 'task 1'\necho 'task 2'\necho 'task 3'\n")

        summary = run_local_sweep(
            task_file=task_file,
            workers=2,
            dry_run=True,
        )
        assert summary["total_tasks"] == 3
        assert summary["executed_tasks"] == 3
        assert summary["succeeded_tasks"] == 3
        assert summary["failed_tasks"] == 0
        assert all(res["status"] == "dry_run" for res in summary["results"])

    def test_multiprocess_execution_success(self, tmp_path):
        out_file1 = tmp_path / "out1.txt"
        out_file2 = tmp_path / "out2.txt"

        task_file = tmp_path / "tasks.txt"
        task_file.write_text(
            f"{sys.executable} -c \"import pathlib; pathlib.Path(r'{out_file1}').write_text('done1')\"\n"
            f"{sys.executable} -c \"import pathlib; pathlib.Path(r'{out_file2}').write_text('done2')\"\n"
        )

        summary = run_local_sweep(
            task_file=task_file,
            workers=2,
            dry_run=False,
        )
        assert summary["total_tasks"] == 2
        assert summary["succeeded_tasks"] == 2
        assert summary["failed_tasks"] == 0
        assert out_file1.read_text() == "done1"
        assert out_file2.read_text() == "done2"

    def test_limit_tasks(self, tmp_path):
        task_file = tmp_path / "tasks.txt"
        task_file.write_text(
            "echo 1\n"
            "echo 2\n"
            "echo 3\n"
            "echo 4\n"
        )
        summary = run_local_sweep(
            task_file=task_file,
            workers=2,
            limit=2,
            dry_run=True,
        )
        assert summary["total_tasks"] == 4
        assert summary["executed_tasks"] == 2
        assert len(summary["results"]) == 2

    def test_execution_failure_handling(self, tmp_path):
        task_file = tmp_path / "tasks.txt"
        task_file.write_text(
            "echo success\n"
            f"{sys.executable} -c \"import sys; sys.exit(42)\"\n"
        )
        summary = run_local_sweep(
            task_file=task_file,
            workers=2,
            dry_run=False,
        )
        assert summary["total_tasks"] == 2
        assert summary["succeeded_tasks"] == 1
        assert summary["failed_tasks"] == 1

        failed = [res for res in summary["results"] if res["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["exit_code"] == 42

    def test_cli_entrypoint_main(self, tmp_path):
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("echo 'pilot'\n")

        ret = local_runner_main([
            "--task-file", str(task_file),
            "--dry-run",
        ])
        assert ret == 0

    def test_cli_entrypoint_failure_exit_code(self, tmp_path):
        task_file = tmp_path / "tasks.txt"
        task_file.write_text(f"{sys.executable} -c \"import sys; sys.exit(1)\"\n")

        ret = local_runner_main([
            "--task-file", str(task_file),
        ])
        assert ret != 0

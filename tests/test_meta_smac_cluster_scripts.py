"""Unit tests for Meta SMAC cluster submission scripts for Methods B, AC, and BC.

Verifies:
1. Script file existence, file type, and executable permissions (chmod +x).
2. Bash syntax validation via 'bash -n'.
3. Method-specific configuration, directory conventions, and unbuffered python execution.
4. CLI argument passthrough ("$@") using mock python interpreter execution.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

SCRIPT_CONFIGS = {
    "b": {
        "file": SCRIPTS_DIR / "submit_meta_smac_b.sh",
        "method": "b",
        "output_dir": "results/meta_smac_proximity_b_hpo",
        "baserundir": "runs/meta_smac_proximity_b_hpo",
    },
    "ac": {
        "file": SCRIPTS_DIR / "submit_meta_smac_ac.sh",
        "method": "ac",
        "output_dir": "results/meta_smac_proximity_ac_hpo",
        "baserundir": "runs/meta_smac_proximity_ac_hpo",
    },
    "bc": {
        "file": SCRIPTS_DIR / "submit_meta_smac_bc.sh",
        "method": "bc",
        "output_dir": "results/meta_smac_proximity_bc_hpo",
        "baserundir": "runs/meta_smac_proximity_bc_hpo",
    },
}

ALL_SCRIPT_PATHS = [cfg["file"] for cfg in SCRIPT_CONFIGS.values()]


class TestScriptFilesIntegrity:
    """Verify cluster submission scripts exist and have appropriate executable permissions."""

    @pytest.mark.parametrize("script_path", ALL_SCRIPT_PATHS)
    def test_script_exists(self, script_path: Path):
        assert script_path.exists(), f"Target script does not exist: {script_path}"
        assert script_path.is_file(), f"Target script is not a regular file: {script_path}"

    @pytest.mark.parametrize("script_path", ALL_SCRIPT_PATHS)
    def test_script_is_executable(self, script_path: Path):
        assert os.access(script_path, os.X_OK), (
            f"Script {script_path.name} is missing executable bit (chmod +x needed)."
        )

    @pytest.mark.parametrize("script_path", ALL_SCRIPT_PATHS)
    def test_bash_syntax_check(self, script_path: Path):
        """Ensure shell script passes syntax validation via bash -n."""
        proc = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Bash syntax error in {script_path.name}:\n{proc.stderr}"


class TestScriptContentAndArchitecture:
    """Verify structural conventions, directory setups, and unbuffered python commands."""

    @pytest.mark.parametrize("method,cfg", SCRIPT_CONFIGS.items())
    def test_shebang_and_safety_settings(self, method: str, cfg: Dict):
        script_path = cfg["file"]
        content = script_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        assert lines[0] == "#!/bin/bash", f"{script_path.name} must start with #!/bin/bash"
        assert "set -e" in content, f"{script_path.name} must specify set -e"

    @pytest.mark.parametrize("method,cfg", SCRIPT_CONFIGS.items())
    def test_method_and_directory_parameters(self, method: str, cfg: Dict):
        content = cfg["file"].read_text(encoding="utf-8")

        assert f"--method {method}" in content, (
            f"{cfg['file'].name} must include '--method {method}'"
        )
        assert f"--output-dir {cfg['output_dir']}" in content, (
            f"{cfg['file'].name} must specify output dir {cfg['output_dir']}"
        )
        assert f"--baserundir {cfg['baserundir']}" in content, (
            f"{cfg['file'].name} must specify baserundir {cfg['baserundir']}"
        )

    @pytest.mark.parametrize("method,cfg", SCRIPT_CONFIGS.items())
    def test_fixed_invocation_parameters(self, method: str, cfg: Dict):
        content = cfg["file"].read_text(encoding="utf-8")

        assert "--start-iteration 1" in content
        assert "--end-iteration 100" in content
        assert "--seeds 5" in content
        assert "--trials 100" in content
        assert '"$@"' in content

    @pytest.mark.parametrize("method,cfg", SCRIPT_CONFIGS.items())
    def test_safe_directory_creation_and_unbuffered_python(self, method: str, cfg: Dict):
        content = cfg["file"].read_text(encoding="utf-8")

        # Verify safe directory creation
        assert f"mkdir -p {cfg['output_dir']}" in content or f"mkdir -p {cfg['output_dir']}/logs" in content
        assert f"mkdir -p {cfg['baserundir']}" in content or f"mkdir -p {cfg['output_dir']}" in content

        # Verify unbuffered python execution
        assert "-u scripts/run_meta_smac_proximity_hpo.py" in content or "PYTHONUNBUFFERED" in content
        assert "scripts/run_meta_smac_proximity_hpo.py" in content


class TestScriptInvocationWithMockPython:
    """Verify actual shell script invocation and CLI argument passthrough via mock python."""

    @pytest.mark.parametrize("method,cfg", SCRIPT_CONFIGS.items())
    def test_script_execution_and_arg_passthrough(self, method: str, cfg: Dict, tmp_path: Path):
        mock_bin_dir = tmp_path / "bin"
        mock_bin_dir.mkdir(parents=True, exist_ok=True)
        record_file = tmp_path / "mock_args.json"

        # Create mock python script that logs sys.argv using real Python interpreter
        mock_python = mock_bin_dir / "python"
        mock_python.write_text(
            f"""#!{sys.executable}
import json
import sys

with open("{record_file}", "w") as f:
    json.dump({{"argv": sys.argv[1:]}}, f)
sys.exit(0)
""",
            encoding="utf-8",
        )
        mock_python.chmod(mock_python.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Create symlinks for python3 as well
        mock_python3 = mock_bin_dir / "python3"
        mock_python3.symlink_to(mock_python)

        # Run script with custom extra args to verify "$@" passthrough
        env = os.environ.copy()
        env["PATH"] = f"{mock_bin_dir}:{env.get('PATH', '')}"
        env["CONDA_PREFIX"] = str(tmp_path)
        env["PYTHON_BIN"] = str(mock_python)
        extra_args = ["--resume", "--end-iteration", "50"]
        proc = subprocess.run(
            ["bash", str(cfg["file"]), *extra_args],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, f"Script failed with output:\n{proc.stdout}\n{proc.stderr}"
        assert record_file.exists(), "Mock python was not invoked by script"

        import json
        with open(record_file, "r") as f:
            data = json.load(f)
        argv = data["argv"]

        # If -u was passed as python flag, sys.argv[1:] might start after or include script path
        script_arg_idx = -1
        for idx, arg in enumerate(argv):
            if "run_meta_smac_proximity_hpo.py" in arg:
                script_arg_idx = idx
                break
        assert script_arg_idx != -1, f"run_meta_smac_proximity_hpo.py not in argv: {argv}"

        cmd_args = argv[script_arg_idx + 1 :]
        assert "--method" in cmd_args
        assert cmd_args[cmd_args.index("--method") + 1] == method
        assert "--output-dir" in cmd_args
        assert cmd_args[cmd_args.index("--output-dir") + 1] == cfg["output_dir"]
        assert "--baserundir" in cmd_args
        assert cmd_args[cmd_args.index("--baserundir") + 1] == cfg["baserundir"]
        assert "--seeds" in cmd_args
        assert cmd_args[cmd_args.index("--seeds") + 1] == "5"
        assert "--trials" in cmd_args
        assert cmd_args[cmd_args.index("--trials") + 1] == "100"

        # Check passthrough arguments
        assert "--resume" in cmd_args
        assert "50" in cmd_args

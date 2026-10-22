#!/usr/bin/env python3
"""Tests for BBOB 128D Task Configurations and Registry Extension."""

import os
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.carps_bbob_full_registry import CarpsBBOBFullRegistry


def test_registry_includes_128_dimension():
    assert 128 in CarpsBBOBFullRegistry.EXTENDED_DIMENSIONS
    dim128_tasks = CarpsBBOBFullRegistry.get_tasks_by_dimension(128)
    assert len(dim128_tasks) == 72
    # Verify format
    for t in dim128_tasks:
        assert t.startswith("+task=BBOB/cfg_128_")
    
    # Check sample task
    sample = "+task=BBOB/cfg_128_1_0"
    assert sample in dim128_tasks
    info = CarpsBBOBFullRegistry.get_task_info(sample)
    assert info.dimension == 128
    assert info.fid == 1
    assert info.iid == 0


def test_bbob128_config_files_exist():
    config_dir = REPO_ROOT / "carps_integration" / "configs" / "task" / "BBOB"
    assert config_dir.is_dir()

    for fid in range(1, 25):
        for inst in (0, 1, 2):
            cfg_file = config_dir / f"cfg_128_{fid}_{inst}.yaml"
            assert cfg_file.is_file(), f"Missing config file: {cfg_file}"
            content = cfg_file.read_text()
            assert "name: bbob/128/" in content
            assert "dimension: 128" in content
            assert "name: x127" in content

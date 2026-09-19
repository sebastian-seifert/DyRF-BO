#!/usr/bin/env python3
"""Tests for Multi-Suite Fast Gatherer."""

import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.gather_multisuite_proximity_results import (
    parse_task_path,
    extract_trial_logs,
)


class TestMultiSuiteFastGatherer:
    def test_parse_task_path(self):
        # Format: runs/sweep_{suite}_proximity/{optimizer_id}/{benchmark}/{task}/{seed}/
        p1 = Path("runs/sweep_yahpo_rbv2_ranger_proximity/SMAC20_ProximityLCB/YAHPO/yahpo_rbv2_ranger_1040_None/1")
        info = parse_task_path(p1)
        assert info["suite"] == "yahpo_rbv2_ranger"
        assert info["optimizer_id"] == "SMAC20_ProximityLCB"
        assert info["benchmark"] == "YAHPO"
        assert info["task"] == "yahpo_rbv2_ranger_1040_None"
        assert info["seed"] == 1

    def test_extract_trial_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            trial_file = run_dir / "trial_logs.jsonl"
            lines = [
                {"trial": 1, "cost": 0.5, "loss": 0.5},
                {"trial": 2, "cost": 0.3, "loss": 0.3},
                {"trial": 3, "cost": 0.4, "loss": 0.4},
            ]
            with open(trial_file, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")

            records = extract_trial_logs(trial_file, min_trials=1)
            assert len(records) == 3
            assert records[-1]["trial"] == 3
            assert records[-1]["min_cost"] == 0.3

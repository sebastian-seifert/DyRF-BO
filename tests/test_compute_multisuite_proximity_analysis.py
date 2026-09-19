#!/usr/bin/env python3
"""Tests for Multi-Suite Statistical Analysis and Scorecard Generator."""

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

from scripts.compute_multisuite_proximity_analysis import compute_paired_statistics


class TestMultiSuiteAnalysis:
    def test_compute_paired_statistics(self):
        records = [
            # Task 1: Proposed wins
            {"task": "task_1", "seed": 1, "optimizer_id": "SMAC20_ProximityLCB", "final_loss": 0.10},
            {"task": "task_1", "seed": 1, "optimizer_id": "SMAC3_HPOFacade_lcb", "final_loss": 0.20},
            # Task 2: Proposed wins
            {"task": "task_2", "seed": 1, "optimizer_id": "SMAC20_ProximityLCB", "final_loss": 0.05},
            {"task": "task_2", "seed": 1, "optimizer_id": "SMAC3_HPOFacade_lcb", "final_loss": 0.15},
            # Task 3: Tie
            {"task": "task_3", "seed": 1, "optimizer_id": "SMAC20_ProximityLCB", "final_loss": 0.30},
            {"task": "task_3", "seed": 1, "optimizer_id": "SMAC3_HPOFacade_lcb", "final_loss": 0.30},
        ]
        stats = compute_paired_statistics(records)
        assert stats["total_pairs"] == 3
        assert stats["wins_proposed"] == 2
        assert stats["wins_baseline"] == 0
        assert stats["ties"] == 1
        assert stats["win_rate_proposed"] == pytest.approx(2 / 3, 0.01)

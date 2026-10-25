"""Unit and integration tests for the batch scorecard analysis runner."""

import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure DyRF-BO project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_batch_scorecard_analysis import (
    resolve_batch_output_dir,
    process_single_log_file,
    run_batch_scorecards,
)


def _create_sample_sweep_df() -> pd.DataFrame:
    records = []
    tasks = [f"task_{i}" for i in range(1, 6)]
    opts = [
        "SMAC3_HPOFacade_lcb",
        "SMAC3_HPOFacade_ei",
        "Candidate_A",
        "Candidate_B",
    ]
    for task_id in tasks:
        for opt in opts:
            for seed in [1, 2]:
                for trial in [10, 100]:
                    if opt == "Candidate_A":
                        cost = 0.10 if trial == 100 else 0.50
                    elif opt == "Candidate_B":
                        cost = 0.20 if trial == 100 else 0.60
                    elif opt == "SMAC3_HPOFacade_ei":
                        cost = 0.30 if trial == 100 else 0.70
                    else:  # SMAC3_HPOFacade_lcb
                        cost = 0.40 if trial == 100 else 0.80

                    records.append({
                        "task_id": task_id,
                        "optimizer_id": opt,
                        "seed": seed,
                        "trial": trial,
                        "trial_value__cost_inc": cost,
                    })
    return pd.DataFrame(records)


def test_resolve_and_process_dir(tmp_path: Path):
    sub1 = tmp_path / "run1"
    sub1.mkdir(parents=True)
    parquet_file = sub1 / "logs.parquet"
    df = _create_sample_sweep_df()
    df.to_parquet(parquet_file)

    out_dir = resolve_batch_output_dir(parquet_file, subfolder="new_table")
    assert out_dir == sub1 / "new_table"

    res = process_single_log_file(
        input_path=parquet_file,
        subfolder="new_table",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        tie_tolerance=1e-6,
    )

    assert res["status"] == "success"
    assert out_dir.exists()
    assert (out_dir / "scorecard.csv").exists()
    assert (out_dir / "scorecard.md").exists()

    scorecard_df = pd.read_csv(out_dir / "scorecard.csv")
    assert "SMAC3_HPOFacade_lcb" not in scorecard_df["optimizer_id"].values
    assert "Candidate_A" in scorecard_df["optimizer_id"].values
    assert set(scorecard_df.columns) == {
        "optimizer_id",
        "ctrl_lcb_win_rate",
        "ctrl_lcb_wilcoxon_adj_p",
        "ctrl_lcb_rank_biserial",
        "ctrl_ei_win_rate",
        "ctrl_ei_wilcoxon_adj_p",
        "ctrl_ei_rank_biserial",
    }


def test_batch_runner_multiple_files_and_resilience(tmp_path: Path):
    sub1 = tmp_path / "sweep1"
    sub2 = tmp_path / "sweep2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    f1 = sub1 / "logs.parquet"
    f2 = sub2 / "logs.parquet"
    f_missing = tmp_path / "missing" / "logs.parquet"

    df = _create_sample_sweep_df()
    df.to_parquet(f1)
    df.to_parquet(f2)

    outcomes = run_batch_scorecards(
        input_paths=[f1, f_missing, f2],
        subfolder="new_table",
        controls=["SMAC3_HPOFacade_lcb", "SMAC3_HPOFacade_ei"],
        tie_tolerance=1e-6,
    )

    assert len(outcomes) == 3
    assert outcomes[0]["status"] == "success"
    assert outcomes[1]["status"] == "error"
    assert outcomes[2]["status"] == "success"

    assert (sub1 / "new_table" / "scorecard.csv").exists()
    assert (sub2 / "new_table" / "scorecard.csv").exists()


def test_batch_cli(tmp_path: Path, monkeypatch):
    from scripts.run_batch_scorecard_analysis import main

    sub1 = tmp_path / "sweep_cli_1"
    sub2 = tmp_path / "sweep_cli_2"
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    f1 = sub1 / "logs.parquet"
    f2 = sub2 / "logs.parquet"
    df = _create_sample_sweep_df()
    df.to_parquet(f1)
    df.to_parquet(f2)

    argv = ["run_batch_scorecard_analysis.py", "--input", str(f1), str(f2)]
    exit_code = main(argv[1:])

    assert exit_code == 0
    assert (sub1 / "new_table" / "scorecard.csv").exists()
    assert (sub2 / "new_table" / "scorecard.csv").exists()


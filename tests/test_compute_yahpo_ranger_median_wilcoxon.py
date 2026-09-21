import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compute_yahpo_ranger_median_wilcoxon import (
    load_data,
    extract_final_incumbents,
    compute_task_medians,
    compute_mean_of_medians,
    compute_wilcoxon_and_cliffs_delta,
    format_markdown_report,
)


@pytest.fixture
def synthetic_logs_df():
    """Generates a small synthetic logs DataFrame for 2 tasks, 2 optimizers, and 3 seeds."""
    rows = []
    # Task 1: Proposed strictly better
    # Baseline incumbents: [0.20, 0.22, 0.24] -> median 0.22
    # Proposed incumbents: [0.10, 0.11, 0.12] -> median 0.11
    # Task 2: Baseline strictly better
    # Baseline incumbents: [0.05, 0.05, 0.06] -> median 0.05
    # Proposed incumbents: [0.08, 0.09, 0.07] -> median 0.08

    task_data = {
        ("task_1", "SMAC3_HPOFacade_lcb"): [0.20, 0.22, 0.24],
        ("task_1", "SMAC20_ProximityLCB"): [0.10, 0.11, 0.12],
        ("task_2", "SMAC3_HPOFacade_lcb"): [0.05, 0.05, 0.06],
        ("task_2", "SMAC20_ProximityLCB"): [0.08, 0.09, 0.07],
    }

    for (task_id, opt_id), final_costs in task_data.items():
        for seed_idx, final_cost in enumerate(final_costs, start=1):
            # 2 trials per run
            rows.append({
                "task_id": task_id,
                "optimizer_id": opt_id,
                "seed": seed_idx,
                "n_trials": 1,
                "trial_value__cost": final_cost + 0.1,
                "trial_value__cost_inc": final_cost + 0.1,
            })
            rows.append({
                "task_id": task_id,
                "optimizer_id": opt_id,
                "seed": seed_idx,
                "n_trials": 2,
                "trial_value__cost": final_cost,
                "trial_value__cost_inc": final_cost,
            })

    return pd.DataFrame(rows)


def test_extract_final_incumbents(synthetic_logs_df):
    incumbents = extract_final_incumbents(synthetic_logs_df)
    assert len(incumbents) == 4 * 3  # 2 tasks * 2 optimizers * 3 seeds
    
    # Check specific incumbent
    row = incumbents[
        (incumbents["task_id"] == "task_1")
        & (incumbents["optimizer_id"] == "SMAC20_ProximityLCB")
        & (incumbents["seed"] == 1)
    ]
    assert len(row) == 1
    assert np.isclose(row["incumbent_cost"].iloc[0], 0.10)


def test_compute_task_medians(synthetic_logs_df):
    incumbents = extract_final_incumbents(synthetic_logs_df)
    medians = compute_task_medians(incumbents)
    assert len(medians) == 4  # 2 tasks * 2 optimizers

    t1_prox = medians[
        (medians["task_id"] == "task_1") & (medians["optimizer_id"] == "SMAC20_ProximityLCB")
    ]["median_cost"].iloc[0]
    assert np.isclose(t1_prox, 0.11)

    t1_base = medians[
        (medians["task_id"] == "task_1") & (medians["optimizer_id"] == "SMAC3_HPOFacade_lcb")
    ]["median_cost"].iloc[0]
    assert np.isclose(t1_base, 0.22)

    t2_prox = medians[
        (medians["task_id"] == "task_2") & (medians["optimizer_id"] == "SMAC20_ProximityLCB")
    ]["median_cost"].iloc[0]
    assert np.isclose(t2_prox, 0.08)

    t2_base = medians[
        (medians["task_id"] == "task_2") & (medians["optimizer_id"] == "SMAC3_HPOFacade_lcb")
    ]["median_cost"].iloc[0]
    assert np.isclose(t2_base, 0.05)


def test_compute_mean_of_medians(synthetic_logs_df):
    incumbents = extract_final_incumbents(synthetic_logs_df)
    medians = compute_task_medians(incumbents)
    means = compute_mean_of_medians(medians)

    # Proposed: (0.11 + 0.08) / 2 = 0.095
    # Baseline: (0.22 + 0.05) / 2 = 0.135
    assert np.isclose(means["SMAC20_ProximityLCB"], 0.095)
    assert np.isclose(means["SMAC3_HPOFacade_lcb"], 0.135)


def test_compute_wilcoxon_and_cliffs_delta(synthetic_logs_df):
    incumbents = extract_final_incumbents(synthetic_logs_df)
    medians = compute_task_medians(incumbents)
    res = compute_wilcoxon_and_cliffs_delta(
        medians,
        proposed_id="SMAC20_ProximityLCB",
        baseline_id="SMAC3_HPOFacade_lcb",
    )

    assert res["n_tasks"] == 2
    assert np.isclose(res["mean_proposed"], 0.095)
    assert np.isclose(res["mean_baseline"], 0.135)
    assert res["wins_proposed"] == 1
    assert res["losses_proposed"] == 1
    assert res["ties"] == 0
    # Cliff's delta on 2 tasks (1 win, 1 loss) -> (1 - 1)/2 = 0.0
    assert np.isclose(res["cliffs_delta"], 0.0)
    assert res["cliffs_magnitude"] == "negligible"
    assert "wilcoxon_p" in res
    assert "wilcoxon_stat" in res


def test_load_data_parquet_and_csv(tmp_path, synthetic_logs_df):
    parquet_path = tmp_path / "logs.parquet"
    csv_path = tmp_path / "logs.csv"

    synthetic_logs_df.to_parquet(parquet_path)
    synthetic_logs_df.to_csv(csv_path, index=False)

    df_pq = load_data(parquet_path)
    df_csv = load_data(csv_path)

    assert len(df_pq) == len(synthetic_logs_df)
    assert len(df_csv) == len(synthetic_logs_df)


def test_format_markdown_report():
    stats = {
        "suite_name": "yahpo_rbv2_ranger",
        "n_tasks": 119,
        "proposed_id": "SMAC20_ProximityLCB",
        "baseline_id": "SMAC3_HPOFacade_lcb",
        "mean_proposed": 0.12345,
        "mean_baseline": 0.15678,
        "wilcoxon_stat": 2045.0,
        "wilcoxon_p": 0.0012,
        "cliffs_delta": 0.25,
        "cliffs_magnitude": "small",
        "wins_proposed": 70,
        "ties": 5,
        "losses_proposed": 44,
    }
    report = format_markdown_report(stats)
    assert "# Statistical Scorecard: yahpo_rbv2_ranger" in report
    assert "Mean of Medians" in report
    assert "SMAC20_ProximityLCB" in report
    assert "Cliff's Delta" in report

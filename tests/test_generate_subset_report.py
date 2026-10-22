import os
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_subset_report import (
    ensure_report_compatibility,
    filter_logs_dataframe,
    resolve_optimizer_column,
)


def test_resolve_optimizer_column():
    df1 = pd.DataFrame({"optimizer_id": ["a", "b"], "cost": [1.0, 2.0]})
    assert resolve_optimizer_column(df1) == "optimizer_id"

    df2 = pd.DataFrame({"optimizer": ["a", "b"], "cost": [1.0, 2.0]})
    assert resolve_optimizer_column(df2) == "optimizer"

    df3 = pd.DataFrame({"cost": [1.0, 2.0]})
    with pytest.raises(KeyError):
        resolve_optimizer_column(df3)


def test_filter_logs_dataframe_exact():
    df = pd.DataFrame({
        "optimizer_id": ["opt_a", "opt_b", "opt_c", "opt_a"],
        "cost": [1.0, 2.0, 3.0, 1.5],
    })
    filtered = filter_logs_dataframe(df, optimizers=["opt_a", "opt_c"])
    assert len(filtered) == 3
    assert set(filtered["optimizer_id"]) == {"opt_a", "opt_c"}


def test_filter_logs_dataframe_patterns():
    df = pd.DataFrame({
        "optimizer_id": [
            "SMAC3_HPOFacade_lcb",
            "SMAC3_HPOFacade_ei",
            "SMAC20_CustomUncertainty_lcb_shaker_entropy",
            "SMAC20_CustomUncertainty_lcb_proximity_bc",
            "SMAC20_CustomUncertainty_lcb_chen_variance",
        ],
        "cost": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    patterns = ["HPOFacade", "shaker_entropy", "proximity_bc"]
    filtered = filter_logs_dataframe(df, patterns=patterns)
    assert len(filtered) == 4
    assert "SMAC20_CustomUncertainty_lcb_chen_variance" not in filtered["optimizer_id"].values


def test_filter_logs_dataframe_no_match():
    df = pd.DataFrame({
        "optimizer_id": ["opt_a", "opt_b"],
        "cost": [1.0, 2.0],
    })
    with pytest.raises(ValueError):
        filter_logs_dataframe(df, optimizers=["opt_x"])


def test_filter_logs_dataframe_exclude_patterns():
    df = pd.DataFrame({
        "optimizer_id": [
            "SMAC20_CustomUncertainty_lcb_proximity_b",
            "SMAC20_CustomUncertainty_lcb_proximity_bc",
            "SMAC20_CustomUncertainty_pi_proximity_b",
            "SMAC3_HPOFacade_lcb",
        ],
        "cost": [1.0, 2.0, 3.0, 4.0],
    })
    # Want proximity_b without proximity_bc, and without pi
    filtered = filter_logs_dataframe(
        df,
        patterns=["proximity_b", "HPOFacade_lcb"],
        exclude_patterns=["proximity_bc", "_pi_"],
    )
    assert len(filtered) == 2
    assert set(filtered["optimizer_id"]) == {
        "SMAC20_CustomUncertainty_lcb_proximity_b",
        "SMAC3_HPOFacade_lcb",
    }


def test_filter_logs_dataframe_regex_anchor():
    df = pd.DataFrame({
        "optimizer_id": [
            "SMAC20_CustomUncertainty_lcb_proximity_b",
            "SMAC20_CustomUncertainty_lcb_proximity_bc",
        ],
        "cost": [1.0, 2.0],
    })
    # Using regex end anchor '$'
    filtered = filter_logs_dataframe(df, patterns=["proximity_b$"])
    assert len(filtered) == 1
    assert filtered["optimizer_id"].iloc[0] == "SMAC20_CustomUncertainty_lcb_proximity_b"


def test_ensure_report_compatibility_adds_default_task_type():
    df = pd.DataFrame({
        "task_id": ["task_1", "task_2"],
        "optimizer_id": ["opt_a", "opt_b"],
        "cost": [1.0, 2.0],
    })
    compat_df = ensure_report_compatibility(df)
    assert "task_type" in compat_df.columns
    assert (compat_df["task_type"] == "blackbox").all()
    # Ensure original dataframe wasn't mutated
    assert "task_type" not in df.columns


def test_ensure_report_compatibility_preserves_existing_task_type():
    df = pd.DataFrame({
        "task_id": ["task_1"],
        "optimizer_id": ["opt_a"],
        "task_type": ["multi-objective"],
    })
    compat_df = ensure_report_compatibility(df)
    assert list(compat_df["task_type"]) == ["multi-objective"]


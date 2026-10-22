import os
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_subset_report import filter_logs_dataframe, resolve_optimizer_column


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

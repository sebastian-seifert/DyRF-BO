#!/usr/bin/env python3
"""Statistical Analysis and Ranking Suite for Proximity Lower Bound Meta-HPO Sweep.

Computes Normalized Regret Loss across benchmark tasks:
1. For each task m and configuration cfg, compute seed-averaged incumbent cost at t=100:
   y_m(cfg) = mean_{seed} cost_inc(cfg, m, seed, t=100)

2. Normalize across all evaluated configurations on task m:
   y_norm_m(cfg) = (y_m(cfg) - y_min_m) / (y_max_m - y_min_m)
   where y_min_m = min_cfg y_m(cfg), y_max_m = max_cfg y_m(cfg).
   (If y_max_m == y_min_m, y_norm_m(cfg) = 0.0)

3. Aggregate into global Meta-HPO Loss:
   Loss(cfg) = (1 / M) * sum_{m=1}^M y_norm_m(cfg)

4. Rank configurations from best (lowest Loss) to worst, merging hyperparameter
   specifications (k, lambda, eps), and outputting GitHub-flavored Markdown and CSV reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Converts a pandas DataFrame to a clean GitHub-flavored markdown table."""
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        formatted_vals = []
        for val in row.values:
            if isinstance(val, (float, np.floating)):
                formatted_vals.append(f"{val:.4f}")
            else:
                formatted_vals.append(str(val))
        lines.append("| " + " | ".join(formatted_vals) + " |")
    return "\n".join(lines)


def compute_normalized_regret_loss(
    task_means: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Computes normalized regret per task and aggregate loss per configuration.

    Args:
        task_means: Mapping of task_id -> {config_id: seed_averaged_cost}.

    Returns:
        Tuple of (loss_dict, normalized_regret_df) where:
        - loss_dict maps config_id -> aggregate Loss(cfg) in [0, 1].
        - normalized_regret_df is a DataFrame with rows=configs and columns=tasks.
    """
    if not task_means:
        return {}, pd.DataFrame()

    tasks = list(task_means.keys())
    # Collect all unique config_ids
    config_ids = sorted({cfg for t in tasks for cfg in task_means[t].keys()})

    norm_data: Dict[str, Dict[str, float]] = {cfg: {} for cfg in config_ids}

    for task in tasks:
        cfg_costs = task_means[task]
        costs = np.array([cfg_costs[cfg] for cfg in config_ids if cfg in cfg_costs], dtype=float)
        valid_costs = costs[np.isfinite(costs)]

        if len(valid_costs) > 0:
            y_min = float(np.min(valid_costs))
            y_max = float(np.max(valid_costs))
            spread = y_max - y_min
        else:
            y_min, y_max, spread = 0.0, 0.0, 0.0

        for cfg in config_ids:
            if cfg not in cfg_costs or not np.isfinite(cfg_costs[cfg]):
                norm_data[cfg][task] = np.nan
            elif spread > 1e-12:
                norm_data[cfg][task] = float((cfg_costs[cfg] - y_min) / spread)
            else:
                norm_data[cfg][task] = 0.0

    norm_df = pd.DataFrame.from_dict(norm_data, orient="index")
    norm_df = norm_df[tasks]  # Maintain original task order

    loss_dict: Dict[str, float] = {}
    for cfg in config_ids:
        row_vals = norm_df.loc[cfg].to_numpy(dtype=float)
        # Penalize missing/failed tasks with worst-case normalized regret (1.0)
        # to prevent configs with sparse completions from artificially topping the leaderboard
        penalized_vals = np.where(np.isfinite(row_vals), row_vals, 1.0)
        loss_dict[cfg] = float(np.mean(penalized_vals))

    return loss_dict, norm_df


def load_logs_dataframe(input_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Loads benchmark results from DataFrame, Parquet, CSV, or telemetry directory."""
    if isinstance(input_path, pd.DataFrame):
        return input_path.copy()

    path = Path(input_path)
    if path.is_file():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            return pd.read_csv(path)

    # If directory, scan for telemetry JSON files or logs
    if path.is_dir():
        cand_parquet = path / "logs.parquet"
        if cand_parquet.exists():
            return pd.read_parquet(cand_parquet)

        cand_csv = path / "logs.csv"
        if cand_csv.exists():
            return pd.read_csv(cand_csv)

        telem_files = list(path.glob("**/telemetry_*.json"))
        if telem_files:
            import re
            telem_re = re.compile(
                r"^telemetry_(?P<optimizer_id>SMAC20_ProximityLCB_cfg_\d+|SMAC20_ProximityLCB_[a-zA-Z0-9]+)_(?P<task_id>.+?)_seed(?P<seed>\d+)\.json$"
            )
            rows = []
            for f in telem_files:
                try:
                    with open(f, "r") as fp:
                        data = json.load(fp)
                    trials = data.get("trials", [])
                    match = telem_re.match(f.name)
                    if match:
                        opt_id = data.get("optimizer_id") or match.group("optimizer_id")
                        task_id = data.get("task_id") or match.group("task_id")
                        seed = data.get("seed") or int(match.group("seed"))
                    else:
                        opt_id = data.get("optimizer_id", "unknown")
                        task_id = data.get("task_id", data.get("task_name", "unknown"))
                        seed = int(data.get("seed", 1))

                    for t_idx, tr in enumerate(trials, 1):
                        cost_val = tr.get("cost_inc", tr.get("cost", np.nan))
                        rows.append({
                            "trial": t_idx,
                            "optimizer_id": opt_id,
                            "task_id": task_id,
                            "seed": seed,
                            "trial_value__cost_inc": cost_val,
                        })
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows)

    raise ValueError(f"Could not load benchmark logs from: {input_path}")


def analyze_proximity_meta_hpo(
    input_data: Union[str, Path, pd.DataFrame],
    configs_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    checkpoint: int = 100,
    cost_col: str = "trial_value__cost_inc",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs end-to-end Meta-HPO evaluation and generates leaderboards."""
    df = load_logs_dataframe(input_data)

    # Standardize column names
    trial_col = None
    for cand in ["n_trials", "n_function_calls", "trial", "trial_number", "step"]:
        if cand in df.columns:
            trial_col = cand
            break

    task_col = "task_id" if "task_id" in df.columns else "task"

    if cost_col not in df.columns:
        for cand in ["cost_inc", "trial_value__cost", "cost", "value"]:
            if cand in df.columns:
                cost_col = cand
                break

    # Extract checkpoint data using robust idxmax slicing
    if trial_col is not None:
        sub = df[df[trial_col] <= checkpoint].copy()
        if sub.empty:
            sub = df.copy()
        idx = sub.groupby(["optimizer_id", task_col, "seed"])[trial_col].idxmax()
        df_slice = sub.loc[idx].reset_index(drop=True)
    else:
        df_slice = df.copy()

    # Extract config_id from optimizer_id
    def parse_cfg_id(opt_id: str) -> str:
        s = str(opt_id)
        if "SMAC20_ProximityLCB_" in s:
            return s.replace("SMAC20_ProximityLCB_", "")
        return s

    df_slice["config_id"] = df_slice["optimizer_id"].apply(parse_cfg_id)

    # Load configs metadata if available
    meta_by_cfg: Dict[str, Dict[str, Any]] = {}
    if configs_path and os.path.exists(configs_path):
        try:
            with open(configs_path, "r") as fp:
                cfg_list = json.load(fp)
            for c in cfg_list:
                cid = c.get("config_id")
                if cid:
                    meta_by_cfg[cid] = c
        except Exception:
            pass

    # Compute seed-averaged cost y_m(cfg) per task and per config
    task_means: Dict[str, Dict[str, float]] = {}
    grouped = df_slice.groupby([task_col, "config_id"])[cost_col].mean()

    for (tsk, cfg_id), avg_cost in grouped.items():
        if tsk not in task_means:
            task_means[tsk] = {}
        task_means[tsk][cfg_id] = float(avg_cost)

    # Compute Normalized Regret Loss
    loss_dict, norm_task_df = compute_normalized_regret_loss(task_means)

    # Compute raw cost statistics per config
    raw_stats = df_slice.groupby("config_id")[cost_col].agg(
        mean_raw_cost="mean",
        std_raw_cost=lambda x: np.std(x, ddof=1) if len(x) > 1 else 0.0,
        n_evals="count",
    ).reset_index()

    # Build Leaderboard
    leaderboard_records: List[Dict[str, Any]] = []
    for cfg_id, loss in loss_dict.items():
        meta = meta_by_cfg.get(cfg_id, {})
        k_val = meta.get("k", np.nan)
        lam_val = meta.get("decay_lambda", meta.get("lambda", np.nan))
        eps_val = meta.get("eps", np.nan)

        stat_row = raw_stats[raw_stats["config_id"] == cfg_id]
        mean_raw = stat_row["mean_raw_cost"].values[0] if not stat_row.empty else np.nan
        std_raw = stat_row["std_raw_cost"].values[0] if not stat_row.empty else 0.0
        n_eval = stat_row["n_evals"].values[0] if not stat_row.empty else 0
        sem_raw = std_raw / np.sqrt(n_eval) if n_eval > 1 else 0.0

        leaderboard_records.append({
            "config_id": cfg_id,
            "k": k_val,
            "lambda": lam_val,
            "eps": eps_val,
            "loss_normalized_regret": loss,
            "mean_raw_cost": mean_raw,
            "sem_raw_cost": sem_raw,
        })

    leaderboard_df = pd.DataFrame(leaderboard_records)
    # Sort ascending by loss_normalized_regret
    leaderboard_df = leaderboard_df.sort_values(
        by=["loss_normalized_regret", "mean_raw_cost"], ascending=[True, True]
    ).reset_index(drop=True)

    # Insert 1-based rank
    leaderboard_df.insert(0, "rank", range(1, len(leaderboard_df) + 1))

    # Add Loss summary column to norm_task_df
    norm_task_df["mean_normalized_regret"] = norm_task_df.index.map(loss_dict)
    norm_task_df = norm_task_df.loc[leaderboard_df["config_id"]]

    # Save reports if output_dir provided
    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        lb_csv = out_p / "proximity_meta_hpo_leaderboard.csv"
        lb_md = out_p / "proximity_meta_hpo_leaderboard.md"
        pt_csv = out_p / "proximity_meta_hpo_per_task.csv"
        pt_md = out_p / "proximity_meta_hpo_per_task.md"

        leaderboard_df.to_csv(lb_csv, index=False)
        with open(lb_md, "w") as f:
            f.write("# Proximity Lower Bound Meta-HPO Leaderboard (T=100)\n\n")
            f.write(f"Evaluated on {len(task_means)} benchmark tasks at horizon t={checkpoint}.\n\n")
            f.write(dataframe_to_markdown(leaderboard_df))
            f.write("\n")

        norm_task_df.to_csv(pt_csv, index=True)
        with open(pt_md, "w") as f:
            f.write("# Proximity Lower Bound Meta-HPO Normalized Regret Per Task\n\n")
            f.write(dataframe_to_markdown(norm_task_df.reset_index().rename(columns={"index": "config_id"})))
            f.write("\n")

        # Save optimal configuration as dedicated JSON artifact
        best_cfg_dict = leaderboard_df.iloc[0].to_dict()
        best_json_path = out_p / "best_config.json"
        with open(best_json_path, "w") as f:
            json.dump(best_cfg_dict, f, indent=2)

    return leaderboard_df, norm_task_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Normalized Regret and Rank Proximity Meta-HPO Configurations."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="results/sweep_proximity_meta_hpo/logs.parquet",
        help="Path to logs.parquet / logs.csv or directory containing telemetry files (default: results/sweep_proximity_meta_hpo/logs.parquet)",
    )
    parser.add_argument(
        "--configs",
        "-c",
        default="results/sweep_proximity_meta_hpo/sobol_configs.json",
        help="Path to sobol_configs.json (default: results/sweep_proximity_meta_hpo/sobol_configs.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results/sweep_proximity_meta_hpo/analysis",
        help="Directory to store analysis outputs (default: results/sweep_proximity_meta_hpo/analysis)",
    )
    parser.add_argument(
        "--checkpoint",
        "-t",
        type=int,
        default=100,
        help="Trial budget checkpoint horizon (default: 100)",
    )
    args = parser.parse_args()

    leaderboard, _ = analyze_proximity_meta_hpo(
        input_data=args.input,
        configs_path=args.configs,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )

    best = leaderboard.iloc[0]
    print("\n" + "=" * 65)
    print("🏆 OPTIMAL HYPERPARAMETER CONFIGURATION FOUND:")
    print("=" * 65)
    print(f"  Rank:                   1")
    print(f"  Config ID:              {best['config_id']}")
    print(f"  k (Neighbors):          {int(best['k'])}")
    print(f"  lambda (Decay Rate):    {best['lambda']:.4f}")
    print(f"  eps (Floor Ratio):      {best['eps']:.4f}")
    print(f"  Normalized Regret Loss: {best['loss_normalized_regret']:.4f}")
    print("=" * 65)
    print(f"Full details saved to: {args.output_dir}/best_config.json")
    print(f"Leaderboard saved to:  {args.output_dir}/proximity_meta_hpo_leaderboard.md\n")


if __name__ == "__main__":
    main()

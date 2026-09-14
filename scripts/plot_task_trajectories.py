#!/usr/bin/env python3
"""Plot Incumbent Optimization Trajectories Averaged Over Seeds.

For each of the 18 CARP-S BBsubset tasks:
- Averages incumbent cost across all seeds at each trial (1..100).
- Generates an individual comparison plot (Proposed Proximity LCB vs Baseline SMAC3 LCB)
  with mean curve and +/- 1 SEM uncertainty ribbon.
- Generates a combined grid plot with all 18 tasks for easy side-by-side comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def plot_task_trajectories(
    input_csv: str = "results/bbsubset_proximity_lcb_analysis/logs.csv",
    output_dir: str = "results/bbsubset_proximity_lcb_analysis/plots",
    proposed_id: str = "SMAC20_ProximityLCB_k10",
    baseline_id: str = "SMAC3_HPOFacade_lcb",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark logs from {input_csv}...")
    df = pd.read_csv(input_csv)

    trial_col = "n_trials" if "n_trials" in df.columns else "trial"
    cost_col = "trial_value__cost_inc" if "trial_value__cost_inc" in df.columns else "cost_inc"
    task_col = "task_id" if "task_id" in df.columns else "task"

    tasks = sorted(df[task_col].unique())
    print(f"Found {len(tasks)} tasks and {df['seed'].nunique()} seeds.")

    # Color palette and styling
    colors = {
        proposed_id: {"line": "#1f77b4", "shade": "#1f77b4", "label": "Proximity LCB (k=10, λ=1.0)"},
        baseline_id: {"line": "#d62728", "shade": "#d62728", "label": "SMAC3 LCB (κ=1.96)"},
    }

    # Prepare for multi-plot grid
    n_tasks = len(tasks)
    n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig_all, axes_all = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=True)
    axes_all = axes_all.flatten()

    summary_rows = []

    for idx, task_name in enumerate(tasks):
        clean_task_name = task_name.replace("blackbox/20/dev/", "").replace("/", "_")
        sub_df = df[df[task_col] == task_name]

        # Individual figure
        fig_single, ax_single = plt.subplots(figsize=(8, 5))

        task_res = {"task": clean_task_name}

        for opt_id, style in colors.items():
            opt_df = sub_df[sub_df["optimizer_id"] == opt_id]
            if opt_df.empty:
                continue

            # Group by trial and compute mean and SEM across seeds
            stats = opt_df.groupby(trial_col)[cost_col].agg(
                mean="mean",
                std="std",
                count="count"
            ).reset_index()

            stats["sem"] = stats["std"] / np.sqrt(stats["count"])
            stats["sem"] = stats["sem"].fillna(0.0)

            trials = stats[trial_col].values
            mean_vals = stats["mean"].values
            sem_vals = stats["sem"].values

            # Record final values at t=100
            final_mean = mean_vals[-1] if len(mean_vals) > 0 else np.nan
            final_sem = sem_vals[-1] if len(sem_vals) > 0 else np.nan
            if opt_id == proposed_id:
                task_res["proposed_final"] = f"{final_mean:.4e} ± {final_sem:.2e}"
                task_res["p_val_100"] = final_mean
            else:
                task_res["baseline_final"] = f"{final_mean:.4e} ± {final_sem:.2e}"
                task_res["b_val_100"] = final_mean

            # Plot on single figure
            ax_single.plot(trials, mean_vals, label=style["label"], color=style["line"], linewidth=2.0)
            ax_single.fill_between(
                trials,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                color=style["shade"],
                alpha=0.18
            )

            # Plot on all-tasks grid figure
            ax_grid = axes_all[idx]
            ax_grid.plot(trials, mean_vals, label=style["label"], color=style["line"], linewidth=1.8)
            ax_grid.fill_between(
                trials,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                color=style["shade"],
                alpha=0.15
            )

        summary_rows.append(task_res)

        # Style individual plot
        display_title = clean_task_name
        ax_single.set_title(f"Task: {display_title}", fontsize=13, fontweight="bold")
        ax_single.set_xlabel("Trial / Evaluation Step", fontsize=11)
        ax_single.set_ylabel("Best Incumbent Cost", fontsize=11)
        ax_single.grid(True, linestyle="--", alpha=0.5)
        ax_single.legend(fontsize=10, loc="best")
        fig_single.tight_layout()

        single_plot_path = output_path / f"trajectory_{clean_task_name}.png"
        fig_single.savefig(single_plot_path, dpi=200)
        plt.close(fig_single)

        # Style grid subplot
        ax_grid = axes_all[idx]
        ax_grid.set_title(display_title, fontsize=10, fontweight="bold")
        ax_grid.grid(True, linestyle="--", alpha=0.4)
        if idx % n_cols == 0:
            ax_grid.set_ylabel("Incumbent Cost", fontsize=9)
        if idx >= (n_rows - 1) * n_cols:
            ax_grid.set_xlabel("Trial", fontsize=9)

    # Hide extra subplots in grid if any
    for j in range(n_tasks, len(axes_all)):
        fig_all.delaxes(axes_all[j])

    # Put a single global legend on the grid figure
    handles, labels = axes_all[0].get_legend_handles_labels()
    fig_all.legend(handles, labels, loc="upper center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig_all.suptitle("CARP-S BBsubset: Proximity LCB vs SMAC3 Baseline Across All 18 Tasks (10 Seeds Mean ± SEM)", fontsize=15, fontweight="bold", y=1.01)
    fig_all.tight_layout()

    grid_plot_path = output_path / "all_18_tasks_trajectories.png"
    fig_all.savefig(grid_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig_all)

    print(f"\n[✓] Generated 18 individual plots in: {output_path}/")
    print(f"[✓] Generated master grid plot: {grid_plot_path}")

    # Summary Markdown Table
    print("\n" + "=" * 80)
    print(f"{'Task':<45} | {'Proximity LCB (t=100)':<22} | {'SMAC3 LCB (t=100)':<22}")
    print("-" * 80)
    for r in summary_rows:
        t_name = r["task"][:43]
        p_val = r.get("proposed_final", "N/A")
        b_val = r.get("baseline_final", "N/A")
        print(f"{t_name:<45} | {p_val:<22} | {b_val:<22}")
    print("=" * 80)


if __name__ == "__main__":
    plot_task_trajectories()

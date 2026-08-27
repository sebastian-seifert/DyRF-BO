#!/usr/bin/env python3
"""CARP-S Native Hypothesis Sub-Suite Analysis Runner.

Partitions master CARP-S logs into hypothesis-driven sub-datasets and executes
native `carps.analysis.generate_report` and `carps.analysis.run_autorank` to produce
focused Critical Difference diagrams, anytime curves, and LaTeX tables.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carps.analysis.generate_report import generate_report
from carps.analysis.run_autorank import calc as calc_autorank, custom_latex_table

def get_hypothesis_subgroups() -> dict[str, list[str]]:
    """Returns the dictionary of hypothesis-driven optimizer subgroups."""
    return {
        # 1. RQ1: All 7 Additive Hybrid Methods vs SMAC3 Baseline (K=8)
        "additive_vs_baseline": [
            "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
            "CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda",
            "CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy",
            "CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement",
            "SMAC3_HPOFacade_ei"
        ],
        # 2. RQ2: All 7 Direct Replacement Methods vs SMAC3 Baseline (K=8)
        "direct_vs_baseline": [
            "SMAC20_CustomUncertainty_ei_likelihood_credal",
            "SMAC20_CustomUncertainty_ei_standard_proximity",
            "SMAC20_CustomUncertainty_ei_proximity_b",
            "SMAC20_CustomUncertainty_ei_proximity_bc",
            "SMAC20_CustomUncertainty_ei_proximity_auto_lambda",
            "SMAC20_CustomUncertainty_ei_shaker_entropy",
            "SMAC20_CustomUncertainty_ei_standard_disagreement",
            "SMAC3_HPOFacade_ei"
        ],
        # 3. RQ3: Proximity Distance Family (4 Direct vs 4 Additive, K=8)
        "proximity_family_head_to_head": [
            "SMAC20_CustomUncertainty_ei_standard_proximity",
            "SMAC20_CustomUncertainty_ei_proximity_b",
            "SMAC20_CustomUncertainty_ei_proximity_bc",
            "SMAC20_CustomUncertainty_ei_proximity_auto_lambda",
            "CARPSDynamicRF_AdditiveEpistemic_ei_standard_proximity",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_b",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_bc",
            "CARPSDynamicRF_AdditiveEpistemic_ei_proximity_auto_lambda"
        ],
        # 4. RQ4: Information & Disagreement Family vs Baseline (K=7)
        "information_disagreement_family": [
            "SMAC20_CustomUncertainty_ei_likelihood_credal",
            "SMAC20_CustomUncertainty_ei_shaker_entropy",
            "SMAC20_CustomUncertainty_ei_standard_disagreement",
            "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
            "CARPSDynamicRF_AdditiveEpistemic_ei_shaker_entropy",
            "CARPSDynamicRF_AdditiveEpistemic_ei_standard_disagreement",
            "SMAC3_HPOFacade_ei"
        ],
        # 5. RQ5: Top Finalists (K=3)
        "top_finalists": [
            "CARPSDynamicRF_AdditiveEpistemic_ei_likelihood_credal",
            "SMAC20_CustomUncertainty_ei_standard_proximity",
            "SMAC3_HPOFacade_ei"
        ]
    }

def partition_carps_logs(
    input_parquet: str,
    out_dir: str,
    subgroups: dict[str, list[str]] = None
) -> dict[str, str]:
    """Filters the master CARP-S parquet file into separate sub-dataset files."""
    if subgroups is None:
        subgroups = get_hypothesis_subgroups()

    df = pd.read_parquet(input_parquet)
    os.makedirs(out_dir, exist_ok=True)
    partition_map = {}

    for name, opt_list in subgroups.items():
        sub_df = df[df["optimizer_id"].isin(opt_list)].copy()
        out_path = os.path.join(out_dir, f"logs_{name}.parquet")
        sub_df.to_parquet(out_path)
        partition_map[name] = out_path
        print(f"Partitioned '{name}': {len(sub_df)} rows across {sub_df['optimizer_id'].nunique()} optimizers -> {out_path}")

    return partition_map

def run_all_hypothesis_reports(
    input_parquet: str = "results/ei_comparison_analysis/logs.parquet",
    base_report_dir: str = "results/ei_comparison_analysis/reports/subgroups"
):
    """Partitions and executes native CARP-S report generation for all hypothesis sub-suites."""
    if not os.path.exists(input_parquet):
        print(f"Error: {input_parquet} not found!")
        sys.exit(1)

    partitions_dir = os.path.join(os.path.dirname(input_parquet), "partitions")
    partition_map = partition_carps_logs(input_parquet, partitions_dir)

    latex_tables_dir = os.path.join(base_report_dir, "latex_tables")
    os.makedirs(latex_tables_dir, exist_ok=True)

    print("\n==================================================")
    print("EXECUTING NATIVE CARP-S SUB-SUITE ANALYSIS")
    print("==================================================")

    for name, parquet_path in partition_map.items():
        print(f"\n---> Processing Subgroup: {name} ...")
        
        # 1. Native CARP-S Full Report (CD plots, rank over time, heatmaps, boxplots)
        try:
            generate_report(
                result_path=parquet_path,
                report_dir=base_report_dir,
                report_name=name,
                normalize_results=True
            )
            print(f"  [✓] Generated full CARP-S report in: {base_report_dir}/{name}/")
        except Exception as e:
            print(f"  [!] Error generating report for {name}: {e}")

        # 2. Native CARP-S Autorank & LaTeX Summary Table
        try:
            cd_res = calc_autorank(
                logs_file=parquet_path,
                output_path=os.path.join(base_report_dir, name, "figures", "standalone_cd_plot"),
                ignore_non_significance=True,
                plot_diagram=True
            )
            latex_table = custom_latex_table(cd_res, label=f"tbl:{name}")
            table_file = os.path.join(latex_tables_dir, f"table_{name}.tex")
            with open(table_file, "w") as f:
                f.write(latex_table)
            print(f"  [✓] Generated native LaTeX table in: {table_file}")
            print(f"      Friedman Omnibus p-value: {cd_res.pvalue:.4e} | Post-hoc: {cd_res.posthoc}")
        except Exception as e:
            print(f"  [!] Error in autorank calculation for {name}: {e}")

    print("\n==================================================")
    print("ALL SUB-SUITE CARP-S ANALYSES COMPLETED!")
    print(f"Reports & Figures: {base_report_dir}/")
    print(f"LaTeX Tables:      {latex_tables_dir}/")
    print("==================================================")

if __name__ == "__main__":
    logs_file = sys.argv[1] if len(sys.argv) > 1 else "results/ei_comparison_analysis/logs.parquet"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "results/ei_comparison_analysis/reports/subgroups"
    run_all_hypothesis_reports(logs_file, out_dir)

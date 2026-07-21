#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import generate_data
from metrics import (
    calculate_roc_metrics,
    calculate_aupr,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_rejection_curve,
    calculate_oracle_rejection_curve,
    calculate_random_rejection_curve,
    calculate_naurc
)
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ

import synthetic_functions

DEFAULT_FUNC_DICT = {}
_getters = [
    synthetic_functions.get_1d_functions,
    synthetic_functions.get_2d_functions,
    synthetic_functions.get_3d_functions,
    synthetic_functions.get_4d_functions,
    synthetic_functions.get_5d_functions,
    synthetic_functions.get_6d_functions,
    synthetic_functions.get_7d_functions,
    synthetic_functions.get_8d_functions,
    synthetic_functions.get_9d_functions,
    synthetic_functions.get_10d_functions,
    synthetic_functions.get_11d_functions,
    synthetic_functions.get_12d_functions,
    synthetic_functions.get_13d_functions,
    synthetic_functions.get_14d_functions,
    synthetic_functions.get_15d_functions,
]

for getter in _getters:
    funcs = getter()
    # Select the first function in the dict for clean 1-15D coverage
    first_key = list(funcs.keys())[0]
    DEFAULT_FUNC_DICT[first_key] = funcs[first_key]


def compute_metrics_for_uncertainty(uncertainty, predictions, y_test, y_true_binary):
    auroc, fpr95 = calculate_roc_metrics(y_true_binary, uncertainty)
    auprc = calculate_aupr(y_true_binary, uncertainty)
    jsd = calculate_jensen_shannon_divergence(uncertainty, y_true_binary)
    nmi = calculate_mutual_information(uncertainty, y_true_binary)
    
    # Calculate NAURC over rejection rates 0.0 to 0.95
    rejection_rates = np.linspace(0.0, 0.95, 50)
    rej_curve = calculate_rejection_curve(uncertainty, predictions, y_test, rejection_rates)
    oracle_curve = calculate_oracle_rejection_curve(predictions, y_test, rejection_rates)
    random_curve = calculate_random_rejection_curve(predictions, y_test, rejection_rates)
    naurc = calculate_naurc(rejection_rates, rej_curve, oracle_curve, random_curve)
    
    return {
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "jsd": jsd,
        "nmi": nmi,
        "naurc": naurc
    }

def run_ood_evaluation(funcs=None, seeds=None, ood_types=None, gap_types=None, device="auto"):
    if funcs is None:
        funcs = list(DEFAULT_FUNC_DICT.keys())
    if seeds is None:
        seeds = [42, 43, 44, 45, 46]
    if ood_types is None:
        ood_types = ["hypercube", "manifold"]
    if gap_types is None:
        gap_types = ["empty"]
        
    results = {}
    
    for ood_type in ood_types:
        results[ood_type] = {}
        for func_name in funcs:
            if func_name not in DEFAULT_FUNC_DICT:
                continue
                
            results[ood_type][func_name] = {
                "standard_rf": {m: [] for m in ["auroc", "auprc", "fpr95", "jsd", "nmi", "naurc"]},
                "proximity_auto_lambda": {m: [] for m in ["auroc", "auprc", "fpr95", "jsd", "nmi", "naurc"]}
            }
            
            for gap_type in gap_types:
                for seed in seeds:
                    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
                        DEFAULT_FUNC_DICT,
                        func_name,
                        seed=seed,
                        gap_type=gap_type,
                        ood_type=ood_type
                    )
                    
                    # 1. Standard RF Baseline ("Standard Standard")
                    rf_std = RandomForestRegressor(n_estimators=50, oob_score=True, random_state=seed)
                    rf_std.fit(X_train, y_train)
                    
                    preds_std = rf_std.predict(X_test)
                    tree_preds = np.array([tree.predict(X_test) for tree in rf_std.estimators_])
                    unc_std = np.var(tree_preds, axis=0) # Standard empirical tree variance
                    
                    m_std = compute_metrics_for_uncertainty(unc_std, preds_std, y_test, y_true_binary)
                    for k, v in m_std.items():
                        results[ood_type][func_name]["standard_rf"][k].append(v)
                        
                    # 2. Dynamic Lambda Proximity Approach (GPU/CUDA accelerated when available)
                    rf_prox = RandomForestRegressor(n_estimators=50, oob_score=True, random_state=seed)
                    rf_prox.fit(X_train, y_train)
                    
                    uq_engine = GPUProximityRegressionUQ(
                        rf_prox,
                        X_train,
                        y_train,
                        device=device,
                        use_density_scaling=True,
                        topological_decay_lambda=1.0
                    )
                    uq_engine.fit()
                    best_lambda = uq_engine.tune_lambda_oob(bounds=(0.001, 20.0))
                    uq_engine.topological_decay_lambda = best_lambda
                    
                    unc_auto = uq_engine.compute_uq(X_test)
                    preds_prox = rf_prox.predict(X_test)
                    
                    m_auto = compute_metrics_for_uncertainty(unc_auto, preds_prox, y_test, y_true_binary)
                    for k, v in m_auto.items():
                        results[ood_type][func_name]["proximity_auto_lambda"][k].append(v)
                        
    # Average across seeds
    final_summary = {}
    for ood_type in results:
        final_summary[ood_type] = {}
        for func_name in results[ood_type]:
            final_summary[ood_type][func_name] = {}
            for app in results[ood_type][func_name]:
                final_summary[ood_type][func_name][app] = {
                    k: float(np.nanmean(v)) for k, v in results[ood_type][func_name][app].items()
                }
                
    return final_summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD performance of Dynamic Lambda Proximity vs Standard RF Baseline")
    parser.add_argument("--output_dir", type=str, default="results/ood_dynamic_lambda", help="Output directory for reports and tables")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', 'cuda', 'gpu', or 'cpu'")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running Pure OOD Performance Benchmark (Hypercube & Manifold OOD) on device='{args.device}'...")
    summary = run_ood_evaluation(
        funcs=list(DEFAULT_FUNC_DICT.keys()),
        seeds=[42, 43, 44, 45, 46],
        ood_types=["hypercube", "manifold"],
        gap_types=["empty"],
        device=args.device
    )
    
    records = []
    for ood_type, funcs in summary.items():
        for func_name, apps in funcs.items():
            for app, metrics in apps.items():
                rec = {"OOD_Type": ood_type, "Function": func_name, "Approach": app}
                rec.update(metrics)
                records.append(rec)
                
    df = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "summary_ood_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    report_lines = [
        "# Pure OOD Performance Benchmark Report",
        "",
        "Comparing **Proximity Auto Lambda** against the **Standard RF Baseline (Empirical Variance)** across Hypercube and Manifold OOD setups.",
        "",
        "## Summary Table",
        "",
        df.to_markdown(index=False),
        ""
    ]
    
    report_path = os.path.join(args.output_dir, "summary_report_ood.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Benchmark complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()

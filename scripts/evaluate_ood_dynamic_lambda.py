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

DEFAULT_FUNC_DICT = {
    "sin_1d": {
        "func": lambda x: np.sin(x),
        "gap": [4.0, 6.0],
        "range": [0.0, 10.0]
    },
    "sin_cos_2d": {
        "func": lambda x, y: np.sin(x) * np.cos(y),
        "gap": [4.0, 6.0],
        "range": [0.0, 10.0]
    },
    "sin_cos_sin_3d": {
        "func": lambda x, y, z: np.sin(x) * np.cos(y) * np.sin(z),
        "gap": [4.0, 6.0],
        "range": [0.0, 10.0]
    },
    "friedman1_5d": {
        "func": lambda x1, x2, x3, x4, x5: 10 * np.sin(np.pi * x1 * x2) + 20 * (x3 - 0.5)**2 + 10 * x4 + 5 * x5,
        "gap": [4.0, 6.0],
        "range": [0.0, 10.0]
    },
    "borehole_8d": {
        "func": lambda rw, r, Tu, Hu, Tl, Hl, L, Kw: (
            (2 * np.pi * Tu * (Hu - Hl)) /
            (np.log(r / rw) * (1 + (2 * L * Tu) / (np.log(r / rw) * rw**2 * Kw) + Tu / Tl))
        ),
        "gap": [4.0, 6.0],
        "range": [0.05, 0.15]
    }
}

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

def run_ood_evaluation(funcs=None, seeds=None, ood_types=None, gap_types=None):
    if funcs is None:
        funcs = ["sin_1d", "sin_cos_2d", "sin_cos_sin_3d", "friedman1_5d"]
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
                        
                    # 2. Dynamic Lambda Proximity Approach
                    rf_prox = RandomForestRegressor(n_estimators=50, oob_score=True, random_state=seed)
                    rf_prox.fit(X_train, y_train)
                    
                    uq_engine = GPUProximityRegressionUQ(
                        rf_prox,
                        X_train,
                        y_train,
                        device="cpu",
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
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Running Pure OOD Performance Benchmark (Hypercube & Manifold OOD)...")
    summary = run_ood_evaluation(
        funcs=["sin_1d", "sin_cos_2d", "sin_cos_sin_3d", "friedman1_5d"],
        seeds=[42, 43, 44, 45, 46],
        ood_types=["hypercube", "manifold"],
        gap_types=["empty"]
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

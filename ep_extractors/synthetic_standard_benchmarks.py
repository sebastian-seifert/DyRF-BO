#!/usr/bin/env python3
"""Standard Synthetic OOD Benchmark Experiment Runner (1D to 15D).

Evaluates custom epistemic uncertainty extractors and SMAC3 baseline on all 55 
standard synthetic benchmark functions across empty and sparse OOD gaps.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

from data_generator import generate_data
import synthetic_functions as sf
from ep_extractors import UQExtractorRegistry

from metrics import (
    calculate_roc_metrics,
    calculate_aupr,
    calculate_aurc_exact,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_nlpd
)

def get_all_standard_functions():
    """Returns exactly 3 primary synthetic benchmark functions for every dimension 1D through 15D (45 functions total)."""
    funcs = {}
    
    # 1D (3)
    f1 = sf.get_1d_functions()
    for k in ["sin", "poly", "damped_osc"]:
        funcs[k] = f1[k]
        
    # 2D (3)
    f2 = sf.get_2d_functions()
    for k in ["sin_cos", "quadratic", "gaussian"]:
        funcs[k] = f2[k]
        
    # 3D (3)
    f3 = sf.get_3d_functions()
    for k in ["sin_cos_sin", "quadratic_3d", "gaussian_3d"]:
        funcs[k] = f3[k]
        
    # 4D (3)
    f4 = sf.get_4d_functions()
    for k in ["sin_cos_4d", "quadratic_4d", "sin_sum_4d"]:
        funcs[k] = f4[k]
        
    # 5D (3)
    funcs.update(sf.get_5d_functions())
    
    # 6D (3)
    f6 = sf.get_6d_functions()
    for k in ["sin_cos_6d", "quadratic_6d", "friedman_6d"]:
        funcs[k] = f6[k]
        
    # 7D through 15D (3 functions each)
    funcs.update(sf.get_7d_functions())
    funcs.update(sf.get_8d_functions())
    funcs.update(sf.get_9d_functions())
    funcs.update(sf.get_10d_functions())
    funcs.update(sf.get_11d_functions())
    funcs.update(sf.get_12d_functions())
    funcs.update(sf.get_13d_functions())
    funcs.update(sf.get_14d_functions())
    funcs.update(sf.get_15d_functions())
    return funcs


def run_standard_benchmark_experiment(
    func_name: str,
    gap_type: str = "empty",
    approach: str = "proximity_bc",
    seed: int = 1,
    noise_std: float = 0.1,
    id_split: float = 0.7,
    output_dir: str | None = None
) -> dict:
    """Executes a single standard synthetic OOD evaluation run across all 8 metrics."""
    func_dict = get_all_standard_functions()
    if func_name not in func_dict:
        raise ValueError(f"Unknown benchmark function '{func_name}'.")

    # 1. Generate Dataset
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict=func_dict,
        func_name=func_name,
        seed=seed,
        gap_type=gap_type,
        noise_std=noise_std,
        id_split=id_split
    )

    # 2. Fit Random Forest Regressor Surrogate
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=seed, min_samples_leaf=5, n_jobs=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    abs_error = np.abs(y_test - y_pred)

    # 3. Extract Epistemic Uncertainty Signal U(x)
    if approach in ["smac3_variance", "standard_variance", "baseline"]:
        predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
        unc_signal = np.std(predictions, axis=0)
    else:
        extractor = UQExtractorRegistry.get(approach, rf)
        extractor.fit(X_train, y_train)
        unc_signal = extractor.extract_epistemic_signal(X_test)

    unc_signal = np.nan_to_num(unc_signal, nan=0.0, posinf=1e6, neginf=0.0)

    # 4. Compute All 8 Evaluation Metrics
    auroc, fpr95 = calculate_roc_metrics(y_true_binary, unc_signal)
    aupr = calculate_aupr(y_true_binary, unc_signal)
    
    spearman_corr, _ = spearmanr(unc_signal, abs_error)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    aurc = calculate_aurc_exact(unc_signal, y_pred, y_test, loss_type="MAE")
    oracle_aurc = calculate_aurc_exact(abs_error, y_pred, y_test, loss_type="MAE")

    jsd = calculate_jensen_shannon_divergence(unc_signal, y_true_binary)
    mi = calculate_mutual_information(unc_signal, y_true_binary)

    nlpd = calculate_nlpd(y_test, y_pred, unc_signal)
    brier_score = np.mean((y_true_binary - unc_signal / (np.max(unc_signal) + 1e-8))**2)

    results = {
        "func_name": func_name,
        "gap_type": gap_type,
        "approach": approach,
        "seed": seed,
        "auroc": float(auroc),
        "fpr95": float(fpr95),
        "aupr": float(aupr),
        "spearman": float(spearman_corr),
        "aurc": float(aurc),
        "oracle_aurc": float(oracle_aurc),
        "jsd": float(jsd),
        "mi": float(mi),
        "nlpd": float(nlpd),
        "brier": float(brier_score),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"standard_result_{func_name}_{gap_type}_{approach}_seed{seed}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single standard benchmark OOD evaluation task.")
    parser.add_argument("--func_name", type=str, required=True)
    parser.add_argument("--gap_type", type=str, default="empty")
    parser.add_argument("--approach", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/standard_sweep")
    args = parser.parse_args()

    res = run_standard_benchmark_experiment(
        func_name=args.func_name,
        gap_type=args.gap_type,
        approach=args.approach,
        seed=args.seed,
        output_dir=args.output_dir
    )
    print(json.dumps(res, indent=2))

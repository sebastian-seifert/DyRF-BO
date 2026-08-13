#!/usr/bin/env python3
"""Synthetic OOD Benchmark Experiment Runner.

Evaluates custom epistemic uncertainty extractors and SMAC3 baseline on 
new benchmark functions (Ackley 2D/4D, Rosenbrock 2D/4D, Hartmann 6D) across
empty and sparse OOD gaps.
"""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

from data_generator import generate_data
from synthetic_functions import (
    get_2d_functions,
    get_4d_functions,
    get_6d_functions
)
from ep_extractors import UQExtractorRegistry
from metrics import (
    calculate_roc_metrics,
    calculate_aupr,
    calculate_aurc,
    calculate_oracle_rejection_curve,
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_nlpd
)

def get_master_func_dict():
    """Combines all custom synthetic functions into a master registry dict."""
    funcs = {}
    funcs.update(get_2d_functions())
    funcs.update(get_4d_functions())
    funcs.update(get_6d_functions())
    return funcs

def run_ood_detection_experiment(
    func_name: str,
    gap_type: str = "empty",
    approach: str = "proximity_bc",
    seed: int = 1,
    noise_std: float = 0.1,
    id_split: float = 0.7,
    n_samples: int | None = None,
    output_dir: str | None = None
) -> dict:
    """Executes a single synthetic OOD evaluation run and returns all 8 metrics."""
    func_dict = get_master_func_dict()
    if func_name not in func_dict:
        raise ValueError(f"Unknown benchmark function '{func_name}'. Available: {list(func_dict.keys())}")

    # 1. Generate Dataset (ID Train, ID Test, OOD Test)
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict=func_dict,
        func_name=func_name,
        seed=seed,
        gap_type=gap_type,
        noise_std=noise_std,
        id_split=id_split
    )

    # 2. Fit Random Forest Regressor Surrogate with OOB scoring enabled
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=seed, min_samples_leaf=5, n_jobs=1)
    rf.fit(X_train, y_train)


    # Predict mean and test residuals
    y_pred = rf.predict(X_test)
    abs_error = np.abs(y_test - y_pred)

    # 3. Extract Epistemic Uncertainty Signal U(x)
    if approach in ["smac3_variance", "standard_variance", "baseline"]:
        # Standard Law of Total Variance across trees
        predictions = np.array([tree.predict(X_test) for tree in rf.estimators_])
        unc_signal = np.std(predictions, axis=0)
    else:
        extractor = UQExtractorRegistry.get(approach, rf)
        extractor.fit(X_train, y_train)
        unc_signal = extractor.extract_epistemic_signal(X_test)

    # Ensure no NaN or Inf in uncertainty signal
    unc_signal = np.nan_to_num(unc_signal, nan=0.0, posinf=1e6, neginf=0.0)

    # 4. Compute All 8 Evaluation Metrics
    # A. AUROC & FPR@95
    auroc, fpr95, _, _ = calculate_roc_metrics(y_true_binary, unc_signal)
    
    # B. AUPR
    aupr = calculate_aupr(y_true_binary, unc_signal)
    
    # C. Spearman Rank Correlation (U(x) vs |y - y_pred|)
    spearman_corr, _ = spearmanr(unc_signal, abs_error)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0

    # D. AURC & Oracle AURC
    aurc, _, _ = calculate_aurc(abs_error, unc_signal)
    oracle_aurc, _, _ = calculate_oracle_rejection_curve(abs_error)

    # E. Information-Theoretic Metrics (JSD & MI)
    jsd = calculate_jensen_shannon_divergence(unc_signal)
    mi = calculate_mutual_information(unc_signal)

    # F. Brier Score & NLPD
    nlpd = calculate_nlpd(y_test, y_pred, unc_signal)
    # Balanced Brier score calculation (ID=0 vs OOD=1)
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
        out_file = os.path.join(output_dir, f"ood_result_{func_name}_{gap_type}_{approach}_seed{seed}.json")
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run single OOD detection benchmark task.")
    parser.add_argument("--func_name", type=str, required=True)
    parser.add_argument("--gap_type", type=str, default="empty")
    parser.add_argument("--approach", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/ood_sweep")
    args = parser.parse_args()

    res = run_ood_detection_experiment(
        func_name=args.func_name,
        gap_type=args.gap_type,
        approach=args.approach,
        seed=args.seed,
        output_dir=args.output_dir
    )
    print(json.dumps(res, indent=2))

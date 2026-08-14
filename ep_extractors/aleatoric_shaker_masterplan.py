#!/usr/bin/env python3
"""Aleatoric Noise Quantification Masterplan: Shaker Entropy vs. Standard Arithmetic Aleatoric UQ.

Investigates whether Shaker Aleatoric Entropy (Geometric Variance Inversion) outperforms 
Standard Arithmetic Leaf Variance in capturing pure noise across 5 benchmark target functions,
6 noise regimes (homoscedastic & heteroscedastic), and 5 Random Forest hyperparameter configurations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# =====================================================================
# 1. Benchmark Target Functions f(x)
# =====================================================================
def get_benchmark_functions():
    """Returns exactly 1 benchmark target function for every dimension 1D through 15D (15 functions total)."""
    funcs = {
        "sin_1d": {
            "dim": 1,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0])
        },
        "sin_cos_2d": {
            "dim": 2,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1])
        },
        "sin_cos_sin_3d": {
            "dim": 3,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2])
        },
        "sin_cos_4d": {
            "dim": 4,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3])
        },
        "sin_cos_5d": {
            "dim": 5,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4])
        },
        "sin_cos_6d": {
            "dim": 6,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5])
        },
        "sin_cos_7d": {
            "dim": 7,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6])
        },
        "sin_cos_8d": {
            "dim": 8,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7])
        },
        "sin_cos_9d": {
            "dim": 9,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8])
        },
        "sin_cos_10d": {
            "dim": 10,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9])
        },
        "sin_cos_11d": {
            "dim": 11,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9]) * np.sin(x[:, 10])
        },
        "sin_cos_12d": {
            "dim": 12,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9]) * np.sin(x[:, 10]) * np.cos(x[:, 11])
        },
        "sin_cos_13d": {
            "dim": 13,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9]) * np.sin(x[:, 10]) * np.cos(x[:, 11]) * np.sin(x[:, 12])
        },
        "sin_cos_14d": {
            "dim": 14,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9]) * np.sin(x[:, 10]) * np.cos(x[:, 11]) * np.sin(x[:, 12]) * np.cos(x[:, 13])
        },
        "sin_cos_15d": {
            "dim": 15,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.sin(x[:, 2]) * np.cos(x[:, 3]) * np.sin(x[:, 4]) * np.cos(x[:, 5]) * np.sin(x[:, 6]) * np.cos(x[:, 7]) * np.sin(x[:, 8]) * np.cos(x[:, 9]) * np.sin(x[:, 10]) * np.cos(x[:, 11]) * np.sin(x[:, 12]) * np.cos(x[:, 13]) * np.sin(x[:, 14])
        }
    }
    return funcs

# =====================================================================
# 2. Noise Regimes \sigma_{true}(x) (Homoscedastic & Heteroscedastic)
# =====================================================================
def get_noise_regimes():
    r"""Returns 6 state-of-the-art noise functions \sigma_{true}(x)."""
    return {
        "homoscedastic_low": {
            "type": "constant",
            "func": lambda x: np.full(len(x), 0.1)
        },
        "homoscedastic_high": {
            "type": "constant",
            "func": lambda x: np.full(len(x), 0.5)
        },
        "hetero_linear": {
            "type": "heteroscedastic",
            "func": lambda x: 0.05 + 0.5 * (x[:, 0] / 10.0)
        },
        "hetero_sinusoidal": {
            "type": "heteroscedastic",
            "func": lambda x: 0.1 * (1.0 + 2.0 * (np.sin(np.pi * x[:, 0] / 5.0)**2))
        },
        "hetero_localized": {
            "type": "heteroscedastic",
            "func": lambda x: 0.1 + 0.8 * np.exp(-((x[:, 0] - 5.0)**2) / 4.0)
        },
        "hetero_quadratic": {
            "type": "heteroscedastic",
            "func": lambda x: 0.05 + 0.4 * (((x[:, 0] - 5.0) / 5.0)**2)
        }
    }


# =====================================================================
# 3. Random Forest Hyperparameter Configurations (5 Configurations)
# =====================================================================
def get_rf_configs():
    """Returns 5 Random Forest hyperparameter configurations for maximum insight."""
    return {
        "RF_Default": {
            "n_estimators": 100,
            "min_samples_leaf": 5,
            "max_depth": None
        },
        "RF_Overfit_Leaf1": {
            "n_estimators": 100,
            "min_samples_leaf": 1,
            "max_depth": None
        },
        "RF_Smoothed_Leaf15": {
            "n_estimators": 100,
            "min_samples_leaf": 15,
            "max_depth": None
        },
        "RF_Shallow": {
            "n_estimators": 100,
            "min_samples_leaf": 5,
            "max_depth": 4
        },
        "RF_DeepEnsemble300": {
            "n_estimators": 300,
            "min_samples_leaf": 5,
            "max_depth": 12
        }
    }

# =====================================================================
# 4. Aleatoric Extraction Engine
# =====================================================================
def extract_leaf_variances(rf: RandomForestRegressor, X: np.ndarray, min_var: float = 1e-6) -> np.ndarray:
    r"""Extracts leaf node variance \sigma_m^2(x) for each tree m and sample x.

    
    Returns:
        vars2: Array of shape (n_trees, n_samples)
    """
    leaf_ids = rf.apply(X) # (n_samples, n_trees)
    n_samples, n_trees = leaf_ids.shape
    vars2 = np.zeros((n_trees, n_samples))

    for m, estimator in enumerate(rf.estimators_):
        t_leaf_ids = leaf_ids[:, m]
        impurity = estimator.tree_.impurity[t_leaf_ids]
        vars2[m, :] = np.maximum(impurity, min_var)

    return vars2

def compute_aleatoric_signals(vars2: np.ndarray) -> dict:
    """Computes all 5 aleatoric uncertainty signals from per-tree leaf variances.
    
    Args:
        vars2: Array of shape (n_trees, n_samples)
    
    Returns:
        dict of aleatoric signal arrays of shape (n_samples,)
    """
    M = vars2.shape[0]

    # 1. Shaker Differential Entropy (bits)
    # H_a(x) = (1/M) * sum_{m=1}^M 0.5 * log2(2 * pi * e * \sigma_m^2(x))
    ind_entropies = 0.5 * np.log2(2.0 * np.pi * np.e * vars2)
    shaker_entropy = np.mean(ind_entropies, axis=0)

    # 2. Shaker Geometric Mean Variance: \sigma_{a,geom}^2 = (1 / 2\pi e) * 2^(2 * H_a)
    shaker_geom_var = (2.0 ** (2.0 * shaker_entropy)) / (2.0 * np.pi * np.e)
    shaker_geom_std = np.sqrt(np.maximum(shaker_geom_var, 1e-12))

    # 3. Standard Arithmetic Mean Variance: \sigma_{a,ari}^2 = (1/M) * sum \sigma_m^2(x)
    standard_ari_var = np.mean(vars2, axis=0)
    standard_ari_std = np.sqrt(np.maximum(standard_ari_var, 1e-12))

    return {
        "shaker_entropy": shaker_entropy,
        "shaker_geom_var": shaker_geom_var,
        "shaker_geom_std": shaker_geom_std,
        "standard_ari_var": standard_ari_var,
        "standard_ari_std": standard_ari_std
    }

# =====================================================================
# 5. Evaluation Metrics Engine
# =====================================================================
def evaluate_aleatoric_metrics(
    unc_signal: np.ndarray,
    sigma_true: np.ndarray,
    abs_residual: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """Calculates all 6 evaluation metrics for a given aleatoric uncertainty signal."""
    sigma_true_var = sigma_true**2
    squared_residual = abs_residual**2

    # 1. Spearman Rank Correlation vs. Ground-Truth Noise Variance
    sp_true, _ = spearmanr(unc_signal, sigma_true_var)
    if np.isnan(sp_true): sp_true = 0.0

    # 2. Spearman Rank Correlation vs. Empirical Squared Residuals
    sp_resid, _ = spearmanr(unc_signal, squared_residual)
    if np.isnan(sp_resid): sp_resid = 0.0

    # 3. Log-Pearson Correlation vs. Ground-Truth Noise Variance
    valid_mask = (unc_signal > 1e-12) & (sigma_true_var > 1e-12)
    if np.sum(valid_mask) > 10:
        lp_true, _ = pearsonr(np.log(unc_signal[valid_mask]), np.log(sigma_true_var[valid_mask]))
        if np.isnan(lp_true): lp_true = 0.0
    else:
        lp_true = 0.0

    # 4. Variance Calibration Errors (MSE and RMSE)
    mse_var = float(np.mean((unc_signal - sigma_true_var)**2))
    rmse_var = float(np.sqrt(mse_var))

    # 5. Heteroscedastic Negative Log Predictive Density (NLPD)
    var_clamped = np.maximum(unc_signal, 1e-8)
    nlpd_val = float(np.mean(0.5 * ((y_test - y_pred)**2 / var_clamped) + 0.5 * np.log(2.0 * np.pi * var_clamped)))

    return {
        "spearman_true": float(sp_true),
        "spearman_resid": float(sp_resid),
        "log_pearson_true": float(lp_true),
        "mse_var": float(mse_var),
        "rmse_var": float(rmse_var),
        "nlpd_aleatoric": float(nlpd_val)
    }

# =====================================================================
# 6. Single Experiment Runner
# =====================================================================
from scipy.stats.qmc import Sobol

def run_single_aleatoric_experiment(
    func_name: str,
    noise_name: str,
    rf_config_name: str,
    seed: int = 1,
    n_train: int = 1000,
    n_test: int = 300
) -> dict:
    """Executes a single benchmark run comparing all 5 aleatoric approaches across all metrics."""
    funcs = get_benchmark_functions()
    noises = get_noise_regimes()
    rf_configs = get_rf_configs()

    f_info = funcs[func_name]
    n_info = noises[noise_name]
    rf_cfg = rf_configs[rf_config_name]

    dim = f_info["dim"]
    domain = f_info["domain"]

    # Dynamic scaling per dimension: 1,000 * d train points, 300 * d test points
    n_train_dim = n_train * dim
    n_test_dim = n_test * dim

    # 1. Generate Uniformly Spread Training and Test Datasets
    np.random.seed(seed)

    if dim == 1:
        X_train = np.linspace(domain[0], domain[1], n_train_dim).reshape(-1, 1)
        X_test = np.linspace(domain[0], domain[1], n_test_dim).reshape(-1, 1)
    else:
        # Quasi-Monte Carlo Sobol sequence sampling for even multi-dimensional coverage
        sampler_train = Sobol(d=dim, scramble=True, seed=seed)
        X_train = domain[0] + (domain[1] - domain[0]) * sampler_train.random(n=n_train_dim)

        sampler_test = Sobol(d=dim, scramble=True, seed=seed + 10000)
        X_test = domain[0] + (domain[1] - domain[0]) * sampler_test.random(n=n_test_dim)



    # Ground-truth noiseless y_clean and true noise scale \sigma_{true}(x)
    y_clean_train = f_info["func"](X_train)
    sigma_true_train = n_info["func"](X_train)
    y_train = y_clean_train + np.random.normal(0, sigma_true_train)

    y_clean_test = f_info["func"](X_test)
    sigma_true_test = n_info["func"](X_test)
    y_test = y_clean_test + np.random.normal(0, sigma_true_test)

    # 2. Fit Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        max_depth=rf_cfg["max_depth"],
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    abs_residual = np.abs(y_test - y_pred)

    # 3. Extract Leaf Variances & Compute Aleatoric Signals
    vars2 = extract_leaf_variances(rf, X_test)
    signals = compute_aleatoric_signals(vars2)

    # 4. Evaluate Metrics for Every Approach
    results = {}
    for app_name, signal in signals.items():
        metrics = evaluate_aleatoric_metrics(
            unc_signal=signal,
            sigma_true=sigma_true_test,
            abs_residual=abs_residual,
            y_test=y_test,
            y_pred=y_pred
        )
        results[app_name] = metrics

    return results

# =====================================================================
# 7. Master Sweep Harness
# =====================================================================
def run_masterplan_sweep(output_dir: str = "results/aleatoric_masterplan", seeds: list = [1, 2, 3, 4, 5]):
    """Runs full 750-model masterplan sweep across all functions, noise regimes, RF configs, and seeds."""
    os.makedirs(output_dir, exist_ok=True)

    funcs = list(get_benchmark_functions().keys())
    noises = list(get_noise_regimes().keys())
    rf_configs = list(get_rf_configs().keys())

    total_experiments = len(funcs) * len(noises) * len(rf_configs) * len(seeds)
    print("==================================================")
    print(f"Executing Aleatoric Noise Masterplan Sweep")
    print(f"Total Model Fits: {total_experiments} (5 Functions × 6 Noise Regimes × 5 RF Configs × {len(seeds)} Seeds)")
    print("==================================================")

    records = []
    count = 0
    for f in funcs:
        for n in noises:
            for rf_cfg in rf_configs:
                for s in seeds:
                    count += 1
                    if count % 25 == 0 or count == total_experiments:
                        print(f"Progress: [{count}/{total_experiments}] running {f} | {n} | {rf_cfg} | seed {s}...", flush=True)

                    res = run_single_aleatoric_experiment(
                        func_name=f,
                        noise_name=n,
                        rf_config_name=rf_cfg,
                        seed=s
                    )

                    for app_name, metrics in res.items():
                        rec = {
                            "func_name": f,
                            "noise_name": n,
                            "rf_config": rf_cfg,
                            "seed": s,
                            "approach": app_name
                        }
                        rec.update(metrics)
                        records.append(rec)

    df = pd.DataFrame(records)
    csv_file = os.path.join(output_dir, "aleatoric_masterplan_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved complete raw dataset to '{csv_file}'")

    # Generate Publication Summary
    summary_file = os.path.join(output_dir, "aleatoric_masterplan_summary.md")
    generate_summary_report(df, summary_file)
    print(f"Saved summary report to '{summary_file}'")
    return df

def generate_summary_report(df: pd.DataFrame, output_md: str):
    """Computes mean metrics and writes a GitHub Markdown report."""
    metrics = ["spearman_true", "spearman_resid", "log_pearson_true", "mse_var", "rmse_var", "nlpd_aleatoric"]

    with open(output_md, "w") as mf:
        mf.write("# Aleatoric Uncertainty Masterplan Report: Shaker Entropy vs. Arithmetic Mean Variance\n\n")
        mf.write(f"**Total Records**: {len(df)} runs across 5 functions, 6 noise regimes, 5 RF configurations, and 5 seeds.\n\n")

        mf.write("## 1. Grand Mean Performance Across All Experiments\n\n")
        grand = df.groupby("approach")[metrics].mean().round(4)
        mf.write(grand.to_markdown() + "\n\n")

        mf.write("## 2. Performance by RF Configuration (Spearman Rank vs. Ground-Truth Noise)\n\n")
        piv_rf = df.groupby(["approach", "rf_config"])["spearman_true"].mean().unstack(level=-1).round(4)
        mf.write(piv_rf.to_markdown() + "\n\n")

        mf.write("## 3. Performance by Noise Regime (Spearman Rank vs. Ground-Truth Noise)\n\n")
        piv_noise = df.groupby(["approach", "noise_name"])["spearman_true"].mean().unstack(level=-1).round(4)
        mf.write(piv_noise.to_markdown() + "\n\n")

        mf.write("## 4. Heteroscedastic NLPD by Approach and RF Configuration\n\n")
        piv_nlpd = df.groupby(["approach", "rf_config"])["nlpd_aleatoric"].mean().unstack(level=-1).round(4)
        mf.write(piv_nlpd.to_markdown() + "\n\n")

if __name__ == "__main__":
    run_masterplan_sweep()

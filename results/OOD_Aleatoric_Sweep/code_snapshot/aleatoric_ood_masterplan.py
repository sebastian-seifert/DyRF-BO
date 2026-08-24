#!/usr/bin/env python3
"""
Aleatoric OOD Masterplan Benchmark Engine:
Evaluates Aleatoric Uncertainty and Epistemic Disagreement on Partitioned In-Distribution & Out-of-Distribution Gaps.

Supported Estimators:
1. shaker_entropy: Differential entropy across Gaussian leaf distributions (in bits).
2. shaker_geom_var: Inverted geometric mean variance across tree leaves (in sigma^2 units).
3. shaker_geom_std: Geometric mean standard deviation across tree leaves (in sigma units).
4. standard_ari_var: Arithmetic mean variance across tree leaves (in sigma^2 units).
5. standard_ari_std: Arithmetic mean standard deviation across tree leaves (in sigma units).
6. standard_disagreement: Standard ensemble variance of tree predictions (Epistemic baseline).
"""

from __future__ import annotations
import os
import math
import json
import warnings
from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.stats.qmc import Sobol
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. 270 Benchmark Functions (1D through 15D)
# =====================================================================
def get_benchmark_functions() -> Dict[str, Dict[str, Any]]:
    """Returns 1D to 15D continuous synthetic benchmark target functions."""
    funcs = {
        "sin_cos_1d": {
            "dim": 1,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 0] / 2.0)
        },
        "sin_cos_2d": {
            "dim": 2,
            "domain": (0.0, 10.0),
            "func": lambda x: np.sin(x[:, 0]) * np.cos(x[:, 1])
        },
        "sin_cos_3d": {
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
# 2. 7 Ground-Truth Noise Regimes (Homoscedastic & Heteroscedastic)
# =====================================================================
def get_noise_regimes() -> Dict[str, Dict[str, Any]]:
    """Returns 7 ground-truth noise profiles sigma_true(x)."""
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
        },
        "hetero_ood_step_double": {
            "type": "heteroscedastic_step",
            "func": lambda x: np.where((x[:, 0] >= 4.0) & (x[:, 0] <= 6.0), 0.20, 0.10)
        }
    }

# =====================================================================
# 3. 5 Random Forest Hyperparameter Configurations
# =====================================================================
def get_rf_configs() -> Dict[str, Dict[str, Any]]:
    """Returns 5 Random Forest hyperparameter configurations."""
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
# 4. Partitioned OOD Dataset Generator
# =====================================================================
def generate_ood_aleatoric_dataset(
    f_info: Dict[str, Any],
    n_info: Dict[str, Any],
    seed: int = 1,
    n_train: int = 1024,
    n_test: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates training data with 0 points inside the central OOD gap [4.0, 6.0],
    and test data with a 70% In-Distribution / 30% OOD mixture.
    Sample sizes are scaled with dimension to the nearest power of 2 for Sobol balance.
    """
    dim = f_info["dim"]
    domain = f_info["domain"]

    # Scaled power-of-2 sample sizes
    n_train_dim = 2 ** math.ceil(math.log2(n_train * dim))
    n_test_dim = 2 ** math.ceil(math.log2(n_test * dim))

    np.random.seed(seed)

    # -------------------------------------------------------------
    # 1. Training Set: 100% In-Distribution (Zero points in [4.0, 6.0])
    # -------------------------------------------------------------
    n_train_half = n_train_dim // 2
    if dim == 1:
        step_l = 4.0 / n_train_half
        X_train_left = np.linspace(domain[0], 4.0 - step_l, n_train_half).reshape(-1, 1)
        step_r = (domain[1] - 6.0) / (n_train_dim - n_train_half)
        X_train_right = np.linspace(6.0 + step_r, domain[1], n_train_dim - n_train_half).reshape(-1, 1)
        X_train = np.vstack([X_train_left, X_train_right])
    else:
        sampler_train_l = Sobol(d=dim, scramble=True, seed=seed)
        u_l = sampler_train_l.random(n=n_train_half)
        X_train_left = np.empty_like(u_l)
        X_train_left[:, 0] = domain[0] + (3.99999 - domain[0]) * u_l[:, 0]
        for d in range(1, dim):
            X_train_left[:, d] = domain[0] + (domain[1] - domain[0]) * u_l[:, d]

        sampler_train_r = Sobol(d=dim, scramble=True, seed=seed + 5000)
        u_r = sampler_train_r.random(n=n_train_dim - n_train_half)
        X_train_right = np.empty_like(u_r)
        X_train_right[:, 0] = 6.00001 + (domain[1] - 6.00001) * u_r[:, 0]
        for d in range(1, dim):
            X_train_right[:, d] = domain[0] + (domain[1] - domain[0]) * u_r[:, d]

        X_train = np.vstack([X_train_left, X_train_right])

    # -------------------------------------------------------------
    # 2. Test Set: 70% In-Distribution / 30% Out-of-Distribution
    # -------------------------------------------------------------
    n_ood = int(round(0.30 * n_test_dim))
    n_id = n_test_dim - n_ood
    n_id_left = n_id // 2
    n_id_right = n_id - n_id_left

    if dim == 1:
        X_test_id_l = np.linspace(domain[0], 4.0, n_id_left, endpoint=False).reshape(-1, 1)
        X_test_id_r = np.linspace(6.0, domain[1], n_id_right, endpoint=True).reshape(-1, 1)
        X_test_ood = np.linspace(4.0, 6.0, n_ood, endpoint=False).reshape(-1, 1)
        X_test = np.vstack([X_test_id_l, X_test_id_r, X_test_ood])
    else:
        sampler_test_l = Sobol(d=dim, scramble=True, seed=seed + 10000)
        u_tl = sampler_test_l.random(n=n_id_left)
        X_test_id_l = np.empty_like(u_tl)
        X_test_id_l[:, 0] = domain[0] + (4.0 - domain[0]) * u_tl[:, 0]
        for d in range(1, dim):
            X_test_id_l[:, d] = domain[0] + (domain[1] - domain[0]) * u_tl[:, d]

        sampler_test_r = Sobol(d=dim, scramble=True, seed=seed + 15000)
        u_tr = sampler_test_r.random(n=n_id_right)
        X_test_id_r = np.empty_like(u_tr)
        X_test_id_r[:, 0] = 6.0 + (domain[1] - 6.0) * u_tr[:, 0]
        for d in range(1, dim):
            X_test_id_r[:, d] = domain[0] + (domain[1] - domain[0]) * u_tr[:, d]

        sampler_test_ood = Sobol(d=dim, scramble=True, seed=seed + 20000)
        u_tood = sampler_test_ood.random(n=n_ood)
        X_test_ood = np.empty_like(u_tood)
        X_test_ood[:, 0] = 4.0 + (6.0 - 4.0) * u_tood[:, 0]
        for d in range(1, dim):
            X_test_ood[:, d] = domain[0] + (domain[1] - domain[0]) * u_tood[:, d]

        X_test = np.vstack([X_test_id_l, X_test_id_r, X_test_ood])

    is_ood_test = (X_test[:, 0] >= 4.0) & (X_test[:, 0] <= 6.0)

    # Calculate targets with heteroscedastic / homoscedastic noise
    y_clean_train = f_info["func"](X_train)
    sigma_true_train = n_info["func"](X_train)
    y_train = y_clean_train + np.random.normal(0, sigma_true_train)

    y_clean_test = f_info["func"](X_test)
    sigma_true_test = n_info["func"](X_test)
    y_test = y_clean_test + np.random.normal(0, sigma_true_test)

    return X_train, y_train, X_test, y_test, sigma_true_test, is_ood_test

# =====================================================================
# 5. Extract Uncertainty Signals (5 Aleatoric + 1 Epistemic Baseline)
# =====================================================================
def extract_all_uncertainty_signals(
    rf: RandomForestRegressor,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    min_sigma2: float = 1e-6
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Extracts leaf variances across all trees and computes:
    - 5 Aleatoric estimators (Shaker Entropy, Shaker Geo Var/Std, Standard Ari Var/Std)
    - 1 Epistemic baseline (Standard Disagreement / Ensemble Variance)
    """
    n_trees = len(rf.estimators_)
    n_test = len(X_test)

    # 1. Prediction of each tree and mean ensemble prediction
    tree_preds = np.zeros((n_trees, n_test), dtype=np.float64)
    for t_idx, tree in enumerate(rf.estimators_):
        tree_preds[t_idx] = tree.predict(X_test)
    y_pred_mean = np.mean(tree_preds, axis=0)

    # 2. Epistemic Baseline: Ensemble Disagreement (Prediction Variance)
    standard_disagreement = np.var(tree_preds, axis=0, ddof=1)

    # 3. Leaf Sample Variance Extraction
    vars2 = np.zeros((n_trees, n_test), dtype=np.float64)
    for t_idx, tree in enumerate(rf.estimators_):
        train_leaf_ids = tree.apply(X_train)
        test_leaf_ids = tree.apply(X_test)

        leaf_map: Dict[int, list] = {}
        for row_idx, leaf_id in enumerate(train_leaf_ids):
            leaf_map.setdefault(leaf_id, []).append(y_train[row_idx])

        leaf_vars = {}
        for leaf_id, target_vals in leaf_map.items():
            if len(target_vals) > 1:
                v = float(np.var(target_vals, ddof=1))
            else:
                v = 0.0
            leaf_vars[leaf_id] = max(v, min_sigma2)

        for i, leaf_id in enumerate(test_leaf_ids):
            vars2[t_idx, i] = leaf_vars.get(leaf_id, min_sigma2)

    # Shaker Differential Entropy: H_a(x) = (1/M) * sum 0.5 * log2(2*pi*e*sigma_m^2)
    ind_entropies = 0.5 * np.log2(2.0 * np.pi * np.e * vars2)
    shaker_entropy = np.mean(ind_entropies, axis=0)

    # Shaker Geometric Mean Variance: sigma_{geom}^2 = (1 / 2*pi*e) * 2^(2 * H_a)
    shaker_geom_var = (2.0 ** (2.0 * shaker_entropy)) / (2.0 * np.pi * np.e)
    shaker_geom_std = np.sqrt(np.maximum(shaker_geom_var, 1e-12))

    # Standard Arithmetic Mean Variance: sigma_{ari}^2 = (1/M) * sum sigma_m^2
    standard_ari_var = np.mean(vars2, axis=0)
    standard_ari_std = np.sqrt(np.maximum(standard_ari_var, 1e-12))

    signals = {
        "shaker_entropy": shaker_entropy,
        "shaker_geom_var": shaker_geom_var,
        "shaker_geom_std": shaker_geom_std,
        "standard_ari_var": standard_ari_var,
        "standard_ari_std": standard_ari_std,
        "standard_disagreement": standard_disagreement
    }
    return signals, y_pred_mean

# =====================================================================
# 6. Sliced Evaluation Metrics Engine (Global, ID-Only, OOD-Only)
# =====================================================================
def _calc_raw_metrics(
    unc_signal: np.ndarray,
    sigma_true: np.ndarray,
    abs_residual: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Helper computing the 6 core metrics on an arbitrary slice of data."""
    sigma_true_var = sigma_true ** 2
    squared_residual = abs_residual ** 2

    std_unc = np.std(unc_signal)
    std_sigma_true = np.std(sigma_true_var)
    std_residual = np.std(squared_residual)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        if std_sigma_true < 1e-12:
            sp_true = np.nan
        elif std_unc < 1e-12:
            sp_true = 0.0
        else:
            sp_true, _ = spearmanr(unc_signal, sigma_true_var)

        if std_residual < 1e-12 or std_unc < 1e-12:
            sp_resid = 0.0
        else:
            sp_resid, _ = spearmanr(unc_signal, squared_residual)

        if std_sigma_true < 1e-12 or std_unc < 1e-12:
            lp_true = np.nan
        else:
            valid_mask = (unc_signal > 1e-12) & (sigma_true_var > 1e-12)
            if np.sum(valid_mask) > 5 and np.std(np.log(unc_signal[valid_mask])) > 1e-12 and np.std(np.log(sigma_true_var[valid_mask])) > 1e-12:
                lp_true, _ = pearsonr(np.log(unc_signal[valid_mask]), np.log(sigma_true_var[valid_mask]))
            else:
                lp_true = 0.0

    mse_var = float(np.mean((unc_signal - sigma_true_var) ** 2))
    rmse_var = float(np.sqrt(mse_var))

    var_clamped = np.maximum(unc_signal, 1e-8)
    nlpd_val = float(np.mean(0.5 * ((y_test - y_pred) ** 2 / var_clamped) + 0.5 * np.log(2.0 * np.pi * var_clamped)))

    return {
        "spearman_true": float(sp_true) if not np.isnan(sp_true) else 0.0,
        "spearman_resid": float(sp_resid) if not np.isnan(sp_resid) else 0.0,
        "log_pearson_true": float(lp_true) if not np.isnan(lp_true) else 0.0,
        "mse_var": float(mse_var),
        "rmse_var": float(rmse_var),
        "nlpd_aleatoric": float(nlpd_val)
    }

def evaluate_sliced_aleatoric_metrics(
    unc_signal: np.ndarray,
    sigma_true: np.ndarray,
    abs_residual: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    is_ood_test: np.ndarray
) -> Dict[str, float]:
    """Computes all 6 metrics across Global, ID-only, and OOD-only partitions."""
    out = {}

    # 1. Global Slices
    m_global = _calc_raw_metrics(unc_signal, sigma_true, abs_residual, y_test, y_pred)
    for k, v in m_global.items():
        out[f"global_{k}"] = v

    # 2. ID-Only Slice
    id_mask = ~is_ood_test
    if np.sum(id_mask) > 5:
        m_id = _calc_raw_metrics(
            unc_signal[id_mask],
            sigma_true[id_mask],
            abs_residual[id_mask],
            y_test[id_mask],
            y_pred[id_mask]
        )
        for k, v in m_id.items():
            out[f"id_only_{k}"] = v
    else:
        for k in m_global.keys():
            out[f"id_only_{k}"] = 0.0

    # 3. OOD-Only Slice
    ood_mask = is_ood_test
    if np.sum(ood_mask) > 5:
        m_ood = _calc_raw_metrics(
            unc_signal[ood_mask],
            sigma_true[ood_mask],
            abs_residual[ood_mask],
            y_test[ood_mask],
            y_pred[ood_mask]
        )
        for k, v in m_ood.items():
            out[f"ood_only_{k}"] = v
    else:
        for k in m_global.keys():
            out[f"ood_only_{k}"] = 0.0

    # 4. OOD / ID Variance Ratio
    mean_id_unc = float(np.mean(unc_signal[id_mask])) if np.sum(id_mask) > 0 else 1.0
    mean_ood_unc = float(np.mean(unc_signal[ood_mask])) if np.sum(ood_mask) > 0 else 1.0
    out["ood_id_variance_ratio"] = float(mean_ood_unc / max(mean_id_unc, 1e-8))

    return out

# =====================================================================
# 7. Single OOD Experiment Execution Entrypoint
# =====================================================================
def run_single_aleatoric_ood_experiment(
    func_name: str,
    noise_name: str,
    rf_config_name: str,
    seed: int = 1,
    n_train: int = 1024,
    n_test: int = 256
) -> Dict[str, Dict[str, float]]:
    """Executes a single benchmark run comparing all 6 approaches across all sliced metrics."""
    funcs = get_benchmark_functions()
    noises = get_noise_regimes()
    rf_configs = get_rf_configs()

    f_info = funcs[func_name]
    n_info = noises[noise_name]
    rf_cfg = rf_configs[rf_config_name]

    # Generate partitioned dataset
    X_train, y_train, X_test, y_test, sigma_true_test, is_ood_test = generate_ood_aleatoric_dataset(
        f_info=f_info,
        n_info=n_info,
        seed=seed,
        n_train=n_train,
        n_test=n_test
    )

    # Train Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        max_depth=rf_cfg["max_depth"],
        random_state=seed,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Extract signals and predictions
    signals, y_pred = extract_all_uncertainty_signals(
        rf=rf,
        X_test=X_test,
        X_train=X_train,
        y_train=y_train
    )

    abs_residual = np.abs(y_test - y_pred)

    results = {}
    for app_name, unc_signal in signals.items():
        metrics = evaluate_sliced_aleatoric_metrics(
            unc_signal=unc_signal,
            sigma_true=sigma_true_test,
            abs_residual=abs_residual,
            y_test=y_test,
            y_pred=y_pred,
            is_ood_test=is_ood_test
        )
        results[app_name] = metrics

    return results

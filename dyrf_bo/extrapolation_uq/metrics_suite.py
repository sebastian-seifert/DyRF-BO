"""Evaluation metrics suite for extrapolation uncertainty quantification.

Implements Epistemic Distance Monotonicity (Spearman rho), Error Alignment,
Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW),
Winkler Interval Score, and Catastrophic Outlier Error Discrimination (AUROC / AUPRC).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

if TYPE_CHECKING:
    from .uq_evaluator import DualUQResult, MultiUQResult



def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two 1D arrays.

    Safely handles constant arrays and zero-variance inputs by returning 0.0.

    Parameters
    ----------
    x : np.ndarray
        First 1D array of values.
    y : np.ndarray
        Second 1D array of values.

    Returns
    -------
    float
        Spearman rank correlation coefficient in [-1, 1], or 0.0 if degenerate.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if len(x_arr) < 2 or len(y_arr) < 2:
        return 0.0

    if np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        return 0.0

    res = spearmanr(x_arr, y_arr)
    stat = float(res.statistic)
    if np.isnan(stat):
        return 0.0
    return stat


def prediction_interval_coverage_probability(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    uncertainty: np.ndarray,
) -> float:
    """Compute empirical coverage probability: fraction of points with |y_true - y_hat| <= U.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_hat : np.ndarray
        Model point predictions.
    uncertainty : np.ndarray
        Half-width uncertainty estimates U(x) >= 0.

    Returns
    -------
    float
        Coverage probability in [0, 1].
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_h = np.asarray(y_hat, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    if len(y_t) == 0:
        return 0.0

    abs_err = np.abs(y_t - y_h)
    covered = abs_err <= u
    return float(np.mean(covered))


def mean_prediction_interval_width(uncertainty: np.ndarray) -> float:
    """Compute Mean Prediction Interval Width: mean(2 * uncertainty).

    Parameters
    ----------
    uncertainty : np.ndarray
        Half-width uncertainty estimates U(x).

    Returns
    -------
    float
        Mean interval width.
    """
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if len(u) == 0:
        return 0.0
    return float(np.mean(2.0 * u))


def winkler_interval_score(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute mean Winkler score for symmetric prediction intervals.

    Formula:
        S_alpha(x) = 2 * U(x) + (2 / alpha) * max(0, (y_hat - U) - y)
                              + (2 / alpha) * max(0, y - (y_hat + U))

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_hat : np.ndarray
        Model point predictions.
    uncertainty : np.ndarray
        Half-width uncertainty estimates U(x).
    alpha : float, default=0.05
        Nominal error level (1 - confidence_level). Default 0.05 corresponds to 95% interval.

    Returns
    -------
    float
        Mean Winkler interval score.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_h = np.asarray(y_hat, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    if len(y_t) == 0:
        return 0.0
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    penalty_scale = 2.0 / alpha
    lower_viol = np.maximum(0.0, (y_h - u) - y_t)
    upper_viol = np.maximum(0.0, y_t - (y_h + u))
    scores = 2.0 * u + penalty_scale * (lower_viol + upper_viol)
    return float(np.mean(scores))


def compute_regression_ece(
    y_true: np.ndarray,
    interval_dict: Dict[Any, Tuple[np.ndarray, np.ndarray]],
    nominal_levels: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) for regression intervals (Schneider et al., 2026).

    Formula:
        ECE = (1 / m) * sum_{i=1}^m |hat_alpha_i - alpha_i|
        hat_alpha_i = (1 / N) * sum_{j=1}^N I(y_j not in [l_j^{1-alpha_i}, u_j^{1-alpha_i}])

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets of shape (N,).
    interval_dict : dict
        Mapping from nominal alpha level (float or suffix string '50', '80', '90', '95')
        to (int_lwr, int_upr) arrays.
    nominal_levels : tuple of float, default=(0.05, 0.10, 0.20, 0.50)
        Miscoverage levels alpha_i to evaluate.

    Returns
    -------
    dict[str, float]
        Dictionary with 'ece', per-level empirical miscoverage, and absolute errors.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    n = len(y_t)
    if n == 0:
        return {"ece": 0.0}

    sfx_map = {0.05: "95", 0.10: "90", 0.20: "80", 0.50: "50"}
    alpha_sfx = {0.05: "05", 0.10: "10", 0.20: "20", 0.50: "50"}

    errs = []
    out: Dict[str, float] = {}

    for alpha in nominal_levels:
        sfx = sfx_map.get(round(alpha, 2), str(int(round((1 - alpha) * 100))))
        a_sfx = alpha_sfx.get(round(alpha, 2), str(int(round(alpha * 100))))

        # Look up bounds
        if alpha in interval_dict:
            lwr, upr = interval_dict[alpha]
        elif sfx in interval_dict:
            lwr, upr = interval_dict[sfx]
        elif a_sfx in interval_dict:
            lwr, upr = interval_dict[a_sfx]
        else:
            raise KeyError(f"Interval bounds for alpha={alpha} (key '{sfx}' or {alpha}) not found in interval_dict.")

        lwr_arr = np.asarray(lwr, dtype=np.float64).ravel()
        upr_arr = np.asarray(upr, dtype=np.float64).ravel()

        miscovered = (y_t < lwr_arr) | (y_t > upr_arr)
        hat_alpha = float(np.mean(miscovered))
        err = float(abs(hat_alpha - alpha))

        out[f"miscoverage_{sfx}"] = hat_alpha
        out[f"err_{sfx}"] = err
        errs.append(err)

    out["ece"] = float(np.mean(errs)) if errs else 0.0
    return out


def compute_multi_level_winkler(
    y_true: np.ndarray,
    interval_dict: Dict[Any, Tuple[np.ndarray, np.ndarray]],
    nominal_levels: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50),
) -> Dict[str, float]:
    """Compute Winkler scores across multiple nominal miscoverage levels (Gneiting & Ziegel).

    Formula:
        W_alpha([l, u], y) = (u - l) + (2/alpha)*(l - y)*I(y < l) + (2/alpha)*(y - u)*I(y > u)

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth targets of shape (N,).
    interval_dict : dict
        Mapping from nominal alpha level (float or suffix string '50', '80', '90', '95')
        to (int_lwr, int_upr) arrays.
    nominal_levels : tuple of float, default=(0.05, 0.10, 0.20, 0.50)
        Miscoverage levels alpha_i to evaluate.

    Returns
    -------
    dict[str, float]
        Dictionary with per-level scores 'winkler_{sfx}' and their unweighted mean 'winkler_mean'.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    n = len(y_t)
    if n == 0:
        return {"winkler_mean": 0.0}

    sfx_map = {0.05: "95", 0.10: "90", 0.20: "80", 0.50: "50"}
    alpha_sfx = {0.05: "05", 0.10: "10", 0.20: "20", 0.50: "50"}

    scores = []
    out: Dict[str, float] = {}

    for alpha in nominal_levels:
        sfx = sfx_map.get(round(alpha, 2), str(int(round((1 - alpha) * 100))))
        a_sfx = alpha_sfx.get(round(alpha, 2), str(int(round(alpha * 100))))

        if alpha in interval_dict:
            lwr, upr = interval_dict[alpha]
        elif sfx in interval_dict:
            lwr, upr = interval_dict[sfx]
        elif a_sfx in interval_dict:
            lwr, upr = interval_dict[a_sfx]
        else:
            raise KeyError(f"Interval bounds for alpha={alpha} (key '{sfx}' or {alpha}) not found in interval_dict.")

        lwr_arr = np.asarray(lwr, dtype=np.float64).ravel()
        upr_arr = np.asarray(upr, dtype=np.float64).ravel()

        width = upr_arr - lwr_arr
        penalty_scale = 2.0 / alpha
        l_viol = np.maximum(0.0, lwr_arr - y_t)
        u_viol = np.maximum(0.0, y_t - upr_arr)
        pt_scores = width + penalty_scale * (l_viol + u_viol)
        mean_score = float(np.mean(pt_scores))

        out[f"winkler_{sfx}"] = mean_score
        scores.append(mean_score)

    out["winkler_mean"] = float(np.mean(scores)) if scores else 0.0
    return out


def compute_regression_eaurc(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    uncertainty: np.ndarray,
) -> Dict[str, float]:
    """Compute regression-adapted Excess Area Under the Risk-Coverage Curve (E-AURC).

    Follows El-Yaniv & Wiener (2010) selective risk with MAE loss and Geifman et al. (2019)
    Excess AURC against the oracle error ordering.

    Applies analytical tie-averaging: points sharing identical uncertainty values receive
    their group-mean absolute error before prefix-risk computation, guaranteeing exact
    permutation-invariance and convergence to the analytical random limit (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (N,).
    y_hat : np.ndarray
        Model point predictions of shape (N,).
    uncertainty : np.ndarray
        Predicted uncertainty scores of shape (N,).

    Returns
    -------
    dict[str, float]
        Dictionary containing 'e_aurc', 'aurc', 'oracle_aurc', 'random_aurc', and 'naurc'.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_h = np.asarray(y_hat, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    n = len(y_t)
    if n == 0:
        return {"e_aurc": 0.0, "aurc": 0.0, "oracle_aurc": 0.0, "random_aurc": 0.0, "naurc": 0.0}

    abs_errors = np.abs(y_t - y_h)
    random_aurc = float(np.mean(abs_errors))

    # 1. Oracle AURC: Sort ascending by true absolute error
    sorted_oracle_errors = np.sort(abs_errors)
    oracle_risk = np.cumsum(sorted_oracle_errors) / np.arange(1, n + 1)
    oracle_aurc = float(np.mean(oracle_risk))

    # 2. Model AURC with analytical tie-averaging
    # If uncertainty values contain duplicates, assign the group-averaged error to each tie
    unique_u, inverse_indices, counts = np.unique(u, return_inverse=True, return_counts=True)
    if len(unique_u) < n:
        # Sum errors per unique uncertainty group
        group_error_sums = np.bincount(inverse_indices, weights=abs_errors)
        group_error_means = group_error_sums / counts
        # Replace each point's individual error with its tie-group mean
        effective_errors = group_error_means[inverse_indices]
    else:
        effective_errors = abs_errors

    # Sort ascending by uncertainty (lower uncertainty retained first)
    sort_idx = np.argsort(u, kind="mergesort")
    sorted_model_errors = effective_errors[sort_idx]
    model_risk = np.cumsum(sorted_model_errors) / np.arange(1, n + 1)
    aurc = float(np.mean(model_risk))

    e_aurc = float(max(0.0, aurc - oracle_aurc))

    denom = random_aurc - oracle_aurc
    if denom < 1e-12:
        naurc = 0.0
    else:
        naurc = float(np.clip(e_aurc / denom, 0.0, 5.0))

    return {
        "e_aurc": e_aurc,
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "random_aurc": random_aurc,
        "naurc": naurc,
    }


def outlier_error_detection_auc(
    abs_errors: np.ndarray,
    uncertainty: np.ndarray,
    top_quantile: float = 0.90,
) -> Tuple[float, float]:
    """Compute AUROC and AUPRC for catastrophic error outlier detection.

    Binary target: Y = I(abs_error > Q_{top_quantile}(abs_error)).

    Parameters
    ----------
    abs_errors : np.ndarray
        Absolute residual errors |y_true - y_hat|.
    uncertainty : np.ndarray
        Uncertainty detector scores U(x).
    top_quantile : float, default=0.90
        Quantile cutoff for outlier definition.

    Returns
    -------
    tuple[float, float]
        (auroc, auprc) scores.
    """
    e = np.asarray(abs_errors, dtype=np.float64).ravel()
    u = np.asarray(uncertainty, dtype=np.float64).ravel()

    if len(e) == 0:
        return (0.5, 0.0)

    q = float(np.quantile(e, top_quantile))
    y_binary = (e > q).astype(np.int64)

    if len(np.unique(y_binary)) < 2:
        return (0.5, float(np.mean(y_binary)))

    try:
        auroc = float(roc_auc_score(y_binary, u))
        auprc = float(average_precision_score(y_binary, u))
        return (auroc, auprc)
    except Exception:
        return (0.5, 0.0)


def _compute_single_method_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    uncertainty: np.ndarray,
    d_norm: np.ndarray,
    abs_errors: np.ndarray,
) -> Dict[str, float]:
    """Helper to compute calibration and statistical metrics for one UQ method."""
    auroc, auprc = outlier_error_detection_auc(abs_errors, uncertainty)
    return {
        "spearman_dist": spearman_rank_correlation(d_norm, uncertainty),
        "spearman_err": spearman_rank_correlation(abs_errors, uncertainty),
        "picp": prediction_interval_coverage_probability(y_true, y_hat, uncertainty),
        "mpiw": mean_prediction_interval_width(uncertainty),
        "winkler": winkler_interval_score(y_true, y_hat, uncertainty),
        "auroc": auroc,
        "auprc": auprc,
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    u_slcb: np.ndarray | Any | None = None,
    u_plcb: np.ndarray | None = None,
    d_norm: np.ndarray | None = None,
    strata_labels: np.ndarray | None = None,
    *,
    uq_result: MultiUQResult | DualUQResult | Dict[str, np.ndarray] | None = None,
) -> Dict[str, Any]:
    """Compute calibration and ranking metrics for all candidate UQ methods.

    Evaluates the complete 7-metric scorecard (spearman_dist, spearman_err,
    picp, mpiw, winkler, auroc, auprc) across all uncertainty signals present,
    both globally and partitioned per stratum (0, 1, 2, 3).

    Supports both legacy calls with (u_slcb, u_plcb) and new MultiUQResult /
    dictionary interfaces via the uq_result parameter.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (M,).
    y_hat : np.ndarray
        Model point predictions of shape (M,).
    u_slcb : np.ndarray | Any | None, default=None
        Standard SMAC3 LCB uncertainties or positional uq_result container.
    u_plcb : np.ndarray | None, default=None
        Proximity LCB uncertainties of shape (M,) or d_norm if uq_result was 3rd arg.
    d_norm : np.ndarray | None, default=None
        Normalized convex hull distances of shape (M,).
    strata_labels : np.ndarray | None, default=None
        Optional integer strata labels in {0, 1, 2, 3} of shape (M,).
    uq_result : MultiUQResult | DualUQResult | Dict[str, np.ndarray] | None, default=None
        Evaluator container or mapping of candidate uncertainty signals.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed metrics globally and per stratum.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_h = np.asarray(y_hat, dtype=np.float64).ravel()
    d_n = np.asarray(d_norm, dtype=np.float64).ravel() if d_norm is not None else np.zeros_like(y_t)
    abs_e = np.abs(y_t - y_h)

    # 1. Resolve uncertainties dictionary
    resolved_uq = uq_result
    if resolved_uq is None:
        if u_slcb is not None and (
            hasattr(u_slcb, "get_uncertainties_dict")
            or (isinstance(u_slcb, dict) and not isinstance(u_slcb, np.ndarray))
        ):
            resolved_uq = u_slcb
            if d_norm is None and u_plcb is not None:
                d_n = np.asarray(u_plcb, dtype=np.float64).ravel()

    uncertainties_map: Dict[str, np.ndarray] = {}
    if resolved_uq is not None:
        if hasattr(resolved_uq, "get_uncertainties_dict"):
            uncertainties_map = dict(resolved_uq.get_uncertainties_dict())
        elif isinstance(resolved_uq, dict):
            excluded = {
                "y_hat",
                "y_true",
                "abs_error",
                "d_norm",
                "d_rel",
                "d_inf",
                "is_interpolating",
                "stratum",
                "strata",
                "point_id",
                "var_between",
                "var_within",
                "var_total",
                "delta_floor",
                "local_mae",
                "q_lower",
                "global",
            }
            uncertainties_map = {
                k: np.asarray(v, dtype=np.float64).ravel()
                for k, v in resolved_uq.items()
                if k not in excluded and hasattr(v, "__len__")
            }
        elif hasattr(resolved_uq, "__dict__"):
            uncertainties_map = {
                k: np.asarray(getattr(resolved_uq, k), dtype=np.float64).ravel()
                for k in dir(resolved_uq)
                if (k.startswith("u_") or k in {"slcb", "plcb"})
                and isinstance(getattr(resolved_uq, k), np.ndarray)
            }

    # Fallback to legacy positional arguments if not in uncertainties_map
    if u_slcb is not None and isinstance(u_slcb, np.ndarray):
        arr_slcb = np.asarray(u_slcb, dtype=np.float64).ravel()
        if "u_slcb" not in uncertainties_map:
            uncertainties_map["u_slcb"] = arr_slcb
        if "slcb" not in uncertainties_map:
            uncertainties_map["slcb"] = arr_slcb

    if u_plcb is not None and isinstance(u_plcb, np.ndarray):
        arr_plcb = np.asarray(u_plcb, dtype=np.float64).ravel()
        if "u_plcb" not in uncertainties_map:
            uncertainties_map["u_plcb"] = arr_plcb
        if "plcb" not in uncertainties_map:
            uncertainties_map["plcb"] = arr_plcb

    # Synchronize legacy aliases
    if "u_slcb" in uncertainties_map and "slcb" not in uncertainties_map:
        uncertainties_map["slcb"] = uncertainties_map["u_slcb"]
    elif "slcb" in uncertainties_map and "u_slcb" not in uncertainties_map:
        uncertainties_map["u_slcb"] = uncertainties_map["slcb"]

    if "u_plcb" in uncertainties_map and "plcb" not in uncertainties_map:
        uncertainties_map["plcb"] = uncertainties_map["u_plcb"]
    elif "plcb" in uncertainties_map and "u_plcb" not in uncertainties_map:
        uncertainties_map["u_plcb"] = uncertainties_map["plcb"]

    # 2. Global Metrics
    global_metrics: Dict[str, Dict[str, float]] = {}
    result: Dict[str, Any] = {"global": global_metrics}

    for method_name, u_arr in uncertainties_map.items():
        m_dict = _compute_single_method_metrics(y_t, y_h, u_arr, d_n, abs_e)
        global_metrics[method_name] = m_dict
        if method_name.startswith("u_"):
            stripped = method_name[2:]
            if stripped not in global_metrics:
                global_metrics[stripped] = m_dict

        # Flatten global keys for convenient root access
        for k, v in m_dict.items():
            result[f"{method_name}_{k}"] = v
            result[f"{k}_{method_name}"] = v
            if method_name.startswith("u_"):
                stripped = method_name[2:]
                result[f"{stripped}_{k}"] = v
                result[f"{k}_{stripped}"] = v

    # 3. Strata Breakdown
    if strata_labels is not None:
        s_arr = np.asarray(strata_labels, dtype=np.int64).ravel()
        unique_strata = np.unique(s_arr)
        strata_dict: Dict[int, Dict[str, Dict[str, float]]] = {}

        for s in unique_strata:
            s_int = int(s)
            mask = s_arr == s
            if not np.any(mask):
                continue

            strata_dict[s_int] = {}
            for method_name, u_arr in uncertainties_map.items():
                m_s = _compute_single_method_metrics(
                    y_t[mask], y_h[mask], u_arr[mask], d_n[mask], abs_e[mask]
                )
                strata_dict[s_int][method_name] = m_s
                if method_name.startswith("u_"):
                    stripped = method_name[2:]
                    if stripped not in strata_dict[s_int]:
                        strata_dict[s_int][stripped] = m_s

                # Root level flattened keys for strata
                for k, v in m_s.items():
                    result[f"stratum_{s_int}_{method_name}_{k}"] = v
                    result[f"stratum_{s_int}_{k}_{method_name}"] = v
                    if method_name.startswith("u_"):
                        stripped = method_name[2:]
                        result[f"stratum_{s_int}_{stripped}_{k}"] = v
                        result[f"stratum_{s_int}_{k}_{stripped}"] = v

        result["strata"] = strata_dict
        result["strata_metrics"] = strata_dict

    return result


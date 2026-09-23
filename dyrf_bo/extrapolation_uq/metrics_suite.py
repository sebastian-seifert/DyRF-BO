"""Evaluation metrics suite for extrapolation uncertainty quantification.

Implements Epistemic Distance Monotonicity (Spearman rho), Error Alignment,
Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW),
Winkler Interval Score, and Catastrophic Outlier Error Discrimination (AUROC / AUPRC).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


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
    u_slcb: np.ndarray,
    u_plcb: np.ndarray,
    d_norm: np.ndarray,
    strata_labels: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute calibration and ranking metrics for both SLCB and PLCB.

    Computes all metrics globally and, if strata_labels is provided, partitioned
    per stratum (0, 1, 2, 3).

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (M,).
    y_hat : np.ndarray
        Model point predictions of shape (M,).
    u_slcb : np.ndarray
        Standard SMAC3 LCB uncertainties of shape (M,).
    u_plcb : np.ndarray
        Proximity LCB uncertainties of shape (M,).
    d_norm : np.ndarray
        Normalized convex hull distances of shape (M,).
    strata_labels : np.ndarray | None, default=None
        Optional integer strata labels in {0, 1, 2, 3} of shape (M,).

    Returns
    -------
    dict[str, Any]
        Dictionary of computed metrics globally and per stratum.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_h = np.asarray(y_hat, dtype=np.float64).ravel()
    u_s = np.asarray(u_slcb, dtype=np.float64).ravel()
    u_p = np.asarray(u_plcb, dtype=np.float64).ravel()
    d_n = np.asarray(d_norm, dtype=np.float64).ravel()
    abs_e = np.abs(y_t - y_h)

    # 1. Global Metrics
    slcb_global = _compute_single_method_metrics(y_t, y_h, u_s, d_n, abs_e)
    plcb_global = _compute_single_method_metrics(y_t, y_h, u_p, d_n, abs_e)

    result: Dict[str, Any] = {
        "global": {
            "slcb": slcb_global,
            "plcb": plcb_global,
        }
    }

    # Flatten global keys for convenient root access
    for k, v in slcb_global.items():
        result[f"slcb_{k}"] = v
        result[f"{k}_slcb"] = v
    for k, v in plcb_global.items():
        result[f"plcb_{k}"] = v
        result[f"{k}_plcb"] = v

    # 2. Strata Breakdown
    if strata_labels is not None:
        s_arr = np.asarray(strata_labels, dtype=np.int64).ravel()
        unique_strata = np.unique(s_arr)
        strata_dict: Dict[int, Dict[str, Dict[str, float]]] = {}

        for s in unique_strata:
            s_int = int(s)
            mask = s_arr == s
            if not np.any(mask):
                continue

            slcb_s = _compute_single_method_metrics(
                y_t[mask], y_h[mask], u_s[mask], d_n[mask], abs_e[mask]
            )
            plcb_s = _compute_single_method_metrics(
                y_t[mask], y_h[mask], u_p[mask], d_n[mask], abs_e[mask]
            )

            strata_dict[s_int] = {
                "slcb": slcb_s,
                "plcb": plcb_s,
            }

            for k, v in slcb_s.items():
                result[f"stratum_{s_int}_slcb_{k}"] = v
                result[f"stratum_{s_int}_{k}_slcb"] = v
            for k, v in plcb_s.items():
                result[f"stratum_{s_int}_plcb_{k}"] = v
                result[f"stratum_{s_int}_{k}_plcb"] = v

        result["strata"] = strata_dict

    return result

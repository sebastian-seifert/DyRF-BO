"""Multi-UQ and Dual Uncertainty Quantification Inference Engine.

Evaluates 10 candidate uncertainty quantification methods across tree ensembles:
2 Non-Proximity:
  1. Hutter Total: 1.96 * sqrt(var_between + var_within) [Alias: u_slcb]
  2. Shaker Epistemic: 1.96 * sigma_aleatoric * sqrt(4^MI - 1) via Huber closed-form entropy
8 Proximity (All unweighted topological tree walk across OOB and 5-fold CV residual pools):
  3. Proximity A OOB: pure unweighted topological path distance fixed-k nearest neighbors on OOB residuals
  4. Proximity A CV: pure unweighted topological path distance fixed-k nearest neighbors on 5-fold CV residuals
  5. Proximity B OOB: continuous weighted quantiles via exp(-lambda * d_t) on OOB residuals
  6. Proximity B CV: continuous weighted quantiles via exp(-lambda * d_t) on 5-fold CV residuals
  7. Proximity AC OOB: Proximity A on OOB residuals scaled by density multiplier gamma
  8. Proximity AC CV: Proximity A on 5-fold CV residuals scaled by density multiplier gamma
  9. Proximity BC OOB: Proximity B on OOB residuals scaled by density multiplier gamma
  10. Proximity BC CV: Proximity B on 5-fold CV residuals scaled by density multiplier gamma
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold

from Epistemic_Quantifier import EpistemicQuantifier
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ


APPROACH_NAMES: List[str] = [
    "hutter_total",
    "shaker_epistemic",
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]

NON_PROXIMITY_APPROACHES: List[str] = [
    "hutter_total",
    "shaker_epistemic",
]

PROXIMITY_APPROACHES: List[str] = [
    "prox_a_oob",
    "prox_a_cv",
    "prox_b_oob",
    "prox_b_cv",
    "prox_ac_oob",
    "prox_ac_cv",
    "prox_bc_oob",
    "prox_bc_cv",
]

COVERAGE_LEVELS: Tuple[float, ...] = (0.50, 0.80, 0.90, 0.95)
COVERAGE_SUFFIXES: Dict[float, str] = {0.50: "50", 0.80: "80", 0.90: "90", 0.95: "95"}


SURROGATE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smac_default": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 10,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "mature": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "shallow": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 25,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_samples_leaf": 5,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "coarse": {
        "estimator_class": ExtraTreesRegressor,
        "model_cls": ExtraTreesRegressor,
        "model_class": ExtraTreesRegressor,
        "n_estimators": 25,
        "max_depth": None,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
    "breiman": {
        "estimator_class": RandomForestRegressor,
        "model_cls": RandomForestRegressor,
        "model_class": RandomForestRegressor,
        "n_estimators": 25,
        "max_depth": None,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "max_features": 5.0 / 6.0,
        "bootstrap": True,
        "oob_score": True,
        "criterion": "squared_error",
    },
}


def create_surrogate_rf(
    surrogate_type: str = "smac_default",
    seed: Optional[int] = None,
    model: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Create an ensemble regressor based on one of the 5 surrogate configurations or reuse existing."""
    if surrogate_type not in SURROGATE_CONFIGS:
        raise ValueError(
            f"Unknown surrogate_type '{surrogate_type}'. "
            f"Supported surrogates: {list(SURROGATE_CONFIGS.keys())}"
        )

    if model is not None:
        if hasattr(model, "estimators_"):
            return model
        try:
            cloned = clone(model)
            if seed is not None and hasattr(cloned, "random_state"):
                cloned.random_state = seed
            return cloned
        except Exception:
            return model

    config = dict(SURROGATE_CONFIGS[surrogate_type])

    # Determine estimator class
    estimator_cls = config.get("estimator_class", ExtraTreesRegressor)
    if "estimator_class" in kwargs:
        estimator_cls = kwargs.pop("estimator_class")
    elif "model_class" in kwargs:
        estimator_cls = kwargs.pop("model_class")
    elif "model_cls" in kwargs:
        estimator_cls = kwargs.pop("model_cls")
    elif "use_extra_trees" in kwargs:
        use_extra = kwargs.pop("use_extra_trees")
        estimator_cls = ExtraTreesRegressor if use_extra else RandomForestRegressor

    # Clean out class metadata keys
    params = {
        k: v
        for k, v in config.items()
        if k not in ("estimator_class", "model_class", "model_cls")
    }
    params["random_state"] = seed
    params.update(kwargs)
    return estimator_cls(**params)


def create_smac_default_rf(
    seed: Optional[int] = None,
    n_trees: int = 10,
    model: Optional[Any] = None,
    use_extra_trees: bool = True,
    **kwargs: Any,
) -> Any:
    """Create an ensemble regressor with native SMAC3 defaults or reuse existing."""
    overrides = dict(kwargs)
    overrides["use_extra_trees"] = use_extra_trees
    if n_trees != 10 or "n_estimators" not in overrides:
        overrides["n_estimators"] = n_trees
    return create_surrogate_rf(
        surrogate_type="smac_default",
        seed=seed,
        model=model,
        **overrides,
    )


@dataclass
class MultiUQResult:
    """Container for predictions, uncertainties, and prediction intervals across 10 approaches.

    Supports both dataclass attribute access (e.g. `res.u_hutter_total`, `res.u_prox_a_oob`)
    and dictionary-style mapping access (e.g. `res['u_prox_a_oob']`, `res['u_plcb']`).
    """

    y_hat: np.ndarray

    # Group 1: Non-Proximity 1: Hutter Total
    u_hutter_total: np.ndarray
    sigma_hutter_total: np.ndarray
    int_lwr_hutter_total: np.ndarray
    int_upr_hutter_total: np.ndarray
    int_lwr_hutter_total_50: np.ndarray
    int_upr_hutter_total_50: np.ndarray
    int_lwr_hutter_total_80: np.ndarray
    int_upr_hutter_total_80: np.ndarray
    int_lwr_hutter_total_90: np.ndarray
    int_upr_hutter_total_90: np.ndarray
    int_lwr_hutter_total_95: np.ndarray
    int_upr_hutter_total_95: np.ndarray

    # Group 2: Non-Proximity 2: Shaker Epistemic
    u_shaker_epistemic: np.ndarray
    sigma_shaker_epistemic: np.ndarray
    int_lwr_shaker_epistemic: np.ndarray
    int_upr_shaker_epistemic: np.ndarray
    int_lwr_shaker_epistemic_50: np.ndarray
    int_upr_shaker_epistemic_50: np.ndarray
    int_lwr_shaker_epistemic_80: np.ndarray
    int_upr_shaker_epistemic_80: np.ndarray
    int_lwr_shaker_epistemic_90: np.ndarray
    int_upr_shaker_epistemic_90: np.ndarray
    int_lwr_shaker_epistemic_95: np.ndarray
    int_upr_shaker_epistemic_95: np.ndarray

    # Group 3: Proximity A (OOB & CV)
    u_prox_a_oob: np.ndarray
    sigma_prox_a_oob: np.ndarray
    int_lwr_prox_a_oob: np.ndarray
    int_upr_prox_a_oob: np.ndarray
    int_lwr_prox_a_oob_50: np.ndarray
    int_upr_prox_a_oob_50: np.ndarray
    int_lwr_prox_a_oob_80: np.ndarray
    int_upr_prox_a_oob_80: np.ndarray
    int_lwr_prox_a_oob_90: np.ndarray
    int_upr_prox_a_oob_90: np.ndarray
    int_lwr_prox_a_oob_95: np.ndarray
    int_upr_prox_a_oob_95: np.ndarray

    u_prox_a_cv: np.ndarray
    sigma_prox_a_cv: np.ndarray
    int_lwr_prox_a_cv: np.ndarray
    int_upr_prox_a_cv: np.ndarray
    int_lwr_prox_a_cv_50: np.ndarray
    int_upr_prox_a_cv_50: np.ndarray
    int_lwr_prox_a_cv_80: np.ndarray
    int_upr_prox_a_cv_80: np.ndarray
    int_lwr_prox_a_cv_90: np.ndarray
    int_upr_prox_a_cv_90: np.ndarray
    int_lwr_prox_a_cv_95: np.ndarray
    int_upr_prox_a_cv_95: np.ndarray

    # Group 4: Proximity B (OOB & CV)
    u_prox_b_oob: np.ndarray
    sigma_prox_b_oob: np.ndarray
    int_lwr_prox_b_oob: np.ndarray
    int_upr_prox_b_oob: np.ndarray
    int_lwr_prox_b_oob_50: np.ndarray
    int_upr_prox_b_oob_50: np.ndarray
    int_lwr_prox_b_oob_80: np.ndarray
    int_upr_prox_b_oob_80: np.ndarray
    int_lwr_prox_b_oob_90: np.ndarray
    int_upr_prox_b_oob_90: np.ndarray
    int_lwr_prox_b_oob_95: np.ndarray
    int_upr_prox_b_oob_95: np.ndarray

    u_prox_b_cv: np.ndarray
    sigma_prox_b_cv: np.ndarray
    int_lwr_prox_b_cv: np.ndarray
    int_upr_prox_b_cv: np.ndarray
    int_lwr_prox_b_cv_50: np.ndarray
    int_upr_prox_b_cv_50: np.ndarray
    int_lwr_prox_b_cv_80: np.ndarray
    int_upr_prox_b_cv_80: np.ndarray
    int_lwr_prox_b_cv_90: np.ndarray
    int_upr_prox_b_cv_90: np.ndarray
    int_lwr_prox_b_cv_95: np.ndarray
    int_upr_prox_b_cv_95: np.ndarray

    # Group 5: Proximity AC (OOB & CV)
    u_prox_ac_oob: np.ndarray
    sigma_prox_ac_oob: np.ndarray
    int_lwr_prox_ac_oob: np.ndarray
    int_upr_prox_ac_oob: np.ndarray
    int_lwr_prox_ac_oob_50: np.ndarray
    int_upr_prox_ac_oob_50: np.ndarray
    int_lwr_prox_ac_oob_80: np.ndarray
    int_upr_prox_ac_oob_80: np.ndarray
    int_lwr_prox_ac_oob_90: np.ndarray
    int_upr_prox_ac_oob_90: np.ndarray
    int_lwr_prox_ac_oob_95: np.ndarray
    int_upr_prox_ac_oob_95: np.ndarray

    u_prox_ac_cv: np.ndarray
    sigma_prox_ac_cv: np.ndarray
    int_lwr_prox_ac_cv: np.ndarray
    int_upr_prox_ac_cv: np.ndarray
    int_lwr_prox_ac_cv_50: np.ndarray
    int_upr_prox_ac_cv_50: np.ndarray
    int_lwr_prox_ac_cv_80: np.ndarray
    int_upr_prox_ac_cv_80: np.ndarray
    int_lwr_prox_ac_cv_90: np.ndarray
    int_upr_prox_ac_cv_90: np.ndarray
    int_lwr_prox_ac_cv_95: np.ndarray
    int_upr_prox_ac_cv_95: np.ndarray

    # Group 6: Proximity BC (OOB & CV)
    u_prox_bc_oob: np.ndarray
    sigma_prox_bc_oob: np.ndarray
    int_lwr_prox_bc_oob: np.ndarray
    int_upr_prox_bc_oob: np.ndarray
    int_lwr_prox_bc_oob_50: np.ndarray
    int_upr_prox_bc_oob_50: np.ndarray
    int_lwr_prox_bc_oob_80: np.ndarray
    int_upr_prox_bc_oob_80: np.ndarray
    int_lwr_prox_bc_oob_90: np.ndarray
    int_upr_prox_bc_oob_90: np.ndarray
    int_lwr_prox_bc_oob_95: np.ndarray
    int_upr_prox_bc_oob_95: np.ndarray

    u_prox_bc_cv: np.ndarray
    sigma_prox_bc_cv: np.ndarray
    int_lwr_prox_bc_cv: np.ndarray
    int_upr_prox_bc_cv: np.ndarray
    int_lwr_prox_bc_cv_50: np.ndarray
    int_upr_prox_bc_cv_50: np.ndarray
    int_lwr_prox_bc_cv_80: np.ndarray
    int_upr_prox_bc_cv_80: np.ndarray
    int_lwr_prox_bc_cv_90: np.ndarray
    int_upr_prox_bc_cv_90: np.ndarray
    int_lwr_prox_bc_cv_95: np.ndarray
    int_upr_prox_bc_cv_95: np.ndarray

    # Diagnostics and backward compatibility fields
    var_between: Optional[np.ndarray] = None
    var_within: Optional[np.ndarray] = None
    var_total: Optional[np.ndarray] = None
    delta_floor: Optional[np.ndarray] = None
    local_mae: Optional[np.ndarray] = None
    q_lower: Optional[np.ndarray] = None
    u_hutter_between: Optional[np.ndarray] = None
    u_hutter_within: Optional[np.ndarray] = None
    u_shaker_total: Optional[np.ndarray] = None
    shaker_mi: Optional[np.ndarray] = None
    shaker_total_entropy: Optional[np.ndarray] = None
    u_rf_fire_lower: Optional[np.ndarray] = None

    def __init__(
        self,
        y_hat: np.ndarray,
        **kwargs: Any,
    ) -> None:
        self.y_hat = np.asarray(y_hat)
        self._legacy_overrides = dict(kwargs)

        # Parse diagnostics
        diag_keys = (
            "var_between",
            "var_within",
            "var_total",
            "delta_floor",
            "local_mae",
            "q_lower",
            "u_hutter_between",
            "u_hutter_within",
            "u_shaker_total",
            "shaker_mi",
            "shaker_total_entropy",
        )
        for dk in diag_keys:
            setattr(self, dk, kwargs.get(dk, None))

        # Parse all 10 approaches
        for app in APPROACH_NAMES:
            # 1. Uncertainty u_{app}
            u_key = f"u_{app}"
            if u_key in kwargs and kwargs[u_key] is not None:
                u_val = np.asarray(kwargs[u_key])
            else:
                if app == "hutter_total":
                    u_val = np.asarray(kwargs.get("u_slcb", kwargs.get("slcb", np.zeros_like(self.y_hat))))
                elif app == "prox_a_oob":
                    u_val = np.asarray(kwargs.get(
                        "u_prox_a_lower", kwargs.get("u_plcb_lower", kwargs.get("u_plcb", kwargs.get("prox_a", np.zeros_like(self.y_hat))))
                    ))
                elif app == "prox_a_cv":
                    u_val = np.asarray(kwargs.get("u_prox_a_cv", getattr(self, "u_prox_a_oob", np.zeros_like(self.y_hat))))
                elif app == "prox_b_oob":
                    u_val = np.asarray(kwargs.get("u_prox_b_lower", kwargs.get("prox_b", getattr(self, "u_prox_a_oob", np.zeros_like(self.y_hat)))))
                elif app == "prox_b_cv":
                    u_val = np.asarray(kwargs.get("u_prox_b_cv", getattr(self, "u_prox_b_oob", np.zeros_like(self.y_hat))))
                elif app == "prox_ac_oob":
                    u_val = np.asarray(kwargs.get("u_prox_ac_lower", kwargs.get("prox_ac", getattr(self, "u_prox_a_oob", np.zeros_like(self.y_hat)))))
                elif app == "prox_ac_cv":
                    u_val = np.asarray(kwargs.get("u_prox_ac_cv", getattr(self, "u_prox_ac_oob", np.zeros_like(self.y_hat))))
                elif app == "prox_bc_oob":
                    u_val = np.asarray(kwargs.get("u_prox_bc_lower", kwargs.get("prox_bc", getattr(self, "u_prox_b_oob", np.zeros_like(self.y_hat)))))
                elif app == "prox_bc_cv":
                    u_val = np.asarray(kwargs.get("u_prox_bc_cv", getattr(self, "u_prox_bc_oob", np.zeros_like(self.y_hat))))
                else:
                    u_val = np.zeros_like(self.y_hat)
            setattr(self, u_key, u_val)

            # 2. Pure sigma_{app}
            sigma_key = f"sigma_{app}"
            if sigma_key in kwargs and kwargs[sigma_key] is not None:
                sigma_val = np.asarray(kwargs[sigma_key])
            else:
                if app == "hutter_total":
                    if self.var_total is not None:
                        sigma_val = np.sqrt(np.maximum(self.var_total, 0.0))
                    else:
                        sigma_val = u_val / 1.96
                elif app == "shaker_epistemic":
                    sigma_val = u_val / 1.96
                else:
                    sigma_val = u_val / 1.96
            setattr(self, sigma_key, sigma_val)

            # 3. Prediction intervals: suffixed (50, 80, 90, 95) and un-suffixed (default 95)
            lwr_95_key = f"int_lwr_{app}_95"
            upr_95_key = f"int_upr_{app}_95"
            un_lwr_key = f"int_lwr_{app}"
            un_upr_key = f"int_upr_{app}"

            if lwr_95_key in kwargs and kwargs[lwr_95_key] is not None:
                lwr_95 = np.asarray(kwargs[lwr_95_key])
            elif un_lwr_key in kwargs and kwargs[un_lwr_key] is not None:
                lwr_95 = np.asarray(kwargs[un_lwr_key])
            else:
                if app in NON_PROXIMITY_APPROACHES:
                    z95 = float(norm.ppf(0.975))
                    lwr_95 = (self.y_hat - z95 * sigma_val).astype(np.float64)
                else:
                    half = kwargs.get(f"u_{app}_half", u_val)
                    lwr_95 = (self.y_hat - half).astype(np.float64)

            if upr_95_key in kwargs and kwargs[upr_95_key] is not None:
                upr_95 = np.asarray(kwargs[upr_95_key])
            elif un_upr_key in kwargs and kwargs[un_upr_key] is not None:
                upr_95 = np.asarray(kwargs[un_upr_key])
            else:
                if app in NON_PROXIMITY_APPROACHES:
                    z95 = float(norm.ppf(0.975))
                    upr_95 = (self.y_hat + z95 * sigma_val).astype(np.float64)
                else:
                    half = kwargs.get(f"u_{app}_half", u_val)
                    upr_95 = (self.y_hat + half).astype(np.float64)

            setattr(self, lwr_95_key, lwr_95)
            setattr(self, upr_95_key, upr_95)
            setattr(self, un_lwr_key, lwr_95)
            setattr(self, un_upr_key, upr_95)

            # Other levels: 50, 80, 90
            for lvl, sfx in [(0.50, "50"), (0.80, "80"), (0.90, "90")]:
                lwr_sfx_key = f"int_lwr_{app}_{sfx}"
                upr_sfx_key = f"int_upr_{app}_{sfx}"

                if lwr_sfx_key in kwargs and kwargs[lwr_sfx_key] is not None:
                    lwr_sfx = np.asarray(kwargs[lwr_sfx_key])
                else:
                    if app in NON_PROXIMITY_APPROACHES:
                        z = float(norm.ppf(1.0 - (1.0 - lvl) / 2.0))
                        lwr_sfx = (self.y_hat - z * sigma_val).astype(np.float64)
                    else:
                        z_ratio = float(norm.ppf(1.0 - (1.0 - lvl) / 2.0)) / float(norm.ppf(0.975))
                        lwr_sfx = (self.y_hat + z_ratio * (lwr_95 - self.y_hat)).astype(np.float64)

                if upr_sfx_key in kwargs and kwargs[upr_sfx_key] is not None:
                    upr_sfx = np.asarray(kwargs[upr_sfx_key])
                else:
                    if app in NON_PROXIMITY_APPROACHES:
                        z = float(norm.ppf(1.0 - (1.0 - lvl) / 2.0))
                        upr_sfx = (self.y_hat + z * sigma_val).astype(np.float64)
                    else:
                        z_ratio = float(norm.ppf(1.0 - (1.0 - lvl) / 2.0)) / float(norm.ppf(0.975))
                        upr_sfx = (self.y_hat + z_ratio * (upr_95 - self.y_hat)).astype(np.float64)

                setattr(self, lwr_sfx_key, lwr_sfx)
                setattr(self, upr_sfx_key, upr_sfx)

        rf_lower = kwargs.get("u_rf_fire_lower", None)
        self.u_rf_fire_lower = rf_lower if rf_lower is not None else self.u_prox_a_oob

    # Backward compatibility aliases
    @property
    def u_slcb(self) -> np.ndarray:
        return self.u_hutter_total

    @property
    def u_prox_a_lower(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_prox_b_lower(self) -> np.ndarray:
        return self.u_prox_b_oob

    @property
    def u_prox_ac_lower(self) -> np.ndarray:
        return self.u_prox_ac_oob

    @property
    def u_prox_bc_lower(self) -> np.ndarray:
        return self.u_prox_bc_oob

    @property
    def u_plcb(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_plcb_lower(self) -> np.ndarray:
        return self.u_prox_a_oob

    @property
    def u_rf_fire_half(self) -> np.ndarray:
        if hasattr(self, "_legacy_overrides") and "u_rf_fire_half" in self._legacy_overrides:
            return self._legacy_overrides["u_rf_fire_half"]
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    @property
    def u_prox_a_half(self) -> np.ndarray:
        if hasattr(self, "_legacy_overrides") and "u_prox_a_half" in self._legacy_overrides:
            return self._legacy_overrides["u_prox_a_half"]
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    @property
    def u_prox_b_half(self) -> np.ndarray:
        if hasattr(self, "_legacy_overrides") and "u_prox_b_half" in self._legacy_overrides:
            return self._legacy_overrides["u_prox_b_half"]
        return (self.int_upr_prox_b_oob - self.int_lwr_prox_b_oob) / 2.0

    @property
    def u_prox_bc_half(self) -> np.ndarray:
        if hasattr(self, "_legacy_overrides") and "u_prox_bc_half" in self._legacy_overrides:
            return self._legacy_overrides["u_prox_bc_half"]
        return (self.int_upr_prox_bc_oob - self.int_lwr_prox_bc_oob) / 2.0

    @property
    def u_plcb_half(self) -> np.ndarray:
        if hasattr(self, "_legacy_overrides") and "u_plcb_half" in self._legacy_overrides:
            return self._legacy_overrides["u_plcb_half"]
        return (self.int_upr_prox_a_oob - self.int_lwr_prox_a_oob) / 2.0

    @classmethod
    def get_approach_names(cls) -> List[str]:
        return list(APPROACH_NAMES)

    def get_uncertainty(self, approach: str) -> np.ndarray:
        if approach not in APPROACH_NAMES and f"u_{approach}" not in APPROACH_NAMES:
            if approach in ("slcb", "u_slcb"):
                return self.u_hutter_total
            if approach in ("plcb", "u_plcb", "u_prox_a_lower", "prox_a"):
                return self.u_prox_a_oob
            if approach in ("u_prox_b_lower", "prox_b"):
                return self.u_prox_b_oob
            if approach in ("u_prox_ac_lower", "prox_ac"):
                return self.u_prox_ac_oob
            if approach in ("u_prox_bc_lower", "prox_bc"):
                return self.u_prox_bc_oob
        key = approach if approach.startswith("u_") else f"u_{approach}"
        if hasattr(self, key):
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                return val
        raise ValueError(f"Unknown approach '{approach}'. Valid approaches: {APPROACH_NAMES}")

    def get_sigma(self, approach: str) -> np.ndarray:
        clean_app = approach[2:] if approach.startswith("u_") else approach
        if clean_app in ("slcb", "hutter"):
            clean_app = "hutter_total"
        elif clean_app in ("plcb", "prox_a_lower", "prox_a"):
            clean_app = "prox_a_oob"
        elif clean_app in ("prox_b_lower", "prox_b"):
            clean_app = "prox_b_oob"
        elif clean_app in ("prox_ac_lower", "prox_ac"):
            clean_app = "prox_ac_oob"
        elif clean_app in ("prox_bc_lower", "prox_bc"):
            clean_app = "prox_bc_oob"

        field_name = f"sigma_{clean_app}"
        if hasattr(self, field_name):
            val = getattr(self, field_name)
            if isinstance(val, np.ndarray):
                return val
        raise ValueError(f"Unknown approach '{approach}'. Valid approaches: {APPROACH_NAMES}")

    def get_prediction_interval(self, approach: str, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        lvl_key = round(float(level), 4)
        if lvl_key in (0.5, 0.50, 50.0):
            suffix = "50"
        elif lvl_key in (0.8, 0.80, 80.0):
            suffix = "80"
        elif lvl_key in (0.9, 0.90, 90.0):
            suffix = "90"
        elif lvl_key in (0.95, 95.0):
            suffix = "95"
        else:
            raise ValueError(
                f"Unsupported coverage level {level}. Supported levels: [0.50, 0.80, 0.90, 0.95]"
            )

        clean_app = approach[2:] if approach.startswith("u_") else approach
        if clean_app in ("slcb", "hutter"):
            clean_app = "hutter_total"
        elif clean_app in ("plcb", "prox_a_lower", "prox_a"):
            clean_app = "prox_a_oob"
        elif clean_app in ("prox_b_lower", "prox_b"):
            clean_app = "prox_b_oob"
        elif clean_app in ("prox_ac_lower", "prox_ac"):
            clean_app = "prox_ac_oob"
        elif clean_app in ("prox_bc_lower", "prox_bc"):
            clean_app = "prox_bc_oob"

        lwr_key = f"int_lwr_{clean_app}_{suffix}"
        upr_key = f"int_upr_{clean_app}_{suffix}"

        if hasattr(self, lwr_key) and hasattr(self, upr_key):
            return getattr(self, lwr_key), getattr(self, upr_key)

        if suffix == "95":
            base_lwr = f"int_lwr_{clean_app}"
            base_upr = f"int_upr_{clean_app}"
            if hasattr(self, base_lwr) and hasattr(self, base_upr):
                return getattr(self, base_lwr), getattr(self, base_upr)

        raise ValueError(f"Unknown approach '{approach}'. Valid approaches: {APPROACH_NAMES}")

    def __getitem__(self, key: str) -> np.ndarray:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                return val
        raise KeyError(f"Key '{key}' not found in {self.__class__.__name__}.")

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            raise KeyError(f"Cannot set unknown field '{key}' in {self.__class__.__name__}.")

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key) and not key.startswith("_")

    def __len__(self) -> int:
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())

    def keys(self) -> Tuple[str, ...]:
        k_list = ["y_hat"]
        for app in APPROACH_NAMES:
            k_list.append(f"u_{app}")
            k_list.append(f"sigma_{app}")
            k_list.append(f"int_lwr_{app}")
            k_list.append(f"int_upr_{app}")
            for sfx in ("50", "80", "90", "95"):
                k_list.append(f"int_lwr_{app}_{sfx}")
                k_list.append(f"int_upr_{app}_{sfx}")

        legacy_keys = [
            "u_slcb",
            "u_plcb",
            "u_prox_a_lower",
            "u_prox_b_lower",
            "u_prox_ac_lower",
            "u_prox_bc_lower",
            "u_plcb_lower",
            "u_rf_fire_half",
            "u_prox_a_half",
            "u_prox_b_half",
            "u_prox_bc_half",
            "u_plcb_half",
        ]
        k_list.extend(legacy_keys)

        diag_keys = (
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
            "u_hutter_between",
            "u_hutter_within",
            "u_shaker_total",
            "shaker_mi",
            "shaker_total_entropy",
            "u_rf_fire_lower",
        )
        for dk in diag_keys:
            if getattr(self, dk, None) is not None:
                k_list.append(dk)
        return tuple(k_list)

    def values(self) -> Tuple[np.ndarray, ...]:
        return tuple(getattr(self, k) for k in self.keys())

    def items(self) -> Tuple[Tuple[str, np.ndarray], ...]:
        return tuple((k, getattr(self, k)) for k in self.keys())

    def get(self, key: str, default: Any = None) -> Any:
        if key in self:
            return getattr(self, key)
        return default

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.keys()}

    def get_uncertainties_dict(self) -> Dict[str, np.ndarray]:
        d = {
            f"u_{app}": getattr(self, f"u_{app}")
            for app in APPROACH_NAMES
        }
        for app in APPROACH_NAMES:
            d[app] = getattr(self, f"u_{app}")
            d[f"sigma_{app}"] = getattr(self, f"sigma_{app}")
        d["u_slcb"] = self.u_hutter_total
        d["slcb"] = self.u_hutter_total
        d["u_plcb"] = self.u_prox_a_oob
        d["plcb"] = self.u_prox_a_oob
        d["u_prox_a_lower"] = self.u_prox_a_oob
        d["prox_a"] = self.u_prox_a_oob
        d["u_prox_b_lower"] = self.u_prox_b_oob
        d["prox_b"] = self.u_prox_b_oob
        d["u_prox_ac_lower"] = self.u_prox_ac_oob
        d["prox_ac"] = self.u_prox_ac_oob
        d["u_prox_bc_lower"] = self.u_prox_bc_oob
        d["prox_bc"] = self.u_prox_bc_oob
        d["u_plcb_lower"] = self.u_prox_a_oob
        d["u_rf_fire_half"] = self.u_rf_fire_half
        d["u_prox_a_half"] = self.u_prox_a_half
        d["u_prox_b_half"] = self.u_prox_b_half
        d["u_prox_bc_half"] = self.u_prox_bc_half
        d["u_plcb_half"] = self.u_plcb_half
        if self.u_hutter_between is not None:
            d["u_hutter_between"] = self.u_hutter_between
            d["hutter_between"] = self.u_hutter_between
        if self.u_hutter_within is not None:
            d["u_hutter_within"] = self.u_hutter_within
            d["hutter_within"] = self.u_hutter_within
        if self.u_shaker_total is not None:
            d["u_shaker_total"] = self.u_shaker_total
            d["shaker_total"] = self.u_shaker_total
        if self.u_rf_fire_lower is not None:
            d["u_rf_fire_lower"] = self.u_rf_fire_lower
            d["rf_fire"] = self.u_rf_fire_lower
        return d


@dataclass
class DualUQResult(MultiUQResult):
    """Backwards-compatible subclass of MultiUQResult preserving legacy 9-key mapping interface."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def keys(self) -> Tuple[str, ...]:
        return (
            "y_hat",
            "u_slcb",
            "u_plcb",
            "var_between",
            "var_within",
            "var_total",
            "q_lower",
            "delta_floor",
            "local_mae",
        )


class MultiUQEvaluator:
    """Unified Multi-UQ Inference Engine evaluating 10 candidate UQ approaches.

    Evaluates:
    - 2 Non-Proximity: hutter_total, shaker_epistemic
    - 8 Proximity: prox_a_oob, prox_a_cv, prox_b_oob, prox_b_cv,
                   prox_ac_oob, prox_ac_cv, prox_bc_oob, prox_bc_cv

    Parameters
    ----------
    model : Optional[Any], default=None
        Pre-configured or pre-fitted ensemble model.
    seed : Optional[int], default=None
        Random state seed.
    n_trees : int, default=10
        Number of trees in ensemble.
    k : int, default=28
        Number of nearest neighbors under proximity metric.
    epsilon : float, default=0.080791
        Exploration floor scaling coefficient.
    topological_decay_lambda : float, default=0.20486
        Exponential decay lambda for topological tree path distance.
    density_scaling_alpha : float, default=1.0
        Power exponent for topological density scaling.
    level : float, default=0.95
        Confidence level for quantile estimation (0.95 corresponds to kappa=1.96).
    kappa : float, default=1.96
        Standard deviation / error multiplier for exploration bounds.
    device : str, default='cpu'
        Computation backend ('cpu', 'gpu', or 'auto').
    use_extra_trees : Optional[bool], default=None
        Whether to use ExtraTreesRegressor for random splitting.
    batch_size : Union[int, str], default=512
        Chunk size for query batching.
    surrogate_type : str, default='smac_default'
        Surrogate configuration identifier.
    eval_mode : str, default='all'
        Evaluation mode ('all' or 'unweighted_proximity_only').
    weighted_proximity : bool, default=False
        Whether proximity calculation uses leaf sample weighting.
    **kwargs : Any
        Additional parameters passed to model creation.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        seed: Optional[int] = None,
        n_trees: int = 10,
        k: int = 28,
        epsilon: float = 0.080791,
        topological_decay_lambda: float = 0.20486,
        density_scaling_alpha: float = 1.0,
        level: float = 0.95,
        kappa: float = 1.96,
        device: str = "cpu",
        use_extra_trees: Optional[bool] = None,
        batch_size: Union[int, str] = 512,
        surrogate_type: str = "smac_default",
        eval_mode: str = "all",
        weighted_proximity: bool = False,
        **kwargs: Any,
    ) -> None:
        if eval_mode not in ("all", "unweighted_proximity_only"):
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Must be 'all' or 'unweighted_proximity_only'."
            )
        self.eval_mode = eval_mode
        self.surrogate_type = surrogate_type
        self.seed = seed
        self.weighted_proximity = weighted_proximity

        if "n_estimators" in kwargs:
            resolved_n_trees = kwargs["n_estimators"]
        elif n_trees != 10:
            resolved_n_trees = n_trees
        else:
            resolved_n_trees = SURROGATE_CONFIGS.get(surrogate_type, {}).get(
                "n_estimators", n_trees
            )

        self.n_trees = resolved_n_trees
        self.n_estimators = resolved_n_trees
        self.k = k
        self.epsilon = epsilon
        self.topological_decay_lambda = topological_decay_lambda
        self.density_scaling_alpha = density_scaling_alpha
        self.level = level
        self.kappa = kappa
        self.device = device
        self.batch_size = batch_size

        rf_kwargs = dict(kwargs)
        if "n_estimators" not in rf_kwargs:
            rf_kwargs["n_estimators"] = resolved_n_trees
        if use_extra_trees is not None:
            rf_kwargs["use_extra_trees"] = use_extra_trees

        if model is not None:
            self.model = create_surrogate_rf(
                surrogate_type=surrogate_type,
                seed=seed,
                model=model,
                **rf_kwargs,
            )
            if hasattr(self.model, "n_estimators"):
                self.n_trees = self.model.n_estimators
                self.n_estimators = self.model.n_estimators
        else:
            self.model = create_surrogate_rf(
                surrogate_type=surrogate_type,
                seed=seed,
                **rf_kwargs,
            )

        self.use_extra_trees = isinstance(self.model, ExtraTreesRegressor)

        self.uq_model: Optional[GPUProximityRegressionUQ] = None
        self.shaker: Optional[EpistemicQuantifier] = None
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.oob_res_: Optional[np.ndarray] = None
        self.cv_res_: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    @classmethod
    def get_approach_names(cls) -> List[str]:
        return list(APPROACH_NAMES)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        refit: bool = False,
    ) -> "MultiUQEvaluator":
        """Fit the underlying forest, initialize engines, extract OOB and 5-fold CV residuals."""
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64).flatten()

        if X_arr.ndim != 2:
            raise ValueError(
                f"X_train must be a 2D array of shape (N, D), got {X_arr.ndim}D array."
            )
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(
                f"X_train cannot have empty dimensions, got shape {X_arr.shape}."
            )
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"Sample size mismatch: X_train has {X_arr.shape[0]} samples, "
                f"but y_train has {y_arr.shape[0]} samples."
            )

        N_train = X_arr.shape[0]

        # Fit model if not yet fitted or if refit is requested
        if not hasattr(self.model, "estimators_") or refit:
            self.model.fit(X_arr, y_arr)

        # 1. Proximity UQ engine wrapping identical forest
        self.uq_model = GPUProximityRegressionUQ(
            model=self.model,
            X_train=X_arr,
            y_train=y_arr,
            device=self.device,
            topological_decay_lambda=self.topological_decay_lambda,
            density_scaling_alpha=self.density_scaling_alpha,
            use_density_scaling=True,
            weighting="leaf_normalized" if self.weighted_proximity else "unweighted_all",
            weighted=self.weighted_proximity,
        )
        self.uq_model.fit()

        # 2. EpistemicQuantifier for Shaker GMM entropy family
        if self.eval_mode == "unweighted_proximity_only":
            self.shaker = None
        else:
            self.shaker = EpistemicQuantifier(
                model=self.model,
                X_train=X_arr,
                y_train=y_arr,
            )

        # 3. Extract Out-of-Bag (OOB) residuals
        if hasattr(self.model, "oob_prediction_") and self.model.oob_prediction_ is not None:
            oob_pred = np.asarray(self.model.oob_prediction_, dtype=np.float64).ravel()
            valid_mask = np.isfinite(oob_pred)
            clean_oob_pred = np.where(valid_mask, oob_pred, y_arr)
            self.oob_res_ = (y_arr - clean_oob_pred).astype(np.float64)
        elif hasattr(self.uq_model, "oob_residuals") and self.uq_model.oob_residuals is not None:
            self.oob_res_ = np.asarray(self.uq_model.oob_residuals, dtype=np.float64)
        else:
            self.oob_res_ = np.zeros(N_train, dtype=np.float64)

        # 4. Compute 5-fold CV out-of-fold residuals
        if N_train < 5:
            warnings.warn(
                f"Training set size N={N_train} is smaller than 5 CV folds. "
                "Falling back to OOB residuals for CV.",
                UserWarning,
                stacklevel=2,
            )
            self.cv_res_ = self.oob_res_.copy()
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            cv_res = np.zeros(N_train, dtype=np.float64)
            for train_idx, val_idx in kf.split(X_arr):
                sub_model = clone(self.model)
                if hasattr(sub_model, "oob_score"):
                    sub_model.set_params(oob_score=False)
                sub_model.fit(X_arr[train_idx], y_arr[train_idx])
                y_pred_val = sub_model.predict(X_arr[val_idx])
                if isinstance(y_pred_val, tuple):
                    y_pred_val = y_pred_val[0]
                cv_res[val_idx] = y_arr[val_idx] - np.asarray(y_pred_val, dtype=np.float64).ravel()
            self.cv_res_ = cv_res

        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self._is_fitted = True
        return self

    def evaluate(self, X_test: np.ndarray) -> MultiUQResult:
        """Evaluate all 10 uncertainty quantification methods and 95% prediction intervals."""
        if not self._is_fitted or self.X_train_ is None:
            raise RuntimeError("MultiUQEvaluator must be fitted before calling evaluate().")

        X_test_arr = np.asarray(X_test, dtype=np.float64)
        if X_test_arr.ndim == 1:
            X_test_2d = X_test_arr[np.newaxis, :]
        elif X_test_arr.ndim == 2:
            X_test_2d = X_test_arr
        else:
            raise ValueError(
                f"X_test must be a 1D or 2D array, got {X_test_arr.ndim}D array."
            )

        if X_test_2d.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: query X_test has {X_test_2d.shape[1]} features, "
                f"but training data had {self.X_train_.shape[1]} features."
            )

        M = X_test_2d.shape[0]
        N_train = len(self.X_train_)
        estimators = self.model.estimators_
        B = len(estimators)

        # =========================================================================
        # Group 1: Non-Proximity 1: Hutter Total (Law of Total Variance)
        # =========================================================================
        tree_preds = np.column_stack([t.predict(X_test_2d) for t in estimators])
        y_hat = np.mean(tree_preds, axis=1).astype(np.float64)

        var_between = (
            np.var(tree_preds, axis=1, ddof=1).astype(np.float64)
            if B > 1
            else np.zeros_like(y_hat)
        )

        tree_impurities = np.column_stack([
            np.maximum(0.0, t.tree_.impurity[t.apply(X_test_2d)]) for t in estimators
        ])
        var_within = np.mean(tree_impurities, axis=1).astype(np.float64)

        if np.any(var_between < 0.0) or np.any(var_within < 0.0):
            warnings.warn("Negative variance detected in decomposition; clamping to 0.0.", RuntimeWarning)
        var_between = np.maximum(0.0, var_between)
        var_within = np.maximum(0.0, var_within)
        var_total = np.maximum(0.0, var_between + var_within)

        sigma_hutter_total = np.sqrt(var_total).astype(np.float64)
        u_hutter_total = (self.kappa * sigma_hutter_total).astype(np.float64)

        z_50 = float(norm.ppf(0.75))
        z_80 = float(norm.ppf(0.90))
        z_90 = float(norm.ppf(0.95))
        z_95 = float(norm.ppf(0.975))

        int_lwr_hutter_total_50 = (y_hat - z_50 * sigma_hutter_total).astype(np.float64)
        int_upr_hutter_total_50 = (y_hat + z_50 * sigma_hutter_total).astype(np.float64)
        int_lwr_hutter_total_80 = (y_hat - z_80 * sigma_hutter_total).astype(np.float64)
        int_upr_hutter_total_80 = (y_hat + z_80 * sigma_hutter_total).astype(np.float64)
        int_lwr_hutter_total_90 = (y_hat - z_90 * sigma_hutter_total).astype(np.float64)
        int_upr_hutter_total_90 = (y_hat + z_90 * sigma_hutter_total).astype(np.float64)
        int_lwr_hutter_total_95 = (y_hat - z_95 * sigma_hutter_total).astype(np.float64)
        int_upr_hutter_total_95 = (y_hat + z_95 * sigma_hutter_total).astype(np.float64)
        int_lwr_hutter_total = int_lwr_hutter_total_95
        int_upr_hutter_total = int_upr_hutter_total_95

        u_hutter_between = (self.kappa * np.sqrt(var_between)).astype(np.float64)
        u_hutter_within = (self.kappa * np.sqrt(var_within)).astype(np.float64)

        # =========================================================================
        # Group 2: Non-Proximity 2: Shaker Epistemic (Huber Mutual Information)
        # =========================================================================
        if self.eval_mode == "unweighted_proximity_only":
            sigma_shaker_epistemic = np.zeros(M, dtype=np.float64)
            u_shaker_epistemic = np.zeros(M, dtype=np.float64)
            int_lwr_shaker_epistemic = y_hat.copy()
            int_upr_shaker_epistemic = y_hat.copy()
            int_lwr_shaker_epistemic_50 = y_hat.copy()
            int_upr_shaker_epistemic_50 = y_hat.copy()
            int_lwr_shaker_epistemic_80 = y_hat.copy()
            int_upr_shaker_epistemic_80 = y_hat.copy()
            int_lwr_shaker_epistemic_90 = y_hat.copy()
            int_upr_shaker_epistemic_90 = y_hat.copy()
            int_lwr_shaker_epistemic_95 = y_hat.copy()
            int_upr_shaker_epistemic_95 = y_hat.copy()
            shaker_mi = np.zeros(M, dtype=np.float64)
            shaker_total_entropy = np.zeros(M, dtype=np.float64)
            u_shaker_total = np.zeros(M, dtype=np.float64)
        else:
            if self.shaker is None:
                self.shaker = EpistemicQuantifier(
                    model=self.model,
                    X_train=self.X_train_,
                    y_train=self.y_train_,
                )

            shaker_mi = np.asarray(
                self.shaker.shaker_get_epistemic_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            shaker_total_entropy = np.asarray(
                self.shaker._shaker_calc_total_entropy(
                    X_test_2d,
                    method="huber",
                    backend=self.device,
                ),
                dtype=np.float64,
            )

            aleatoric_var = np.asarray(
                self.shaker.base_get_aleatoric_variance(X_test_2d),
                dtype=np.float64,
            )
            sigma_aleatoric = np.sqrt(np.maximum(aleatoric_var, 1e-6))

            safe_mi_exp = np.clip(2.0 * shaker_mi, 0.0, 50.0)
            sigma_epistemic = sigma_aleatoric * np.sqrt(np.maximum(2.0 ** safe_mi_exp - 1.0, 0.0))
            sigma_shaker_epistemic = sigma_epistemic.astype(np.float64)
            u_shaker_epistemic = (self.kappa * sigma_shaker_epistemic).astype(np.float64)

            int_lwr_shaker_epistemic_50 = (y_hat - z_50 * sigma_shaker_epistemic).astype(np.float64)
            int_upr_shaker_epistemic_50 = (y_hat + z_50 * sigma_shaker_epistemic).astype(np.float64)
            int_lwr_shaker_epistemic_80 = (y_hat - z_80 * sigma_shaker_epistemic).astype(np.float64)
            int_upr_shaker_epistemic_80 = (y_hat + z_80 * sigma_shaker_epistemic).astype(np.float64)
            int_lwr_shaker_epistemic_90 = (y_hat - z_90 * sigma_shaker_epistemic).astype(np.float64)
            int_upr_shaker_epistemic_90 = (y_hat + z_90 * sigma_shaker_epistemic).astype(np.float64)
            int_lwr_shaker_epistemic_95 = (y_hat - z_95 * sigma_shaker_epistemic).astype(np.float64)
            int_upr_shaker_epistemic_95 = (y_hat + z_95 * sigma_shaker_epistemic).astype(np.float64)
            int_lwr_shaker_epistemic = int_lwr_shaker_epistemic_95
            int_upr_shaker_epistemic = int_upr_shaker_epistemic_95

            safe_tot_exp = np.clip(2.0 * shaker_total_entropy, -50.0, 50.0)
            entropy_power_var = (2.0 ** safe_tot_exp) / (2.0 * np.pi * np.e)
            u_shaker_total = (
                self.kappa * np.sqrt(np.maximum(entropy_power_var, 0.0))
            ).astype(np.float64)

        # =========================================================================
        # Group 3: Proximity Approaches (8 variants: A, B, AC, BC across OOB and CV)
        # =========================================================================
        k_eff = min(self.k, N_train)

        prox_names = [
            "prox_a_oob", "prox_a_cv",
            "prox_b_oob", "prox_b_cv",
            "prox_ac_oob", "prox_ac_cv",
            "prox_bc_oob", "prox_bc_cv",
        ]
        u_prox: Dict[str, np.ndarray] = {app: np.zeros(M, dtype=np.float64) for app in prox_names}
        int_lwr_prox: Dict[str, Dict[str, np.ndarray]] = {
            app: {sfx: np.zeros(M, dtype=np.float64) for sfx in ("50", "80", "90", "95")}
            for app in prox_names
        }
        int_upr_prox: Dict[str, Dict[str, np.ndarray]] = {
            app: {sfx: np.zeros(M, dtype=np.float64) for sfx in ("50", "80", "90", "95")}
            for app in prox_names
        }

        q_lower = np.zeros(M, dtype=np.float64)
        delta_floor = np.zeros(M, dtype=np.float64)
        local_mae = np.zeros(M, dtype=np.float64)
        u_rf_fire_lower = np.zeros(M, dtype=np.float64)

        has_precomputed_topo = (
            self.topological_decay_lambda is not None
            and self.uq_model is not None
            and hasattr(self.uq_model, "tree_leaf_distances")
            and self.uq_model.tree_leaf_distances is not None
            and len(self.uq_model.tree_leaf_distances) > 0
        )

        if has_precomputed_topo:
            oob_res = np.asarray(self.oob_res_, dtype=np.float64)
            cv_res = np.asarray(self.cv_res_, dtype=np.float64)
            valid_oob_mask = (
                np.asarray(self.uq_model.valid_oob_mask)
                if hasattr(self.uq_model, "valid_oob_mask") and self.uq_model.valid_oob_mask is not None
                else None
            )
            n_baseline = float(getattr(self.uq_model, "N_baseline", 1.0))
            chunk_size = 512 if isinstance(self.batch_size, str) else int(self.batch_size)

            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                b_len = end - start
                X_chunk = X_test_2d[start:end, :]
                y_hat_chunk = y_hat[start:end]
                leaf_batch = self.model.apply(X_chunk)

                prox_topo_unweighted = np.zeros((b_len, N_train), dtype=np.float32)
                density_chunk = np.zeros(b_len, dtype=np.float32)
                prox_fire = np.zeros((b_len, N_train), dtype=np.float32)

                for t in range(B):
                    id_to_dense = self.uq_model.tree_leaf_id_to_dense[t]
                    if hasattr(id_to_dense, "get"):
                        id_to_dense = id_to_dense.get()
                    dense_test = id_to_dense[leaf_batch[:, t]]
                    dense_train_inbag = id_to_dense[self.uq_model.in_bag_leaves[:, t]]
                    dense_train_all = id_to_dense[self.uq_model.leaf_matrix_train[:, t]]

                    tree_dists = self.uq_model.tree_leaf_distances[t]
                    if hasattr(tree_dists, "get"):
                        tree_dists = tree_dists.get()

                    d_t_unweighted = tree_dists[dense_test[:, None], dense_train_all[None, :]]
                    decay_t_unweighted = np.exp(-self.topological_decay_lambda * d_t_unweighted)
                    prox_topo_unweighted += decay_t_unweighted

                    d_t_inbag = tree_dists[dense_test[:, None], dense_train_inbag[None, :]]
                    decay_t_inbag = np.exp(-self.topological_decay_lambda * d_t_inbag)
                    in_bag_c = self.uq_model.in_bag_counts[:, t]
                    density_chunk += np.sum(decay_t_inbag * in_bag_c[None, :], axis=1)

                    train_w = self.uq_model.train_weights[:, t]
                    matches_t = (leaf_batch[:, t, None] == self.uq_model.in_bag_leaves[None, :, t])
                    prox_fire += matches_t * train_w[None, :]

                prox_topo_unweighted /= B
                prox_fire /= B

                avg_test_leaf_sizes = density_chunk / B
                avg_test_leaf_sizes = np.maximum(avg_test_leaf_sizes, 1e-5)
                gamma = np.maximum(1.0, (n_baseline / avg_test_leaf_sizes) ** self.density_scaling_alpha)

                alpha_pairs = [
                    ("50", 0.25, 0.75),
                    ("80", 0.10, 0.90),
                    ("90", 0.05, 0.95),
                    ("95", 0.025, 0.975),
                ]

                for pool in ("oob", "cv"):
                    pool_residuals = oob_res if pool == "oob" else cv_res

                    if pool == "oob" and valid_oob_mask is not None:
                        prox_pool = prox_topo_unweighted.copy()
                        prox_pool[:, ~valid_oob_mask] = 0.0
                    else:
                        prox_pool = prox_topo_unweighted

                    # Method A: Top k_eff nearest neighbors
                    if k_eff < N_train:
                        partition_idx = np.flip(np.argsort(prox_pool, axis=1), axis=1)[:, :k_eff]
                        k_residuals = pool_residuals[partition_idx]
                    else:
                        k_residuals = np.broadcast_to(pool_residuals[None, :], (b_len, N_train))

                    app_a = f"prox_a_{pool}"
                    app_ac = f"prox_ac_{pool}"
                    app_b = f"prox_b_{pool}"
                    app_bc = f"prox_bc_{pool}"

                    for sfx, a_lwr, a_upr in alpha_pairs:
                        # Method A quantiles
                        q_lwr_a = np.quantile(k_residuals, a_lwr, axis=1)
                        q_upr_a = np.quantile(k_residuals, a_upr, axis=1)

                        # UNCAPPED prediction intervals
                        int_lwr_prox[app_a][sfx][start:end] = y_hat_chunk + q_lwr_a
                        int_upr_prox[app_a][sfx][start:end] = y_hat_chunk + q_upr_a

                        int_lwr_prox[app_ac][sfx][start:end] = y_hat_chunk + gamma * q_lwr_a
                        int_upr_prox[app_ac][sfx][start:end] = y_hat_chunk + gamma * q_upr_a

                        if sfx == "95":
                            # Scalar exploration term preserves zero-cap: max(0, -q)
                            u_prox[app_a][start:end] = np.maximum(0.0, -q_lwr_a)
                            u_prox[app_ac][start:end] = gamma * np.maximum(0.0, -q_lwr_a)

                        # Method B quantiles (continuous weighted)
                        q_lwr_b = self.uq_model._compute_weighted_quantile(pool_residuals, prox_pool, a_lwr)
                        q_upr_b = self.uq_model._compute_weighted_quantile(pool_residuals, prox_pool, a_upr)
                        if hasattr(q_lwr_b, "get"):
                            q_lwr_b = q_lwr_b.get()
                        if hasattr(q_upr_b, "get"):
                            q_upr_b = q_upr_b.get()
                        q_lwr_b = np.asarray(q_lwr_b, dtype=np.float64)
                        q_upr_b = np.asarray(q_upr_b, dtype=np.float64)

                        # UNCAPPED prediction intervals
                        int_lwr_prox[app_b][sfx][start:end] = y_hat_chunk + q_lwr_b
                        int_upr_prox[app_b][sfx][start:end] = y_hat_chunk + q_upr_b

                        int_lwr_prox[app_bc][sfx][start:end] = y_hat_chunk + gamma * q_lwr_b
                        int_upr_prox[app_bc][sfx][start:end] = y_hat_chunk + gamma * q_upr_b

                        if sfx == "95":
                            u_prox[app_b][start:end] = np.maximum(0.0, -q_lwr_b)
                            u_prox[app_bc][start:end] = gamma * np.maximum(0.0, -q_lwr_b)

                    if pool == "oob":
                        # Diagnostics from OOB Method A at 95%
                        q_lwr_a_95 = np.quantile(k_residuals, 0.025, axis=1)
                        q_upr_a_95 = np.quantile(k_residuals, 0.975, axis=1)
                        in_int_topo_u = (k_residuals >= q_lwr_a_95[:, None]) & (k_residuals <= q_upr_a_95[:, None])
                        cnt_topo_u = np.sum(in_int_topo_u, axis=1)
                        sum_abs_topo_u = np.sum(np.abs(k_residuals) * in_int_topo_u, axis=1)
                        b_local_mae_u = np.where(
                            cnt_topo_u > 0,
                            sum_abs_topo_u / np.maximum(cnt_topo_u, 1),
                            float(self.uq_model.oob_mae) if hasattr(self.uq_model, "oob_mae") else 1.0,
                        )
                        b_delta_floor_u = self.epsilon * self.kappa * b_local_mae_u

                        q_lower[start:end] = q_lwr_a_95
                        delta_floor[start:end] = b_delta_floor_u
                        local_mae[start:end] = b_local_mae_u

                # RF-FIRE lower quantile
                if valid_oob_mask is not None:
                    prox_fire_pool = prox_fire.copy()
                    prox_fire_pool[:, ~valid_oob_mask] = 0.0
                else:
                    prox_fire_pool = prox_fire

                if k_eff < N_train:
                    partition_idx_fire = np.flip(np.argsort(prox_fire_pool, axis=1), axis=1)[:, :k_eff]
                    k_residuals_fire = oob_res[partition_idx_fire]
                else:
                    k_residuals_fire = np.broadcast_to(oob_res[None, :], (b_len, N_train))
                q_lwr_fire = np.quantile(k_residuals_fire, 0.025, axis=1)
                u_rf_fire_lower[start:end] = np.maximum(0.0, -q_lwr_fire)
        else:
            for lvl, sfx in [(0.50, "50"), (0.80, "80"), (0.90, "90"), (0.95, "95")]:
                y_pred_lwr, y_pred_mid, y_pred_upr, b_local_mae = self.uq_model.predict_with_intervals(
                    X_test_2d,
                    n_neighbors=k_eff,
                    level=lvl,
                    return_mae=True,
                )
                lwr_fb = np.asarray(y_pred_lwr, dtype=np.float64)
                upr_fb = np.asarray(y_pred_upr, dtype=np.float64)

                for app in prox_names:
                    int_lwr_prox[app][sfx][:] = lwr_fb
                    int_upr_prox[app][sfx][:] = upr_fb

                if sfx == "95":
                    u_fb = np.maximum(0.0, y_hat - lwr_fb)
                    for app in prox_names:
                        u_prox[app][:] = u_fb

                    ref_mid = y_pred_mid if y_pred_mid is not None else y_hat
                    q_lower[:] = np.asarray(lwr_fb - ref_mid, dtype=np.float64)
                    local_mae[:] = np.asarray(b_local_mae, dtype=np.float64)
                    delta_floor[:] = self.epsilon * self.kappa * local_mae
                    u_rf_fire_lower[:] = u_fb

        sigma_prox: Dict[str, np.ndarray] = {
            app: (u_prox[app] / 1.96).astype(np.float64) for app in prox_names
        }

        result_kwargs: Dict[str, Any] = {
            "y_hat": y_hat,
            "u_hutter_total": u_hutter_total,
            "sigma_hutter_total": sigma_hutter_total,
            "int_lwr_hutter_total": int_lwr_hutter_total,
            "int_upr_hutter_total": int_upr_hutter_total,
            "int_lwr_hutter_total_50": int_lwr_hutter_total_50,
            "int_upr_hutter_total_50": int_upr_hutter_total_50,
            "int_lwr_hutter_total_80": int_lwr_hutter_total_80,
            "int_upr_hutter_total_80": int_upr_hutter_total_80,
            "int_lwr_hutter_total_90": int_lwr_hutter_total_90,
            "int_upr_hutter_total_90": int_upr_hutter_total_90,
            "int_lwr_hutter_total_95": int_lwr_hutter_total_95,
            "int_upr_hutter_total_95": int_upr_hutter_total_95,

            "u_shaker_epistemic": u_shaker_epistemic,
            "sigma_shaker_epistemic": sigma_shaker_epistemic,
            "int_lwr_shaker_epistemic": int_lwr_shaker_epistemic,
            "int_upr_shaker_epistemic": int_upr_shaker_epistemic,
            "int_lwr_shaker_epistemic_50": int_lwr_shaker_epistemic_50,
            "int_upr_shaker_epistemic_50": int_upr_shaker_epistemic_50,
            "int_lwr_shaker_epistemic_80": int_lwr_shaker_epistemic_80,
            "int_upr_shaker_epistemic_80": int_upr_shaker_epistemic_80,
            "int_lwr_shaker_epistemic_90": int_lwr_shaker_epistemic_90,
            "int_upr_shaker_epistemic_90": int_upr_shaker_epistemic_90,
            "int_lwr_shaker_epistemic_95": int_lwr_shaker_epistemic_95,
            "int_upr_shaker_epistemic_95": int_upr_shaker_epistemic_95,

            "var_between": var_between,
            "var_within": var_within,
            "var_total": var_total,
            "delta_floor": delta_floor,
            "local_mae": local_mae,
            "q_lower": q_lower,
            "u_hutter_between": u_hutter_between,
            "u_hutter_within": u_hutter_within,
            "u_shaker_total": u_shaker_total,
            "shaker_mi": shaker_mi,
            "shaker_total_entropy": shaker_total_entropy,
            "u_rf_fire_lower": u_rf_fire_lower,
        }

        for app in prox_names:
            result_kwargs[f"u_{app}"] = u_prox[app]
            result_kwargs[f"sigma_{app}"] = sigma_prox[app]
            result_kwargs[f"int_lwr_{app}"] = int_lwr_prox[app]["95"]
            result_kwargs[f"int_upr_{app}"] = int_upr_prox[app]["95"]
            for sfx in ("50", "80", "90", "95"):
                result_kwargs[f"int_lwr_{app}_{sfx}"] = int_lwr_prox[app][sfx]
                result_kwargs[f"int_upr_{app}_{sfx}"] = int_upr_prox[app][sfx]

        return MultiUQResult(**result_kwargs)


class DualUQEvaluator(MultiUQEvaluator):
    """Dual Uncertainty Quantification Inference Engine."""

    def __init__(
        self,
        *args: Any,
        eval_mode: str = "all",
        weighted_proximity: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, eval_mode=eval_mode, weighted_proximity=weighted_proximity, **kwargs)

    def evaluate(self, X_test: np.ndarray) -> DualUQResult:
        """Evaluate both Standard SMAC3 LCB and Proximity LCB."""
        res = super().evaluate(X_test)
        field_dict = {f.name: getattr(res, f.name) for f in fields(MultiUQResult)}
        return DualUQResult(**field_dict)

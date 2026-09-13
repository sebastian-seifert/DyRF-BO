"""
Distance-Aware Evidential Hybrid Random Forest (DA-EHRF) Epistemic Uncertainty Extractor.
Decomposes total surrogate uncertainty and isolates pure epistemic ignorance:
    U_E(x) = sqrt( V_ens(x) + V_leaf(x) + V_spatial(x) )
in linear standard deviation units [y].
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist

from ep_extractors.base import BaseEpistemicExtractor
from ep_extractors import UQExtractorRegistry


@UQExtractorRegistry.register("distance_evidential")
class DistanceAwareEvidentialExtractor(BaseEpistemicExtractor):
    """
    Computes pure epistemic uncertainty U_E(x) in linear standard deviation units:
        U_E(x) = sqrt( V_ens(x) + V_leaf(x) + V_spatial(x) )

    Parameters
    ----------
    model : RandomForestRegressor or EPMRandomForest
        Fitted random forest surrogate.
    lengthscale : float | str | None, default="adaptive"
        RBF bandwidth ell_0 for spatial kernel distance. If "adaptive",
        precomputed in fit() from normalized median nearest-neighbor distance.
    kappa_leaf : float, default=1.0
        Scaling factor for finite-sample leaf ignorance V_leaf.
    c_spatial : float, default=1.0
        Scaling factor for spatial extrapolation ignorance V_spatial.
    use_feature_importances : bool, default=True
        Whether to scale distance dimensions by MDI feature importances.
    length_scale : Optional[float | str], default=None
        Alias for lengthscale.
    """
    def __init__(
        self,
        model,
        lengthscale: float | str | None = "adaptive",
        kappa_leaf: float = 1.0,
        c_spatial: float = 1.0,
        use_feature_importances: bool = True,
        length_scale: Optional[float | str] = None,
        sigma_0_fallback: float = 1.0,
        **kwargs
    ):
        super().__init__(model)
        if length_scale is not None:
            lengthscale = length_scale
        self.lengthscale_param = lengthscale
        if isinstance(lengthscale, (int, float)):
            self.lengthscale = max(float(lengthscale), 1e-6)
        else:
            self.lengthscale = 1.0
        self.kappa_leaf = float(kappa_leaf)
        self.c_spatial = float(c_spatial)
        self.use_feature_importances = bool(use_feature_importances)
        self.sigma_0_fallback = float(sigma_0_fallback)
        
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.sigma_0: float = 1.0
        self.weights: Optional[np.ndarray] = None
        self.X_train_w: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Precomputes training dataset statistics, feature weights, target prior scale,
        and pre-scaled training coordinates.
        """
        self.X_train = np.atleast_2d(np.asarray(X_train, dtype=np.float64))
        self.y_train = np.asarray(y_train, dtype=np.float64).flatten()
        
        # Compute global target standard deviation prior sigma_0 with scale equivariance
        if len(self.y_train) > 1:
            raw_std = float(np.std(self.y_train))
            if raw_std > 1e-12:
                self.sigma_0 = raw_std
            else:
                mean_mag = abs(float(np.mean(self.y_train)))
                self.sigma_0 = mean_mag if mean_mag > 1e-12 else float(self.sigma_0_fallback)
        else:
            self.sigma_0 = float(self.sigma_0_fallback)
            
        n_features = self.X_train.shape[1]
        
        # Coordinate ranges with zero-variance protection
        ranges = np.ptp(self.X_train, axis=0)
        ranges = np.where(ranges > 1e-12, ranges, 1.0)
        
        # Feature weighting from MDI
        if (
            self.use_feature_importances
            and hasattr(self.model, "feature_importances_")
            and self.model.feature_importances_ is not None
        ):
            mdi = np.asarray(self.model.feature_importances_, dtype=np.float64)
            if np.all(mdi <= 0.0) or np.any(np.isnan(mdi)):
                mdi = np.ones(n_features, dtype=np.float64) / n_features
            w = (np.sqrt(n_features) / ranges) * np.sqrt(np.maximum(mdi, 1e-4))
        else:
            w = 1.0 / ranges
            
        self.weights = w
        self.X_train_w = self.X_train * self.weights

        # Precompute adaptive spatial bandwidth ell_0 (PROJECT.md line 50)
        if self.lengthscale_param in ("adaptive", "auto", None) or (isinstance(self.lengthscale_param, str) and self.lengthscale_param.lower() in ("adaptive", "auto")):
            if len(self.X_train) >= 2:
                X_norm = self.X_train / ranges
                dists = cdist(X_norm, X_norm, metric="euclidean")
                np.fill_diagonal(dists, np.inf)
                nn_dists = dists.min(axis=1)
                nn_valid = nn_dists[np.isfinite(nn_dists) & (nn_dists > 1e-12)]
                if len(nn_valid) > 0:
                    median_nn = float(np.median(nn_valid))
                    self.lengthscale = float(np.clip(median_nn * 2.0, 0.05, 0.5))
                else:
                    self.lengthscale = 0.15
            else:
                self.lengthscale = 0.15
        elif isinstance(self.lengthscale_param, (int, float)):
            self.lengthscale = max(float(self.lengthscale_param), 1e-6)

        self._is_fitted = True

    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the epistemic standard deviation sigma_E for candidate configurations.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features) or (n_features,).

        Returns
        -------
        sigma_E : np.ndarray
            Standard deviation array of shape (n_samples,).
        """
        if not self._is_fitted or self.X_train is None:
            raise RuntimeError("Extractor must be fitted before extracting signal.")
            
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        elif X_arr.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D array, got ndim={X_arr.ndim}")
            
        n_samples = X_arr.shape[0]
        if n_samples == 0:
            return np.empty(0, dtype=np.float64)

        # Safe coordinate clipping to prevent float32 overflow in scikit-learn Cython tree apply
        X_arr = np.clip(X_arr, -1e37, 1e37)

        # 1. Ensemble Disagreement V_ens(x) & Leaf Ignorance V_leaf(x)
        estimators = getattr(self.model, "estimators_", None)
        if estimators is not None and len(estimators) > 0:
            all_leaf_ids = self.model.apply(X_arr)  # shape: (n_samples, n_estimators)
            n_estimators = len(estimators)
            
            tree_means = np.zeros((n_estimators, n_samples), dtype=np.float64)
            tree_inv_counts = np.zeros((n_estimators, n_samples), dtype=np.float64)
            
            for m, estimator in enumerate(estimators):
                leaf_ids_m = all_leaf_ids[:, m]
                # Tree leaf predictive values: shape (n_nodes, 1, 1) in sklearn
                tree_means[m, :] = estimator.tree_.value[leaf_ids_m, 0, 0]
                # Number of training samples in the leaf
                counts_m = np.maximum(estimator.tree_.n_node_samples[leaf_ids_m], 1.0)
                tree_inv_counts[m, :] = 1.0 / counts_m
                
            v_ens = np.var(tree_means, axis=0, ddof=0)
            v_leaf = (self.sigma_0 ** 2) * self.kappa_leaf * np.mean(tree_inv_counts, axis=0)
        else:
            v_ens = np.zeros(n_samples, dtype=np.float64)
            v_leaf = np.zeros(n_samples, dtype=np.float64)

        # 2. Spatial Extrapolation Ignorance V_spatial(x)
        X_w = X_arr * self.weights
        min_sq_dists = cdist(X_w, self.X_train_w, metric="sqeuclidean").min(axis=1)
        min_sq_dists = np.maximum(min_sq_dists, 0.0)
        
        exponent = -min_sq_dists / (2.0 * (self.lengthscale ** 2))
        # Protect against extreme values underflow
        exponent = np.clip(exponent, -700.0, 0.0)
        v_spatial = (self.sigma_0 ** 2) * self.c_spatial * (1.0 - np.exp(exponent))

        # 3. Combine in standard deviation units
        total_var = np.maximum(v_ens + v_leaf + v_spatial, 0.0)
        u_e = np.sqrt(total_var)
        return np.nan_to_num(u_e, nan=0.0, posinf=self.sigma_0, neginf=0.0)

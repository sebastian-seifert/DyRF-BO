import os
import sys
import numpy as np
import warnings

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from Credal_Regression_UQ import CredalRegressionUQ
from Epistemic_Quantifier import EpistemicQuantifier

class HybridProximityEpistemicUQ:
    def __init__(
        self, 
        model, 
        X_train, 
        y_train, 
        base_epistemic_method="likelihood",
        proximity_decay_lambda=1.0,
        normalize_by_depth=False,
        lambda_blend=0.4,
        k_neighbors=20,
        device="auto",
        batch_size="auto"
    ):
        """
        Hybrid UQ approach blending local neighborhood epistemic uncertainty (RF Proximity KNN)
        with the query point's own epistemic uncertainty.
        
        Args:
            model: Fitted RandomForestRegressor.
            X_train: array-like training inputs.
            y_train: array-like training targets.
            base_epistemic_method: str, "likelihood" (Credal relative likelihood) or "shaker_entropy" (GMM Entropy).
            proximity_decay_lambda: float, lambda decay factor for proximity weights.
            normalize_by_depth: bool, normalize topological distance by tree depth.
            lambda_blend: float, lambda_1 parameter blending neighborhood uncertainty (weight of neighbor avg).
            k_neighbors: int, number of nearest neighbors (KNN) to pool.
            device: str, backend device ("auto", "cpu", "gpu").
            batch_size: int or "auto", batch sizing for proximity compute.
        """
        self.model = model
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.base_epistemic_method = base_epistemic_method.lower()
        self.proximity_decay_lambda = proximity_decay_lambda
        self.normalize_by_depth = normalize_by_depth
        self.lambda_blend = lambda_blend
        self.k_neighbors = k_neighbors
        self.device = device
        self.batch_size = batch_size
        
        if hasattr(self.model, "estimators_"):
            self.fit()

    def fit(self):
        """Precomputes training set epistemic uncertainties and fits proximity model."""
        # Compute training set uncertainties once
        self.train_epistemic_values = self._get_base_epistemic(self.X_train)
        
        # Initialize and fit proximity model
        self.prox_model = GPUProximityRegressionUQ(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            device=self.device,
            batch_size=self.batch_size,
            use_density_scaling=False,
            topological_decay_lambda=self.proximity_decay_lambda,
            normalize_by_depth=self.normalize_by_depth
        )
        self.prox_model.fit()

    def _get_base_epistemic(self, X):
        """Helper to calculate raw epistemic uncertainty for given inputs X."""
        if self.base_epistemic_method == "shaker_entropy":
            eq = EpistemicQuantifier(self.model, self.X_train, self.y_train)
            # Use 'cpu' backend here to keep things simple/stable, or auto
            return eq.shaker_get_epistemic_entropy(X, backend="cpu")
        elif self.base_epistemic_method == "likelihood":
            cq = CredalRegressionUQ(self.model, self.X_train, self.y_train)
            epistemic_var, _ = cq.compute_uq(X, backend="cpu", likelihood_type="normal")
            return epistemic_var
        else:
            raise ValueError(f"Unknown base_epistemic_method: {self.base_epistemic_method}")

    def compute_uq(self, X_test):
        """
        Computes the blended hybrid epistemic uncertainty for X_test.
        """
        X_test = np.atleast_2d(X_test)
        n_test = X_test.shape[0]
        
        # 1. Compute query point's own uncertainty
        U_query = self._get_base_epistemic(X_test)
        
        # 2. Compute proximity matrix P: shape (n_test, n_train)
        P = self.prox_model.compute_proximity_matrix(X_test)
        
        # 3. Pull to host CPU if on GPU
        if hasattr(P, "get"):
            P = P.get()
            
        U_neighbors = np.zeros(n_test, dtype=np.float32)
        k = min(self.k_neighbors, len(self.X_train))
        
        # 4. Extract top k neighbors and pool uncertainties
        for i in range(n_test):
            row_i = P[i]
            # argsort sorts ascending, so top k nearest neighbors are at the end
            top_indices = np.argsort(row_i)[-k:]
            
            w = row_i[top_indices]
            u_vals = self.train_epistemic_values[top_indices]
            
            sum_w = np.sum(w)
            if sum_w > 1e-10:
                u_pool = np.sum(w * u_vals) / sum_w
            else:
                u_pool = np.mean(u_vals)
                
            U_neighbors[i] = u_pool
            
        # 5. Blend query uncertainty and pooled neighbor uncertainty
        U_final = self.lambda_blend * U_neighbors + (1.0 - self.lambda_blend) * U_query
        return U_final

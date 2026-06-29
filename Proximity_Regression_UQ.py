import os
import sys
import numpy as np

# Reconfigure stdout and stderr to UTF-8 to prevent encoding issues in cluster environments
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from rfgap import RFGAP

class ProximityRegressionUQ:
    def __init__(self, model, X_train, y_train):
        """
        Wrapper for Localized Uncertainty Quantification in Random Forests via Proximities (RF-FIRE).
        
        Args:
            model: An unfitted or fitted RandomForestRegressor from which to copy hyper-parameters.
            X_train: np.ndarray of shape (n_samples, n_features) representing training inputs.
            y_train: np.ndarray of shape (n_samples,) representing training targets.
        """
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        
        # Copy baseline Random Forest hyper-parameters
        params = model.get_params()
        
        # RF-FIRE requires OOB score to be enabled for computing OOB residuals
        params['oob_score'] = True
        
        # Instantiate the RFGAP model using the factory class
        self.rfgap_model = RFGAP(
            prediction_type='regression',
            y=self.y_train,
            prox_method='rfgap',
            matrix_type='sparse',
            triangular=True,
            non_zero_diagonal=False,
            force_symmetric=False,
            **params
        )

    def fit(self):
        """
        Fits the RFGAP model on the training data.
        """
        self.rfgap_model.fit(self.X_train, self.y_train)

    def compute_uq(self, X_test, n_neighbors='auto', level=0.95):
        """
        Computes the localized uncertainty for the test set using RF-GAP prediction intervals (RF-FIRE).
        
        Args:
            X_test: np.ndarray of shape (n_samples, n_features) representing query points.
            n_neighbors: Number of proximate neighbors (int, 'auto', or 'all')
            level: Confidence level for the prediction interval
            
        Returns:
            uncertainty: np.ndarray of shape (n_samples,) representing the interval width (y_upper - y_lower).
        """
        # Fit the model if not already fitted
        if not hasattr(self.rfgap_model, 'leaf_matrix'):
            self.fit()
            
        y_pred_lwr, y_pred, y_pred_upr = self.rfgap_model.predict_with_intervals(
            X_test=X_test,
            n_neighbors=n_neighbors,
            level=level,
            verbose=False
        )
        
        # Width of the prediction interval represents the localized uncertainty
        uncertainty = y_pred_upr - y_pred_lwr
        return uncertainty

from abc import ABC, abstractmethod
import numpy as np

class BaseEpistemicExtractor(ABC):
    """
    Abstract base class for all Epistemic Uncertainty Quantification (UQ) extractors.
    All custom UQ extractors must subclass this to ensure compatibility with 
    the dynamic RF adaptation pipeline and the CARP-S benchmark adapter.
    """
    
    def __init__(self, model, **kwargs):
        """
        Initializes the extractor with the Random Forest model.
        
        Parameters:
        -----------
        model : RandomForestRegressor
            The fitted Random Forest surrogate model.
        """
        self.model = model

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fits/updates internal estimators or precomputes training leaf statistics.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training configurations of shape (n_samples, n_features).
        y_train : np.ndarray
            Target objective values of shape (n_samples,).
        """
        pass

    @abstractmethod
    def extract_epistemic_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the epistemic uncertainty signal for each configuration in X.
        
        Parameters:
        -----------
        X : np.ndarray
            Input configuration points of shape (n_samples, n_features).
            
        Returns:
        --------
        epistemic_signal : np.ndarray
            Epistemic uncertainty signal array of shape (n_samples,).
        """
        pass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ep_extractors import UQExtractorRegistry
from rf_dynamic.sliding_window_adaptor import SlidingWindowRFAdaptor

class DynamicRFSurrogate:
    """
    Random Forest surrogate model wrapper that dynamically adjusts its 
    hyperparameters (n_estimators, max_depth) based on epistemic uncertainty 
    signals computed on evaluated candidate points.
    """
    def __init__(
        self,
        extractor_name: str = "standard_disagreement",
        window_size: int = 5,
        n_base: int = 100,
        n_min: int = 10,
        n_max: int = 200,
        gamma: float = 1.0,
        depth_base: int = 12,
        depth_min: int = 5,
        depth_max: int = 30,
        beta: float = 5.0,
        extractor_kwargs: dict = None,
        rf_kwargs: dict = None
    ):
        self.extractor_name = extractor_name
        self.extractor_kwargs = extractor_kwargs or {}
        self.rf_kwargs = rf_kwargs or {}
        
        self.adaptor = SlidingWindowRFAdaptor(
            window_size=window_size,
            n_base=n_base,
            n_min=n_min,
            n_max=n_max,
            gamma=gamma,
            depth_base=depth_base,
            depth_min=depth_min,
            depth_max=depth_max,
            beta=beta
        )
        
        self.model = None
        self.extractor = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Random Forest model using the adapted parameters, 
        and updates/fits the epistemic extractor on the new training set.
        """
        # Get next adapted parameters
        n_estimators, max_depth = self.adaptor.get_next_parameters()
        
        # Instantiate and fit the RF model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            **self.rf_kwargs
        )
        self.model.fit(X, y)
        
        # Instantiate and fit the epistemic extractor
        self.extractor = UQExtractorRegistry.get(
            self.extractor_name,
            self.model,
            **self.extractor_kwargs
        )
        self.extractor.fit(X, y)

    def predict(self, X: np.ndarray):
        """
        Predicts mean and returns the standard disagreement (standard deviation) 
        as the uncertainty value.
        Also extracts the epistemic uncertainty signal of the selected approach 
        to trigger parameter updates in the sliding window adaptor.
        """
        if self.model is None or self.extractor is None:
            raise RuntimeError("Surrogate must be fitted before prediction.")
            
        preds = self.model.predict(X)
        
        # Extract raw epistemic uncertainty signal for hyperparameter adaptation
        raw_signals = self.extractor.extract_epistemic_signal(X)
        
        # Update sliding window adaptor (updates RF hyperparameters internally)
        _ = self.adaptor.update_and_normalize(raw_signals)
        
        # Compute standard disagreement (standard deviation) of the forest for acquisition function
        X_test = np.atleast_2d(X)
        all_test_leaf_ids = self.model.apply(X_test)
        n_samples = X_test.shape[0]
        n_trees = len(self.model.estimators_)
        
        tree_preds = np.zeros((n_trees, n_samples))
        for t, estimator in enumerate(self.model.estimators_):
            tree_preds[t, :] = estimator.tree_.value[all_test_leaf_ids[:, t], 0, 0]
            
        # Standard deviation of the tree predictions
        std_disagreement = np.std(tree_preds, axis=0)
        
        return preds, std_disagreement

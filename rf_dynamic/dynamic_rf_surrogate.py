import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ep_extractors import UQExtractorRegistry
from rf_dynamic.sliding_window_adaptor import SlidingWindowRFAdaptor

class DynamicRFSurrogate:
    """
    Random Forest surrogate model wrapper that dynamically adjusts its 
    hyperparameters (min_samples_leaf, max_features) based on epistemic uncertainty 
    signals computed on evaluated candidate points.
    """
    def __init__(
        self,
        extractor_name: str = "standard_disagreement",
        window_size: int = 5,
        min_samples_leaf_base: int = 2,
        min_samples_leaf_min: int = 1,
        min_samples_leaf_max: int = 15,
        alpha: float = 1.0,
        max_features_base: float = 0.5,
        max_features_min: float = 0.1,
        max_features_max: float = 0.8,
        eta: float = 0.5,
        extractor_kwargs: dict = None,
        rf_kwargs: dict = None
    ):
        self.extractor_name = extractor_name
        self.extractor_kwargs = extractor_kwargs or {}
        self.rf_kwargs = rf_kwargs or {}
        
        self.adaptor = SlidingWindowRFAdaptor(
            window_size=window_size,
            min_samples_leaf_base=min_samples_leaf_base,
            min_samples_leaf_min=min_samples_leaf_min,
            min_samples_leaf_max=min_samples_leaf_max,
            alpha=alpha,
            max_features_base=max_features_base,
            max_features_min=max_features_min,
            max_features_max=max_features_max,
            eta=eta
        )
        
        self.model = None
        self.extractor = None
        self.n_samples = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the Random Forest model using the adapted parameters, 
        and updates/fits the epistemic extractor on the new training set.
        """
        # Save training sample count for dynamic capping
        self.n_samples = X.shape[0]
        
        # Get next adapted parameters
        min_samples_leaf, max_features = self.adaptor.get_next_parameters()
        
        # Pre-enable oob_score for proximity extractors to prevent double fitting
        rf_kwargs = dict(self.rf_kwargs)
        if "proximity" in self.extractor_name:
            rf_kwargs["oob_score"] = True

        # Instantiate and fit the RF model
        self.model = RandomForestRegressor(
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            **rf_kwargs
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
        _ = self.adaptor.update_and_normalize(raw_signals, n_samples=self.n_samples)
        
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

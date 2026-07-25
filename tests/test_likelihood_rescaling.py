import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors.likelihood_credal import LikelihoodCredalExtractor

def test_likelihood_credal_outputs_sigma():
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(50, 2))
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, size=50)
    
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    X_test = np.random.uniform(0, 10, size=(5, 2))
    
    extractor = LikelihoodCredalExtractor(rf)
    extractor.fit(X_train, y_train)
    
    # Extract epistemic signal (should be standard deviation sigma = I_ep)
    sigma_credal = extractor.extract_epistemic_signal(X_test)
    
    # Check that output matches sqrt(epistemic_var)
    epistemic_var, _ = extractor.cruq.compute_uq(
        X_test,
        backend=extractor.backend,
        n_grid=extractor.n_grid,
        integration_method=extractor.integration_method
    )
    assert np.allclose(sigma_credal, np.sqrt(epistemic_var))
    assert np.all(sigma_credal > 0)
    print("LIKELIHOOD CREDAL SIGMA RESCALING TEST PASSED!")


if __name__ == "__main__":
    test_likelihood_credal_outputs_sigma()

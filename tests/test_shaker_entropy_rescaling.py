import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ep_extractors.shaker_entropy import ShakerEntropyExtractor

def test_shaker_entropy_outputs_sigma():
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(50, 2))
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, size=50)
    
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    X_test = np.random.uniform(0, 10, size=(5, 2))
    
    extractor = ShakerEntropyExtractor(rf, num_samples=1000)
    extractor.fit(X_train, y_train)
    
    # Extract rescaled epistemic signal (should be standard deviation sigma)
    sigma_shaker = extractor.extract_epistemic_signal(X_test)
    
    # Compute expected sigma via shaker_get_epistemic_variance
    epistemic_var = extractor.eq.shaker_get_epistemic_variance(X_test, num_samples=1000)
    expected_sigma = np.sqrt(epistemic_var)
    
    assert np.allclose(sigma_shaker, expected_sigma)
    assert np.all(sigma_shaker >= 0)
    print("SHAKER ENTROPY SIGMA RESCALING TEST PASSED!")

if __name__ == "__main__":
    test_shaker_entropy_outputs_sigma()

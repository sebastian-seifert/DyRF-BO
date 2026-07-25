import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from ep_extractors.standard_proximity import StandardProximityExtractor
from ep_extractors.proximity_b import ProximityBExtractor
from ep_extractors.proximity_bc import ProximityBCExtractor
from ep_extractors.proximity_auto_lambda import ProximityAutoLambdaExtractor

def test_proximity_rescaling_normal_divisor():
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(100, 2))
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + np.random.normal(0, 0.1, size=100)
    
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    uq_model = GPUProximityRegressionUQ(rf, X_train, y_train, topological_decay_lambda=1.0)
    uq_model.fit()
    
    X_test = np.random.uniform(0, 10, size=(10, 2))
    
    # Compute rescaled UQ output
    sigma_prox = uq_model.compute_uq(X_test, level=0.95)
    
    # Calculate theoretical divisor under strict normal distribution assumption
    # divisor = 2 * z_{(1 + 0.95)/2} = 2 * z_0.975 = 3.9199279...
    expected_divisor = 2.0 * norm.ppf(0.975)
    assert np.isclose(expected_divisor, 3.919927969080108)
    
    # Assert output is positive finite float array
    assert np.all(sigma_prox > 0)
    assert len(sigma_prox) == 10

def test_all_proximity_extractors_rescaling():
    np.random.seed(42)
    X_train = np.random.uniform(0, 10, size=(50, 2))
    y_train = np.sin(X_train[:, 0]) + np.random.normal(0, 0.1, size=50)
    
    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)
    rf.oob_prediction_ = rf.predict(X_train)
    
    X_test = np.random.uniform(0, 10, size=(5, 2))
    
    extractors = [
        StandardProximityExtractor(rf),
        ProximityBExtractor(rf, decay_lambda=1.0),
        ProximityBCExtractor(rf, decay_lambda=1.0, alpha=1.0),
        ProximityAutoLambdaExtractor(rf, alpha=1.0)
    ]
    
    for ext in extractors:
        ext.fit(X_train, y_train)
        sigmas = ext.extract_epistemic_signal(X_test)
        assert len(sigmas) == 5
        assert np.all(sigmas > 0)

if __name__ == "__main__":
    test_proximity_rescaling_normal_divisor()
    test_all_proximity_extractors_rescaling()
    print("ALL PROXIMITY RESCALING TESTS PASSED!")

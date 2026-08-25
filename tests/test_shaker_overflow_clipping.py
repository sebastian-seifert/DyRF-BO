import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from Epistemic_Quantifier import EpistemicQuantifier

def test_shaker_convert_entropy_to_var_overflow_safe():
    """Verify _shaker_convert_entropy_to_var does not overflow or produce inf/nan for huge entropy values."""
    # Create dummy fitted model
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)
    eq = EpistemicQuantifier(rf, X, y)
    
    # Astronomical entropy in bits (e.g. 5000 bits which would compute 2^10000 = inf without clipping)
    huge_entropy = np.array([0.0, 10.0, 500.0, 5000.0, 1e6])
    var = eq._shaker_convert_entropy_to_var(huge_entropy)
    
    assert not np.any(np.isnan(var))
    assert not np.any(np.isinf(var))
    assert np.all(var >= 0.0)
    # Check monotonicity: larger entropy yields >= variance
    assert np.all(np.diff(var) >= 0.0)

def test_shaker_get_epistemic_variance_overflow_safe(monkeypatch):
    """Verify shaker_get_epistemic_variance does not overflow when mi_bits is astronomical."""
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    rf = RandomForestRegressor(n_estimators=5, random_state=42).fit(X, y)
    eq = EpistemicQuantifier(rf, X, y)
    
    # Mock shaker_get_epistemic_entropy to return huge bits (e.g. 2000 bits)
    def mock_entropy(X_test, **kwargs):
        return np.array([0.5, 2.0, 1000.0, 5000.0])
    
    monkeypatch.setattr(eq, "shaker_get_epistemic_entropy", mock_entropy)
    
    X_test = np.array([[0.5], [1.5], [2.5], [3.5]])
    ep_var = eq.shaker_get_epistemic_variance(X_test)
    
    assert not np.any(np.isnan(ep_var))
    assert not np.any(np.isinf(ep_var))
    assert np.all(ep_var >= 0.0)

import pytest
import numpy as np
from carps_integration.acquisitions import (
    ExpectedImprovement,
    AdditiveEpistemicAcquisition,
    normalize_max_relative
)

def test_normalize_max_relative_basic():
    """Verify max-relative normalization scales array by max value."""
    arr = np.array([0.0, 2.0, 5.0, 10.0])
    normed = normalize_max_relative(arr)
    expected = np.array([0.0, 0.2, 0.5, 1.0])
    np.testing.assert_allclose(normed, expected, rtol=1e-5)

def test_zero_beta_rank_preservation():
    """Verify argmax(alpha_total) == argmax(alpha_base) when beta_t = 0."""
    base_acq = ExpectedImprovement(xi=0.0)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)
    
    preds = np.array([1.0, 2.0, 0.5, 3.0])
    unc_tot = np.array([0.2, 0.5, 0.1, 0.8])
    u_ep = np.array([10.0, 100.0, 0.0, 50.0])  # Disagreement/epistemic
    y_best = 0.8
    
    base_scores = base_acq.compute(preds, unc_tot, y_best)
    additive_scores = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=0.0)
    
    assert np.argmax(additive_scores) == np.argmax(base_scores)
    # Ranks must match exactly
    assert np.array_equal(np.argsort(additive_scores), np.argsort(base_scores))

def test_target_scale_invariance():
    """Verify relative scores and argmax are invariant when multiplying objective scale by 10^6."""
    base_acq = ExpectedImprovement(xi=0.0)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)
    
    preds = np.array([1.0, 2.0, 0.5, 3.0])
    unc_tot = np.array([0.2, 0.5, 0.1, 0.8])
    u_ep = np.array([0.2, 0.4, 0.1, 0.5])
    y_best = 0.8
    
    scores_standard = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=0.5)
    
    # Scale objective by 10^6
    scale = 1e6
    scores_scaled = additive_acq.compute_additive(
        preds * scale, unc_tot * scale, u_ep * scale, y_best * scale, beta_t=0.5
    )
    
    np.testing.assert_allclose(scores_standard, scores_scaled, rtol=1e-4)

def test_degenerate_zero_improvement():
    """Verify numerical stability and pure epistemic fallback when alpha_base is all zeros."""
    base_acq = ExpectedImprovement(xi=0.0)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)
    
    # Very high predictions so improvement is effectively 0 everywhere
    preds = np.array([100.0, 100.0, 100.0])
    unc_tot = np.array([1e-5, 1e-5, 1e-5])
    u_ep = np.array([0.1, 0.9, 0.3])
    y_best = 0.0
    
    scores = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
    
    assert not np.any(np.isnan(scores))
    assert not np.any(np.isinf(scores))
    # Highest epistemic uncertainty point (index 1) must be selected
    assert np.argmax(scores) == 1

def test_degenerate_zero_uncertainty():
    """Verify numerical stability when epistemic uncertainty is all zeros."""
    base_acq = ExpectedImprovement(xi=0.0)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)
    
    preds = np.array([0.1, 0.5, 0.9])
    unc_tot = np.array([0.2, 0.2, 0.2])
    u_ep = np.array([0.0, 0.0, 0.0])
    y_best = 0.5
    
    scores = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=1.0)
    
    assert not np.any(np.isnan(scores))
    assert not np.any(np.isinf(scores))
    # Standard base acquisition must determine the argmax
    base_scores = base_acq.compute(preds, unc_tot, y_best)
    assert np.argmax(scores) == np.argmax(base_scores)

def test_pareto_dominance_guarantee():
    """Verify candidate with >= base acquisition and >= epistemic uncertainty has >= total score."""
    base_acq = ExpectedImprovement(xi=0.0)
    additive_acq = AdditiveEpistemicAcquisition(base_acq=base_acq)
    
    # Candidate 0 dominates Candidate 1 in both EI and Epistemic Uncertainty
    preds = np.array([0.2, 0.8])  # lower mean is better for minimization
    unc_tot = np.array([0.5, 0.2])
    u_ep = np.array([0.8, 0.2])
    y_best = 0.5
    
    for beta in [0.0, 0.2, 0.5, 1.0, 2.0]:
        scores = additive_acq.compute_additive(preds, unc_tot, u_ep, y_best, beta_t=beta)
        assert scores[0] >= scores[1]

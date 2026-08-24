import pytest
import numpy as np
from carps_integration.acquisitions import WarmupCosineScheduler

def test_warmup_plateau_exact():
    """Verify beta_t remains exactly beta_max during the warmup window."""
    total_trials = 50
    warmup_ratio = 0.20  # t_warmup = 10
    beta_max = 1.0
    beta_min = 0.0
    
    scheduler = WarmupCosineScheduler(
        total_trials=total_trials,
        warmup_ratio=warmup_ratio,
        beta_max=beta_max,
        beta_min=beta_min
    )
    
    assert scheduler.t_warmup == 10
    for t in range(11):  # t = 0..10
        assert scheduler.get_beta(t) == pytest.approx(1.0)

def test_cosine_decay_endpoint():
    """Verify beta_T reaches beta_min at or beyond horizon T."""
    scheduler = WarmupCosineScheduler(
        total_trials=50,
        warmup_ratio=0.20,
        beta_max=1.0,
        beta_min=0.0
    )
    
    assert scheduler.get_beta(50) == pytest.approx(0.0, abs=1e-6)
    assert scheduler.get_beta(55) == pytest.approx(0.0, abs=1e-6)

def test_cosine_midpoint_analytical_value():
    """Verify beta at t = t_warmup + (T - t_warmup)/2 equals beta_min + 0.5 * (beta_max - beta_min)."""
    scheduler = WarmupCosineScheduler(
        total_trials=50,
        warmup_ratio=0.20,
        beta_max=1.0,
        beta_min=0.0
    )
    
    # t_warmup = 10, remaining = 40, midpoint = 10 + 20 = 30
    # cos(pi/2) = 0 -> beta = 0.5 * (1 + 0) = 0.5
    assert scheduler.get_beta(30) == pytest.approx(0.5, abs=1e-6)

def test_monotonic_decrease_post_warmup():
    """Verify beta_{t+1} <= beta_t for all t >= t_warmup."""
    scheduler = WarmupCosineScheduler(
        total_trials=50,
        warmup_ratio=0.20,
        beta_max=1.0,
        beta_min=0.0
    )
    
    prev_beta = scheduler.get_beta(10)
    for t in range(11, 51):
        curr_beta = scheduler.get_beta(t)
        assert curr_beta <= prev_beta + 1e-9
        prev_beta = curr_beta

def test_boundary_edge_cases():
    """Test zero warmup (omega=0), full warmup (omega=1), and single-trial budget (T=1)."""
    # 1. Zero warmup (pure cosine)
    sched_zero = WarmupCosineScheduler(total_trials=50, warmup_ratio=0.0, beta_max=1.0, beta_min=0.0)
    assert sched_zero.t_warmup == 0
    assert sched_zero.get_beta(0) == pytest.approx(1.0)
    assert sched_zero.get_beta(25) == pytest.approx(0.5, abs=1e-6)
    assert sched_zero.get_beta(50) == pytest.approx(0.0, abs=1e-6)
    
    # 2. Full warmup (always beta_max)
    sched_full = WarmupCosineScheduler(total_trials=50, warmup_ratio=1.0, beta_max=1.0, beta_min=0.0)
    assert sched_full.t_warmup == 50
    for t in range(51):
        assert sched_full.get_beta(t) == pytest.approx(1.0)
        
    # 3. T=1 edge case
    sched_one = WarmupCosineScheduler(total_trials=1, warmup_ratio=0.2, beta_max=1.0, beta_min=0.0)
    assert sched_one.get_beta(0) == pytest.approx(1.0)
    assert sched_one.get_beta(1) == pytest.approx(0.0, abs=1e-6)

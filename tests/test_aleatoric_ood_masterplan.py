import pytest
import numpy as np

def test_ood_dataset_partitioning():
    from ep_extractors.aleatoric_ood_masterplan import generate_ood_aleatoric_dataset, get_benchmark_functions, get_noise_regimes
    
    funcs = get_benchmark_functions()
    noises = get_noise_regimes()
    
    f_info = funcs["sin_cos_1d"]
    n_info = noises["hetero_ood_step_double"]
    
    X_train, y_train, X_test, y_test, sigma_true_test, is_ood_test = generate_ood_aleatoric_dataset(
        f_info=f_info,
        n_info=n_info,
        seed=42,
        n_train=1024,
        n_test=256
    )
    
    # 1. Assert training set has ZERO points inside OOD gap [4.0, 6.0]
    in_gap_train = (X_train[:, 0] >= 4.0) & (X_train[:, 0] <= 6.0)
    assert np.sum(in_gap_train) == 0, f"Found {np.sum(in_gap_train)} training points inside OOD gap [4.0, 6.0]!"
    assert len(X_train) == 1024
    
    # 2. Assert test set has both ID and OOD points (70/30 split target)
    assert len(X_test) == 256
    assert len(is_ood_test) == 256
    n_ood = np.sum(is_ood_test)
    n_id = len(is_ood_test) - n_ood
    assert n_ood > 0, "Test set contains 0 OOD points!"
    assert n_id > 0, "Test set contains 0 ID points!"
    # 30% of 256 is ~77 points
    assert 50 <= n_ood <= 100, f"Unexpected OOD test point count: {n_ood}"
    
    # 3. Assert is_ood_test strictly matches x_0 in [4.0, 6.0]
    expected_ood = (X_test[:, 0] >= 4.0) & (X_test[:, 0] <= 6.0)
    assert np.array_equal(is_ood_test, expected_ood)


def test_sobol_power_of_two_sampling():
    from ep_extractors.aleatoric_ood_masterplan import generate_ood_aleatoric_dataset, get_benchmark_functions, get_noise_regimes
    import math
    
    funcs = get_benchmark_functions()
    n_info = get_noise_regimes()["homoscedastic_low"]
    
    for d in [1, 2, 3, 5, 8, 10, 15]:
        key = f"sin_cos_{d}d"
        if key not in funcs:
            continue
        f_info = funcs[key]
        X_train, _, X_test, _, _, _ = generate_ood_aleatoric_dataset(
            f_info=f_info,
            n_info=n_info,
            seed=1,
            n_train=1024,
            n_test=256
        )
        
        # Check power of 2
        log2_train = math.log2(len(X_train))
        log2_test = math.log2(len(X_test))
        assert math.isclose(log2_train, round(log2_train), abs_tol=1e-9), f"dim {d}: train size {len(X_train)} not power of 2"
        assert math.isclose(log2_test, round(log2_test), abs_tol=1e-9), f"dim {d}: test size {len(X_test)} not power of 2"


def test_hetero_ood_step_double():
    from ep_extractors.aleatoric_ood_masterplan import get_noise_regimes
    
    noises = get_noise_regimes()
    assert "hetero_ood_step_double" in noises
    n_info = noises["hetero_ood_step_double"]
    
    X_samples = np.array([
        [2.0, 5.0],
        [5.0, 5.0],
        [8.0, 5.0]
    ])
    sigmas = n_info["func"](X_samples)
    
    assert np.isclose(sigmas[0], 0.10, atol=1e-5), f"Expected 0.10 for ID point x0=2.0, got {sigmas[0]}"
    assert np.isclose(sigmas[1], 0.20, atol=1e-5), f"Expected 0.20 for OOD point x0=5.0, got {sigmas[1]}"
    assert np.isclose(sigmas[2], 0.10, atol=1e-5), f"Expected 0.10 for ID point x0=8.0, got {sigmas[2]}"


def test_single_experiment_and_metric_slices():
    from ep_extractors.aleatoric_ood_masterplan import run_single_aleatoric_ood_experiment
    
    res = run_single_aleatoric_ood_experiment(
        func_name="sin_cos_1d",
        noise_name="hetero_ood_step_double",
        rf_config_name="RF_Default",
        seed=1,
        n_train=256,
        n_test=128
    )
    
    expected_approaches = [
        "shaker_entropy",
        "shaker_geom_var",
        "shaker_geom_std",
        "standard_ari_var",
        "standard_ari_std",
        "standard_disagreement"
    ]
    for app in expected_approaches:
        assert app in res, f"Missing approach {app} in experiment results"
        app_metrics = res[app]
        
        # Check that metrics contain global, id_only, and ood_only keys
        for scope in ["global", "id_only", "ood_only"]:
            assert f"{scope}_spearman_true" in app_metrics
            assert f"{scope}_spearman_resid" in app_metrics
            assert f"{scope}_log_pearson_true" in app_metrics
            assert f"{scope}_mse_var" in app_metrics
            assert f"{scope}_rmse_var" in app_metrics
            assert f"{scope}_nlpd_aleatoric" in app_metrics
            
        assert "ood_id_variance_ratio" in app_metrics

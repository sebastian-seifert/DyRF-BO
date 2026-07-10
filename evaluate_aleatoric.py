import numpy as np
import scipy.stats
from sklearn.ensemble import RandomForestRegressor
from Epistemic_Quantifier import EpistemicQuantifier
from Credal_Regression_UQ import CredalRegressionUQ
from synthetic_functions import (
    get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
    get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
    get_9d_functions, get_10d_functions
)
import sys
import os
import argparse

def compute_nll(y_true, y_pred, variance):
    """Computes Gaussian Negative Log-Likelihood."""
    var_clipped = np.clip(variance, 1e-6, None)
    nll_elements = 0.5 * np.log(2.0 * np.pi * var_clipped) + ((y_true - y_pred) ** 2) / (2.0 * var_clipped)
    return float(np.mean(nll_elements))

def generate_heteroscedastic_data(func, x_range, ndim, n_samples, seed):
    """Generates synthetic data with input-dependent heteroscedastic Gaussian noise."""
    rng = np.random.default_rng(seed)
    # Generate coordinates uniformly in the box range
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, ndim))
    # input-dependent true noise level (heteroscedasticity based on first coordinate)
    sigma_true = 0.05 + 0.25 * (np.sin(X[:, 0]) ** 2)
    
    # Calculate true labels
    y_true = func(*[X[:, d] for d in range(ndim)]).ravel()
    
    # Add noise scaled by local sigma
    noise = rng.normal(0, sigma_true)
    y = y_true + noise
    return X, y, sigma_true

def evaluate_aleatoric_quality(X_train, y_train, X_test, y_test, sigma_test_true, seed):
    """Fits RandomForest and evaluates aleatoric estimations against ground truth and squared residuals."""
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, oob_score=True, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Standard Aleatoric Uncertainty
    quantifier = EpistemicQuantifier(rf, X_train, y_train)
    u_a_standard = quantifier.base_get_aleatoric_variance(X_test)
    
    # Shaker / Credal Aleatoric Uncertainty
    credal_q = CredalRegressionUQ(rf, X_train, y_train)
    _, u_a_shaker = credal_q.compute_uq(X_test, backend="auto")
    
    # True target variance
    true_var = sigma_test_true ** 2
    
    # Empirical squared residuals
    sq_res = (y_test - y_pred) ** 2
    
    results = {}
    for name, u_a in [("Standard", u_a_standard), ("Shaker", u_a_shaker)]:
        # 1. Pearson and Spearman correlations with true variance
        try:
            pearson_true, _ = scipy.stats.pearsonr(u_a, true_var)
        except Exception:
            pearson_true = 0.0
            
        try:
            spearman_true, _ = scipy.stats.spearmanr(u_a, true_var)
        except Exception:
            spearman_true = 0.0
            
        # 2. Pearson and Spearman correlations with squared residuals
        try:
            pearson_res, _ = scipy.stats.pearsonr(u_a, sq_res)
        except Exception:
            pearson_res = 0.0
            
        try:
            spearman_res, _ = scipy.stats.spearmanr(u_a, sq_res)
        except Exception:
            spearman_res = 0.0
            
        # 3. MSE and MAE compared to true variance
        mse_val = float(np.mean((u_a - true_var) ** 2))
        mae_val = float(np.mean(np.abs(u_a - true_var)))
        
        # 4. Negative Log-Likelihood
        nll_val = compute_nll(y_test, y_pred, u_a)
        
        results[name] = {
            "pearson_true_var": pearson_true,
            "spearman_true_var": spearman_true,
            "pearson_sq_res": pearson_res,
            "spearman_sq_res": spearman_res,
            "mse_true_var": mse_val,
            "mae_true_var": mae_val,
            "nll": nll_val
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Aleatoric Uncertainty Estimation Quality")
    parser.add_argument("--quick", action="store_true", help="Run a quick version for test validation")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of random seeds/runs to evaluate")
    parser.add_argument("--dims", type=str, default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated list of dimensions to run")
    args = parser.parse_args()
    
    # Resolve parameters
    if args.quick:
        n_runs = 1
        n_samples_train = 200
        n_samples_test = 100
        target_dims = [1, 2]
    else:
        n_runs = args.n_runs
        n_samples_train = 2000
        n_samples_test = 1000
        target_dims = [int(x) for x in args.dims.split(",")]
        
    # Gather representative functions from synthetic_functions
    getters = [
        get_1d_functions, get_2d_functions, get_3d_functions, get_4d_functions,
        get_5d_functions, get_6d_functions, get_7d_functions, get_8d_functions,
        get_9d_functions, get_10d_functions
    ]
    
    func_dict = {}
    for d, getter in enumerate(getters, start=1):
        if d in target_dims:
            funcs = getter()
            first_key = list(funcs.keys())[0]
            func_dict[d] = {
                "name": first_key,
                "func": funcs[first_key]["func"],
                "range": funcs[first_key]["range"]
            }
            
    print("=" * 80)
    print("ALEATORIC UNCERTAINTY ESTIMATION QUALITY EVALUATION")
    print(f"Data settings: Heteroscedastic noise sigma(x) = 0.05 + 0.25 * sin^2(x_1)")
    print(f"Evaluations over {n_runs} runs per dimensionality function")
    print(f"Active dimensions: {target_dims}")
    print("=" * 80)
    
    all_results = {}
    
    for dim, info in sorted(func_dict.items()):
        func_name = info["name"]
        print(f"\nEvaluating dimension {dim}D: Function={func_name}...")
        func_obj = info["func"]
        x_range = info["range"]
        
        run_results = {"Standard": [], "Shaker": []}
        
        for seed in range(n_runs):
            X_train, y_train, _ = generate_heteroscedastic_data(func_obj, x_range, dim, n_samples_train, seed)
            X_test, y_test, sigma_test_true = generate_heteroscedastic_data(func_obj, x_range, dim, n_samples_test, seed + 1000)
            
            res = evaluate_aleatoric_quality(X_train, y_train, X_test, y_test, sigma_test_true, seed)
            for app in ["Standard", "Shaker"]:
                run_results[app].append(res[app])
                
        all_results[dim] = {
            "name": func_name,
            "runs": run_results
        }
        
        # Display averaged results
        for app in ["Standard", "Shaker"]:
            metrics_list = run_results[app]
            p_true = np.mean([m["pearson_true_var"] for m in metrics_list])
            s_true = np.mean([m["spearman_true_var"] for m in metrics_list])
            p_res = np.mean([m["pearson_sq_res"] for m in metrics_list])
            s_res = np.mean([m["spearman_sq_res"] for m in metrics_list])
            mse_val = np.mean([m["mse_true_var"] for m in metrics_list])
            mae_val = np.mean([m["mae_true_var"] for m in metrics_list])
            nll_val = np.mean([m["nll"] for m in metrics_list])
            
            print(f"  -> Approach: {app}")
            print(f"     * Corr with True Var:   Pearson = {p_true:.4f}, Spearman = {s_true:.4f}")
            print(f"     * Corr with Sq Residual: Pearson = {p_res:.4f}, Spearman = {s_res:.4f}")
            print(f"     * Error vs True Var:    MSE = {mse_val:.6f}, MAE = {mae_val:.6f}")
            print(f"     * Gaussian NLL:         Mean = {nll_val:.4f}")
            print("-" * 50)
            
    # Write a comprehensive report
    report_path = "results/aleatoric_estimation_quality_report.md"
    os.makedirs("results", exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Aleatoric Uncertainty Estimation Quality Report\n\n")
        f.write("This report evaluates the accuracy of aleatoric uncertainty estimations under input-dependent heteroscedastic noise:\n")
        f.write("$$\\sigma_{\\text{true}}(x) = 0.05 + 0.25 \\cdot \\sin^2(x_1)$$\n\n")
        
        for dim, res_info in sorted(all_results.items()):
            func_name = res_info["name"]
            run_results = res_info["runs"]
            
            f.write(f"## {dim}D Function: {func_name}\n\n")
            f.write("| Approach | Pearson (True Var) | Spearman (True Var) | Pearson (Sq Res) | Spearman (Sq Res) | MSE (True Var) | MAE (True Var) | Gaussian NLL |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for app in ["Standard", "Shaker"]:
                metrics_list = run_results[app]
                p_true = np.mean([m["pearson_true_var"] for m in metrics_list])
                s_true = np.mean([m["spearman_true_var"] for m in metrics_list])
                p_res = np.mean([m["pearson_sq_res"] for m in metrics_list])
                s_res = np.mean([m["spearman_sq_res"] for m in metrics_list])
                mse_val = np.mean([m["mse_true_var"] for m in metrics_list])
                mae_val = np.mean([m["mae_true_var"] for m in metrics_list])
                nll_val = np.mean([m["nll"] for m in metrics_list])
                
                f.write(f"| **{app}** | {p_true:.4f} | {s_true:.4f} | {p_res:.4f} | {s_res:.4f} | {mse_val:.6f} | {mae_val:.6f} | {nll_val:.4f} |\n")
            f.write("\n")
            
    print(f"\nReport successfully saved to: {report_path}\n")

if __name__ == "__main__":
    main()

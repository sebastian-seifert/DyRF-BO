import os
import sys
# Reconfigure stdout and stderr to UTF-8 to prevent encoding errors on cluster environments with non-UTF-8 locales
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
if sys.stderr is not None:
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon
from Credal_Regression_UQ import CredalRegressionUQ
from GPU_Proximity_Regression_UQ import GPUProximityRegressionUQ
from data_generator import generate_data
from metrics import (
    calculate_jensen_shannon_divergence,
    calculate_mutual_information,
    calculate_rejection_curve,
    calculate_aurc,
    calculate_oracle_rejection_curve,
    calculate_random_rejection_curve,
    calculate_aurc_exact,
    calculate_roc_metrics,
    calculate_aupr,
    calculate_nlpd
)
from Epistemic_Quantifier import EpistemicQuantifier, LeafCache
from Hybrid_Proximity_Epistemic_UQ import HybridProximityEpistemicUQ

# Helps on clusters where NVRTC does not directly support the GPU's native arch.
os.environ.setdefault("CUPY_COMPILE_WITH_PTX", "1")

"""
Bachelor Thesis: Epistemic Uncertainty Quantification
Primary Focus: Quantifying uncertainty due to lack of data/exploration (Epistemic).
Evaluation Metric: Correlation with Error in Out-of-Distribution (OOD) regions.
"""




def plot_uncertainty(name, X_test, y_test, y_pred, var_pred, X_train, y_train):
    """
    Visualizes the prediction mean and the uncertainty bands.
    Shaded area represents 2 standard deviations (approx 95% confidence).
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color='black', s=10, alpha=0.3, label='Training Data')
    plt.plot(X_test, y_test, color='green', alpha=0.5, label='True Function')
    plt.plot(X_test, y_pred, color='blue', label='RF Prediction')
    
    # Calculate 2-sigma bands
    std = np.sqrt(var_pred)
    plt.fill_between(X_test.ravel(), y_pred - 2*std, y_pred + 2*std, color='blue', alpha=0.2, label='2-sigma (Epistemic)')
    
    # Highlight the Gap
    plt.axvspan(4, 6, color='red', alpha=0.1, label='Exploration Gap')
    
    plt.title(f"Epistemic Uncertainty: {name}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    
    # Save the plot
    os.makedirs("figures", exist_ok=True)
    filename = f"uncertainty_{name.lower().replace(' ', '_')}.png"
    filename = os.path.join("figures", filename)
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

# ==========================================
# TEST FUNCTION GENERATORS (15 functions)
# ==========================================
from synthetic_functions import (
    get_1d_functions,
    get_2d_functions,
    get_3d_functions,
    get_4d_functions,
    get_5d_functions,
    get_6d_functions,
    get_7d_functions,
    get_8d_functions,
    get_9d_functions,
    get_10d_functions,
    get_11d_functions,
    get_12d_functions,
    get_13d_functions,
    get_14d_functions,
    get_15d_functions
)




def save_results_to_file(results_all, results_by_dim, approaches, n_runs, alpha=0.05, suffix="", use_density_scaling=False, output_dir=None):
    """Save comprehensive summary to a .txt file and structured JSON."""
    import io
    import json
    from contextlib import redirect_stdout

    if output_dir:
        out_dir = output_dir
    else:
        out_dir = "results/density_scaling" if use_density_scaling else "results"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_suffix = f"_{suffix}" if suffix else ""
    filename = f"{out_dir}/uncertainty_quantification_results{file_suffix}_{timestamp}.txt"
    json_filename = f"{out_dir}/uncertainty_quantification_results{file_suffix}_{timestamp}.json"

    # Save formatted txt report
    string_buffer = io.StringIO()
    with redirect_stdout(string_buffer):
        print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=alpha)

    summary_text = string_buffer.getvalue()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"EPISTEMIC UNCERTAINTY QUANTIFICATION - RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        f.write(summary_text)
        f.write(f"\n{'='*80}\n")
        f.write(f"End of Report\n")
        f.write(f"{'='*80}\n")

    # Convert numpy types to native float/int for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return convert_to_serializable(obj.tolist())
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    serializable_data = {
        "approaches": approaches,
        "n_runs": n_runs,
        "results_all": convert_to_serializable(results_all),
        "results_by_dim": convert_to_serializable(results_by_dim)
    }

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"\n[Report] Results saved to: {filename} and {json_filename}")
    return filename

def print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=0.05):
    print(f"\n\n{'='*80}")
    print(f"COMPREHENSIVE STATISTICAL SUMMARY")
    print(f"{'='*80}\n")

    metrics = ["auroc", "fpr95", "aupr", "spearman", "brier", "mi", "jsd", "naurc", "nlpd"]
    dimensions = [("All Functions", results_all),
                  ("1D Functions", results_by_dim["1D"]),
                  ("2D Functions", results_by_dim["2D"]),
                  ("3D Functions", results_by_dim["3D"]),
                  ("4D Functions", results_by_dim["4D"]),
                  ("5D Functions", results_by_dim["5D"]),
                  ("6D Functions", results_by_dim["6D"]),
                  ("7D Functions", results_by_dim["7D"]),
                  ("8D Functions", results_by_dim["8D"]),
                  ("9D Functions", results_by_dim["9D"]),
                  ("10D Functions", results_by_dim["10D"]),
                  ("11D Functions", results_by_dim["11D"]),
                  ("12D Functions", results_by_dim["12D"]),
                  ("13D Functions", results_by_dim["13D"]),
                  ("14D Functions", results_by_dim["14D"]),
                  ("15D Functions", results_by_dim["15D"])]

    for metric in metrics:
        print(f"\n{'-'*80}")
        print(f"METRIC: {metric.upper()}")
        print(f"{'-'*80}\n")

        print(f"{'DESCRIPTIVE STATISTICS':^80}")
        header = f"{'':<20} " + " ".join(f"{app:>18}" for app in approaches)
        print(header)
        print(f"{'-'*80}")

        for dim_name, results_dict in dimensions:
            print(f"{dim_name:<20}", end="")
            for app in approaches:
                values = np.array([v for v in results_dict[app][metric] if not np.isnan(v)])
                if len(values) > 0:
                    print(f" {np.mean(values):.4f}+/-{np.std(values):.4f}  ", end="")
                else:
                    print(f" {'N/A':>18}", end="")
            print()

        print(f"\n{'STATISTICAL TESTS (Friedman + Bonferroni-corrected Wilcoxon)':^80}\n")

        for dim_name, results_dict in dimensions:
            print(f"> {dim_name}")
            
            # FIXED: Reshape and average across seeds to eliminate Pseudo-Replication
            total_items = len(results_dict[approaches[0]][metric])
            n_functions = total_items // n_runs
            
            processed_data = []
            for app in approaches:
                flat_vals = np.array(results_dict[app][metric], dtype=float)
                matrix = flat_vals.reshape(n_runs, n_functions)
                processed_data.append(np.nanmean(matrix, axis=0))
                
            valid_mask = ~np.isnan(processed_data).any(axis=0)
            data = [d[valid_mask] for d in processed_data]

            if all(len(d) > 2 for d in data):
                if len(approaches) == 2:
                    print("  Friedman: Bypassed (only 2 approaches). Running direct pairwise test.")
                    import itertools
                    pairs = list(itertools.combinations(approaches, 2))
                    alpha_bonf = alpha / len(pairs)
                    print(f"  Bonferroni alpha = {alpha_bonf:.4e}")
                    print(f"  {'Pairwise Comparisons':<30} {'p-value':<15} {'Significant?':<15}")
                    print(f"  {'-'*65}")

                    for app1, app2 in pairs:
                        idx1 = approaches.index(app1)
                        idx2 = approaches.index(app2)
                        try:
                            _, p_w = wilcoxon(data[idx1], data[idx2])
                        except ValueError:
                            p_w = 1.0  # Safe fallback if differences are all zero
                        sig = "[SIG]" if p_w < alpha_bonf else "[NS]"
                        pair_str = f"{app1} vs {app2}"
                        print(f"  {pair_str:<30} {p_w:>14.4e} {sig:>15}")
                else:
                    stat, p_f = friedmanchisquare(*data)
                    sig_symbol = "***" if p_f < 0.001 else "**" if p_f < 0.01 else "*" if p_f < alpha else "ns"
                    print(f"  Friedman: chi2 = {stat:8.4f}, p = {p_f:.4e} {sig_symbol}")

                    if p_f < alpha:
                        import itertools
                        pairs = list(itertools.combinations(approaches, 2))
                        alpha_bonf = alpha / len(pairs)
                        print(f"  Bonferroni alpha = {alpha_bonf:.4e}")
                        print(f"  {'Pairwise Comparisons':<30} {'p-value':<15} {'Significant?':<15}")
                        print(f"  {'-'*65}")

                        for app1, app2 in pairs:
                            idx1 = approaches.index(app1)
                            idx2 = approaches.index(app2)
                            try:
                                _, p_w = wilcoxon(data[idx1], data[idx2])
                            except ValueError:
                                p_w = 1.0  # Safe fallback if differences are all zero
                            sig = "[SIG]" if p_w < alpha_bonf else "[NS]"
                            pair_str = f"{app1} vs {app2}"
                            print(f"  {pair_str:<30} {p_w:>14.4e} {sig:>15}")
                    else:
                        print(f"  -> No significant difference across methods (Friedman p >= {alpha})")
            else:
                print("  -> Not enough valid independent functions (blocks) to perform paired testing (Requires >= 3)")
            print()

    print(f"{'='*80}")
    print(f"Legend: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"{'='*80}\n")

def run_single_test(func_dict, func_name, seed, approaches, rf_config=1, k_neighbors='auto', gap_type='empty', sparse_multiplier=12, scaling_law='linear', debug_timing=False, use_density_scaling=False, density_scaling_alpha=1.0, topological_decay_lambda=None, n_jobs=-1, ood_type='hypercube'):
    # Determine standard Random Forest hyperparameters based on config selection to pass min_leaf to generate_data
    if rf_config in ['A', 'a']:
        n_est, min_leaf, min_split, max_feat = 100, 5, 2, "sqrt"
    elif rf_config in ['B', 'b']:
        n_est, min_leaf, min_split, max_feat = 500, 10, 2, "sqrt"
    elif rf_config in ['C', 'c']:
        n_est, min_leaf, min_split, max_feat = 1000, 25, 2, "sqrt"
    elif rf_config == 1:
        n_est, min_leaf, min_split, max_feat = 100, 5, 2, "sqrt"
    elif rf_config == 2:
        n_est, min_leaf, min_split, max_feat = 100, 25, 2, "sqrt"
    elif rf_config == 3:
        n_est, min_leaf, min_split, max_feat = 100, 50, 2, "sqrt"
    elif rf_config == 4:
        n_est, min_leaf, min_split, max_feat = 300, 10, 2, "sqrt"
    else: # Config 5
        n_est, min_leaf, min_split, max_feat = 300, 30, 2, "sqrt"

    import time
    import sys
    
    if debug_timing:
        print(f"\n  >>> Starting: Function={func_name}, Seed={seed}")
        sys.stdout.flush()

    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test, y_true_binary = generate_data(
        func_dict, func_name, seed, gap_type=gap_type, sparse_multiplier=sparse_multiplier,
        scaling_law=scaling_law, min_samples_leaf=min_leaf, ood_type=ood_type
    )
    t1 = time.perf_counter()
    if debug_timing:
        print(f"    [TIMING] Data Generation: {t1 - t0:.4f} s")
        sys.stdout.flush()

    rf = RandomForestRegressor(
        n_estimators=n_est,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        max_features=max_feat,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=seed
    )
    rf.fit(X_train, y_train)
    t2 = time.perf_counter()
    if debug_timing:
        print(f"    [TIMING] Random Forest Fitting: {t2 - t1:.4f} s")
        sys.stdout.flush()
    
    leaf_cache = LeafCache(rf, X_test)
    quantifier = EpistemicQuantifier(rf, X_train, y_train, leaf_cache=leaf_cache)
    t3 = time.perf_counter()

    y_pred = rf.predict(X_test)
    sq_error = (y_test - y_pred)**2
    gap_mask = y_true_binary == 1
    t4 = time.perf_counter()
    if debug_timing:
        print(f"    [TIMING] RF Predicting: {t4 - t3:.4f} s")
        sys.stdout.flush()

    results = {}
    u_a = quantifier.base_get_aleatoric_variance(X_test)
    t5 = time.perf_counter()
    if debug_timing:
        print(f"    [TIMING] Base Aleatoric: {t5 - t4:.4f} s")
        sys.stdout.flush()

    uncertainties = {}
    u_a_credal_dict = {}
    app_timings = {}
    for app in approaches:
        t_app_start = time.perf_counter()
        if app == "Standard": uncertainties[app] = quantifier.standard_get_epistemic_variance(X_test)
        elif app == "Standard_Aleatoric":
            uncertainties[app] = u_a
            u_a_credal_dict[app] = np.zeros_like(u_a)
        elif app == "Shaker_Aleatoric":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            _, u_a_shaker = credal_q.compute_uq(X_test, backend="auto")
            uncertainties[app] = u_a_shaker
            u_a_credal_dict[app] = np.zeros_like(u_a_shaker)
        elif app == "Shaker" or app == "Shaker_GMM_Entropy": uncertainties[app] = quantifier.shaker_get_epistemic_variance(X_test, random_state=seed)
        elif app == "Chen": uncertainties[app] = quantifier.chen_get_epistemic_variance(X_test)
        elif app == "Credal_GL_Bisect" or app == "Shaker_Likelihood_GL_Bisect":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="gauss_legendre", sup_solver="bisection")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_GL_Newton" or app == "Shaker_Likelihood_GL_Newton":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="gauss_legendre", sup_solver="newton")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_Trapz_Bisect" or app == "Shaker_Likelihood_Trapz_Bisect":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="bisection")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Credal_Trapz_Newton" or app == "Shaker_Likelihood_Trapz_Newton":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="newton")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Shaker_Likelihood_Normal":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="bisection", likelihood_type="normal")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Shaker_Likelihood_StudentT":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="bisection", likelihood_type="student_t")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Shaker_Likelihood_StudentT_Corrected":
            credal_q = CredalRegressionUQ(rf, X_train, y_train, leaf_cache=leaf_cache)
            u_e_credal, u_a_credal = credal_q.compute_uq(X_test, backend="auto", integration_method="trapezoid", sup_solver="bisection", likelihood_type="student_t_corrected")
            uncertainties[app] = u_e_credal
            u_a_credal_dict[app] = u_a_credal
        elif app == "Proximity":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=use_density_scaling,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=topological_decay_lambda
            )
            if topological_decay_lambda is not None and topological_decay_lambda < 0:
                prox_q.tune_lambda_oob()
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_neighbors, level=0.95)
        elif app == "Proximity_Auto_Lambda":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=use_density_scaling,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=1.0
            )
            prox_q.tune_lambda_oob()
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_neighbors, level=0.95)
        elif app == "Proximity_Baseline":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=False,
                topological_decay_lambda=None
            )
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_neighbors, level=0.95)
        elif app == "Proximity_Method_A":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=False,
                topological_decay_lambda=1.0
            )
            k_val = 20 if isinstance(k_neighbors, str) and k_neighbors == "auto" else k_neighbors
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_val, level=0.95)
        elif app == "Proximity_Method_B":
            l_val = 1.0 if topological_decay_lambda is None else topological_decay_lambda
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=False,
                topological_decay_lambda=l_val
            )
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors="auto", level=0.95)
        elif app == "Proximity_Method_C":
            l_val = 5.0 if topological_decay_lambda is None else topological_decay_lambda
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=True,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=l_val
            )
            k_val = 20 if isinstance(k_neighbors, str) and k_neighbors == "auto" else k_neighbors
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_val, level=0.95)
        elif app == "Proximity_Method_B_C":
            l_val = 5.0 if topological_decay_lambda is None else topological_decay_lambda
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=True,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=l_val
            )
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors="auto", level=0.95)
        elif app == "Proximity_Method_B_Norm":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=False,
                topological_decay_lambda=1.0,
                normalize_by_depth=True
            )
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors="auto", level=0.95)
        elif app == "Proximity_Method_C_Norm":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=True,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=5.0,
                normalize_by_depth=True
            )
            k_val = 20 if isinstance(k_neighbors, str) and k_neighbors == "auto" else k_neighbors
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors=k_val, level=0.95)
        elif app == "Proximity_Method_B_C_Norm":
            prox_q = GPUProximityRegressionUQ(
                rf, X_train, y_train, device="auto", batch_size="auto",
                use_density_scaling=True,
                density_scaling_alpha=density_scaling_alpha,
                topological_decay_lambda=5.0,
                normalize_by_depth=True
            )
            uncertainties[app] = prox_q.compute_uq(X_test, n_neighbors="auto", level=0.95)
        elif app in ["Hybrid_Shaker_Entropy_L20", "Hybrid_Shaker_Entropy_L40", "Hybrid_Shaker_Entropy_L70"]:
            lambda_val = 0.2 if "L20" in app else (0.4 if "L40" in app else 0.7)
            k_val = 20 if isinstance(k_neighbors, str) and k_neighbors == "auto" else k_neighbors
            hybrid_q = HybridProximityEpistemicUQ(
                rf, X_train, y_train,
                base_epistemic_method="shaker_entropy",
                proximity_decay_lambda=1.0,
                normalize_by_depth=False,
                lambda_blend=lambda_val,
                k_neighbors=k_val,
                device="auto",
                batch_size="auto"
            )
            uncertainties[app] = hybrid_q.compute_uq(X_test)
        elif app in ["Hybrid_Likelihood_L20", "Hybrid_Likelihood_L40", "Hybrid_Likelihood_L70"]:
            lambda_val = 0.2 if "L20" in app else (0.4 if "L40" in app else 0.7)
            k_val = 20 if isinstance(k_neighbors, str) and k_neighbors == "auto" else k_neighbors
            hybrid_q = HybridProximityEpistemicUQ(
                rf, X_train, y_train,
                base_epistemic_method="likelihood",
                proximity_decay_lambda=1.0,
                normalize_by_depth=False,
                lambda_blend=lambda_val,
                k_neighbors=k_val,
                device="auto",
                batch_size="auto"
            )
            uncertainties[app] = hybrid_q.compute_uq(X_test)
        t_app_end = time.perf_counter()
        app_timings[app] = t_app_end - t_app_start
        if debug_timing:
            print(f"    [TIMING] UQ Calculation ({app}): {app_timings[app]:.4f} s")
            sys.stdout.flush()

    t6 = time.perf_counter()
    for app in approaches:
        u_e = uncertainties[app]
        results[app] = {"auroc": None, "fpr95": None, "aupr": None, "spearman": None, "brier": None, "mi": None, "jsd": None, "naurc": None, "nlpd": None}

        # Calculate NLPD: total_var = epistemic + aleatoric
        if app in u_a_credal_dict and u_a_credal_dict[app] is not None:
            total_var = u_e + u_a_credal_dict[app]
        else:
            total_var = u_e + u_a
        results[app]["nlpd"] = calculate_nlpd(y_test, y_pred, total_var)

        results[app]["auroc"], results[app]["fpr95"] = calculate_roc_metrics(y_true_binary, u_e)
        results[app]["aupr"] = calculate_aupr(y_true_binary, u_e)
        if np.any(gap_mask):
            if app in u_a_credal_dict and u_a_credal_dict[app] is not None:
                spear_corr, _ = spearmanr(sq_error[gap_mask], (u_e + u_a_credal_dict[app])[gap_mask])
            else:
                spear_corr, _ = spearmanr(sq_error[gap_mask], (u_e + u_a)[gap_mask])
            results[app]["spearman"] = spear_corr
        else:
            results[app]["spearman"] = np.nan

        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            p_calibrated = np.zeros(len(u_e))
            for train_idx, val_idx in skf.split(u_e, y_true_binary):
                # 1D logistic calibration with C=1e10 (unregularized approximation, suppresses warnings)
                lr = LogisticRegression(C=1e10)
                X_train = u_e[train_idx].reshape(-1, 1)
                y_train = y_true_binary[train_idx]
                X_val = u_e[val_idx].reshape(-1, 1)
                
                lr.fit(X_train, y_train)
                p_calibrated[val_idx] = lr.predict_proba(X_val)[:, 1]
        except Exception:
            u_min = np.min(u_e)
            u_max = np.max(u_e)
            u_range = u_max - u_min + 1e-10
            p_calibrated = (u_e - u_min) / u_range
        brier_id = np.mean((p_calibrated[y_true_binary == 0] - 0) ** 2)
        brier_ood = np.mean((p_calibrated[y_true_binary == 1] - 1) ** 2)
        results[app]["brier"] = 0.5 * (brier_id + brier_ood)

        results[app]["mi"] = calculate_mutual_information(u_e, y_true_binary)
        results[app]["jsd"] = calculate_jensen_shannon_divergence(u_e, y_true_binary)

        # Compute exact NAURC analytically (discretization-free and noise-free)
        aurc_model = calculate_aurc_exact(u_e, y_pred, y_test, p_max=0.95, loss_type="MSE")
        aurc_oracle = calculate_aurc_exact((y_pred - y_test)**2, y_pred, y_test, p_max=0.95, loss_type="MSE")
        aurc_random = 0.95 * np.mean((y_pred - y_test)**2)

        denom = aurc_random - aurc_oracle
        if denom < 1e-10:
            results[app]["naurc"] = 0.0
        else:
            results[app]["naurc"] = float(np.clip((aurc_model - aurc_oracle) / denom, 0.0, 5.0))
    t7 = time.perf_counter()
    if debug_timing:
        print(f"    [TIMING] Metrics Calculation: {t7 - t6:.4f} s")
        print(f"    [TIMING] Total Single Evaluation: {t7 - t0:.4f} s")
        sys.stdout.flush()

    timings = {
        "data_generation": t1 - t0,
        "rf_fitting": t2 - t1,
        "rf_predicting": t4 - t3,
        "base_aleatoric": t5 - t4,
        "app_uq": app_timings,
        "metrics_calc": t7 - t6,
        "total_test": t7 - t0
    }

    return results, timings

def print_results(results_dict, test_name):
    print(f"\n{'='*70}")
    print(f"{test_name}")
    print(f"{'='*70}")

    for metric in ["auroc", "fpr95", "spearman", "brier", "mi", "jsd", "naurc"]:
        print(f"\n--- {metric.upper()} ---")
        for app in results_dict:
            values = np.array([v for v in results_dict[app][metric] if not np.isnan(v)])
            if len(values) > 0:
                print(f"{app:12s}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}")

def run_statistical_tests(results_dict, approaches, n_runs, alpha=0.05):
    print(f"\n--- Statistical Validation (alpha = {alpha}) ---")

    for metric in ["auroc", "fpr95", "spearman", "brier", "mi", "jsd", "naurc"]:
        print(f"\n{metric.upper()}:")
        
        # FIXED: Reshape and average across seeds to eliminate Pseudo-Replication bias
        total_items = len(results_dict[approaches[0]][metric])
        n_functions = total_items // n_runs
        
        processed_data = []
        for app in approaches:
            flat_vals = np.array(results_dict[app][metric], dtype=float)
            matrix = flat_vals.reshape(n_runs, n_functions)
            processed_data.append(np.nanmean(matrix, axis=0))
            
        valid_mask = ~np.isnan(processed_data).any(axis=0)
        data = [d[valid_mask] for d in processed_data]
        if not all(len(d) > 2 for d in data):
            print("  Result: NOT ENOUGH VALID DATA FOR Paired Testing (Requires >= 3 independent functions)")
            continue

        if len(approaches) == 2:
            print("  Bypassing Friedman Test (requires >= 3 approaches). Running paired Wilcoxon Signed-Rank Test directly:")
            app1, app2 = approaches[0], approaches[1]
            try:
                _, p_w = wilcoxon(data[0], data[1])
            except ValueError:
                p_w = 1.0  # Safe fallback if differences are all zero
            sig = "[SIG]" if p_w < alpha else "[NS]"
            print(f"    {app1} vs {app2:<25}: p = {p_w:.4e} ({sig})")
        else:
            # len(approaches) >= 3
            stat, p_f = friedmanchisquare(*data)
            print(f"  Friedman Test: chi2 = {stat:.4f}, p = {p_f:.4e}")

            if p_f < alpha:
                print(f"  Result: SIGNIFICANT (p < {alpha})")
                import itertools
                pairs = list(itertools.combinations(approaches, 2))
                alpha_bonf = alpha / len(pairs)
                print(f"  Bonferroni-corrected alpha = {alpha_bonf:.4e}")

                for app1, app2 in pairs:
                    idx1 = approaches.index(app1)
                    idx2 = approaches.index(app2)
                    try:
                        _, p_w = wilcoxon(data[idx1], data[idx2])
                    except ValueError:
                        p_w = 1.0  # Safe fallback if differences are all zero
                    sig = "[SIG]" if p_w < alpha_bonf else "[NS]"
                    pair_str = f"{app1} vs {app2}"
                    print(f"    {pair_str:<25}: p = {p_w:.4e} ({sig})")
            else:
                print(f"  Result: NOT SIGNIFICANT (p >= {alpha})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Epistemic UQ Benchmarks")
    parser.add_argument("--rf_config", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Random Forest Config ID")
    parser.add_argument("--k_neighbors", type=str, default="20", help="Neighborhood size for Proximity UQ (int or 'auto' or 'all')")
    parser.add_argument("--gap_type", type=str, default="empty", choices=["empty", "sparse"], help="OOD gap type")
    parser.add_argument("--sparse_multiplier", type=int, default=12, help="Multiplier for sparse gap points (n_keep = multiplier * ndim)")
    parser.add_argument("--scaling_law", type=str, default="linear", choices=["linear", "fractional", "leaf"], help="Scaling law for sparse gap points")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of seeds/runs")
    parser.add_argument("--seed_offset", type=int, default=0, help="Offset to add to the random seed index")
    parser.add_argument("--debug_timing", action="store_true", help="Print detailed execution timings for each section during evaluation")
    parser.add_argument("--use_density_scaling", action="store_true", help="Use leaf density scaling to prevent the overconfidence trap in Proximity UQ")
    parser.add_argument("--density_scaling_alpha", type=float, default=1.0, help="Exponent parameter alpha for leaf density scaling")
    parser.add_argument("--topological_decay_lambda", type=float, default=None, help="Decay parameter lambda for topological UQ distance. If None, topological UQ is disabled.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of CPU cores for RF training")
    parser.add_argument("--approaches", type=str, default="Standard,Proximity", help="Comma-separated list of approaches to run")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom directory path to save results")
    parser.add_argument("--ood_type", type=str, default="hypercube", choices=["hypercube", "manifold"], help="OOD generation type")
    parser.add_argument("--function", type=str, default=None, help="Evaluate only a specific function by name")
    args = parser.parse_args()

    rf_config_arg = args.rf_config
    try:
        k_neighbors_arg = int(args.k_neighbors)
    except ValueError:
        k_neighbors_arg = args.k_neighbors  # 'auto' or 'all'
    gap_type_arg = args.gap_type
    sparse_multiplier_arg = args.sparse_multiplier
    scaling_law_arg = args.scaling_law
    n_runs = args.n_runs
    debug_timing_arg = args.debug_timing
    use_density_scaling_arg = args.use_density_scaling
    density_scaling_alpha_arg = args.density_scaling_alpha
    topological_decay_lambda_arg = args.topological_decay_lambda
    n_jobs_arg = args.n_jobs
    output_dir_arg = args.output_dir
    ood_type_arg = args.ood_type


    if debug_timing_arg:
        os.environ["PROXIMITY_DEBUG"] = "1"

    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"EPISTEMIC UNCERTAINTY QUANTIFICATION - COMPREHENSIVE TEST SUITE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: RF Config={rf_config_arg}, K Neighbors={k_neighbors_arg}, Gap Type={gap_type_arg}, Multiplier={sparse_multiplier_arg}, Scaling Law={scaling_law_arg}, Runs={n_runs}, Topological Decay Lambda={topological_decay_lambda_arg}")
    print(f"{'='*70}")

    approaches = [app.strip() for app in args.approaches.split(",")]
    alpha = 0.05

    functions_1d = get_1d_functions()
    functions_2d = get_2d_functions()
    functions_3d = get_3d_functions()
    functions_4d = get_4d_functions()
    functions_5d = get_5d_functions()
    functions_6d = get_6d_functions()
    functions_7d = get_7d_functions()
    functions_8d = get_8d_functions()
    functions_9d = get_9d_functions()
    functions_10d = get_10d_functions()
    functions_11d = get_11d_functions()
    functions_12d = get_12d_functions()
    functions_13d = get_13d_functions()
    functions_14d = get_14d_functions()
    functions_15d = get_15d_functions()
    all_functions = {
        **functions_1d, **functions_2d, **functions_3d, **functions_4d, **functions_5d,
        **functions_6d, **functions_7d, **functions_8d, **functions_9d, **functions_10d,
        **functions_11d, **functions_12d, **functions_13d, **functions_14d, **functions_15d
    }
    if args.function is not None:
        if args.function in all_functions:
            all_functions = {args.function: all_functions[args.function]}
        else:
            raise ValueError(f"Function '{args.function}' not found in registered functions.")

    print(f"\n[SETUP SUMMARY]")
    print(f"  * Functions: {len(all_functions)} total (5 1D, 5 2D, 5 3D, 3 4D, 3 5D, 3 6D, 3 7D, 3 8D, 3 9D, 3 10D, 1 11D, 1 12D, 1 13D, 1 14D, 1 15D)")
    print(f"  * Runs: {n_runs} (total evaluations: {len(all_functions) * n_runs})")
    print(f"  * Approaches: {', '.join(approaches)}")
    print(f"  * Metrics: AUROC, FPR@95TPR, Spearman, Brier, MI, JSD")
    print(f"  * Statistical tests: Friedman + Bonferroni-corrected Wilcoxon (alpha={alpha})")
    
    # Detect active device for Proximity UQ
    using_gpu = False
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            using_gpu = True
    except Exception:
        pass
    device_name = "GPU (CuPy)" if using_gpu else "CPU (NumPy)"
    print(f"  * Proximity Device: {device_name}")
    sys.stdout.flush()

    # ====================
    # UNIFIED TEST: Single Pass - Aggregate by Dimension
    # ====================
    print(f"\n\n{'#'*70}")
    print("# UNIFIED TEST: All Functions (aggregated by dimension)")
    print(f"{'#'*70}")
    print(f"Running {len(all_functions)} functions x {n_runs} seeds = {len(all_functions) * n_runs} evaluations (single pass)\n")

    results_all = {app: {"auroc": [], "fpr95": [], "aupr": [], "spearman": [], "brier": [], "mi": [], "jsd": [], "naurc": [], "nlpd": []} for app in approaches}
    results_by_dim = {
        f"{d}D": {app: {"auroc": [], "fpr95": [], "aupr": [], "spearman": [], "brier": [], "mi": [], "jsd": [], "naurc": [], "nlpd": []} for app in approaches}
        for d in range(1, 16)
    }

    global_timings = {
        "data_generation": 0.0,
        "rf_fitting": 0.0,
        "rf_predicting": 0.0,
        "base_aleatoric": 0.0,
        "metrics_calc": 0.0,
        "total_tests_runtime": 0.0,
        "statistical_tests": 0.0
    }
    for app in approaches:
        global_timings[f"app_uq_{app}"] = 0.0

    test_start = time.time()
    for run_idx in range(n_runs):
        seed = run_idx + args.seed_offset
        seed_start = time.time()
        print(f"[RUN {run_idx+1}/{n_runs}] ", end="", flush=True)

        for func_name in all_functions:
            try:
                test_results, test_timings = run_single_test(
                    all_functions, func_name, seed, approaches,
                    rf_config=rf_config_arg, k_neighbors=k_neighbors_arg,
                    gap_type=gap_type_arg, sparse_multiplier=sparse_multiplier_arg,
                    scaling_law=scaling_law_arg, debug_timing=debug_timing_arg,
                    use_density_scaling=use_density_scaling_arg,
                    density_scaling_alpha=density_scaling_alpha_arg,
                    topological_decay_lambda=topological_decay_lambda_arg,
                    n_jobs=n_jobs_arg, ood_type=ood_type_arg
                )
                
                # Accumulate timings
                global_timings["data_generation"] += test_timings["data_generation"]
                global_timings["rf_fitting"] += test_timings["rf_fitting"]
                global_timings["rf_predicting"] += test_timings["rf_predicting"]
                global_timings["base_aleatoric"] += test_timings["base_aleatoric"]
                global_timings["metrics_calc"] += test_timings["metrics_calc"]
                global_timings["total_tests_runtime"] += test_timings["total_test"]
                for app in approaches:
                    global_timings[f"app_uq_{app}"] += test_timings["app_uq"].get(app, 0.0)

                if func_name in functions_1d:
                    dim_key = "1D"
                elif func_name in functions_2d:
                    dim_key = "2D"
                elif func_name in functions_3d:
                    dim_key = "3D"
                elif func_name in functions_4d:
                    dim_key = "4D"
                elif func_name in functions_5d:
                    dim_key = "5D"
                elif func_name in functions_6d:
                    dim_key = "6D"
                elif func_name in functions_7d:
                    dim_key = "7D"
                elif func_name in functions_8d:
                    dim_key = "8D"
                elif func_name in functions_9d:
                    dim_key = "9D"
                elif func_name in functions_10d:
                    dim_key = "10D"
                elif func_name in functions_11d:
                    dim_key = "11D"
                elif func_name in functions_12d:
                    dim_key = "12D"
                elif func_name in functions_13d:
                    dim_key = "13D"
                elif func_name in functions_14d:
                    dim_key = "14D"
                else:
                    dim_key = "15D"

                for app in approaches:
                    results_all[app]["auroc"].append(test_results[app]["auroc"])
                    results_all[app]["fpr95"].append(test_results[app]["fpr95"])
                    results_all[app]["aupr"].append(test_results[app]["aupr"])
                    results_all[app]["spearman"].append(test_results[app]["spearman"])
                    results_all[app]["brier"].append(test_results[app]["brier"])
                    results_all[app]["mi"].append(test_results[app]["mi"])
                    results_all[app]["jsd"].append(test_results[app]["jsd"])
                    results_all[app]["naurc"].append(test_results[app]["naurc"])
                    results_all[app]["nlpd"].append(test_results[app]["nlpd"])

                    results_by_dim[dim_key][app]["auroc"].append(test_results[app]["auroc"])
                    results_by_dim[dim_key][app]["fpr95"].append(test_results[app]["fpr95"])
                    results_by_dim[dim_key][app]["aupr"].append(test_results[app]["aupr"])
                    results_by_dim[dim_key][app]["spearman"].append(test_results[app]["spearman"])
                    results_by_dim[dim_key][app]["brier"].append(test_results[app]["brier"])
                    results_by_dim[dim_key][app]["mi"].append(test_results[app]["mi"])
                    results_by_dim[dim_key][app]["jsd"].append(test_results[app]["jsd"])
                    results_by_dim[dim_key][app]["naurc"].append(test_results[app]["naurc"])
                    results_by_dim[dim_key][app]["nlpd"].append(test_results[app]["nlpd"])

            except Exception as e:
                print(f"\n[ERROR] in seed={seed}, func={func_name}: {str(e)}")
                sys.stdout.flush()
                raise

        seed_time = time.time() - seed_start
        remaining_seeds = n_runs - (seed + 1)
        eta_total_sec = seed_time * remaining_seeds
        eta_min = int(eta_total_sec / 60)

        print(f" [OK] ({seed_time:.1f}s, ETA: {eta_min}m remaining)")
        sys.stdout.flush()

    test_time = time.time() - test_start
    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Unified test completed in {test_time/60:.1f} minutes")
    sys.stdout.flush()

    t_stats_start = time.time()
    
    # ====================
    # TEST 1: All Functions Together
    # ====================
    print(f"\n\n{'#'*70}")
    print("# TEST 1: ALL FUNCTIONS TOGETHER")
    print(f"{'#'*70}")
    print_results(results_all, f"ALL FUNCTIONS ({len(all_functions)} x {n_runs} = {len(all_functions) * n_runs} tests)")
    run_statistical_tests(results_all, approaches, n_runs, alpha=alpha)
    sys.stdout.flush()

    # ====================
    # TEST 2: By Dimension
    # ====================
    print(f"\n\n{'#'*70}")
    print("# TEST 2: BY DIMENSION")
    print(f"{'#'*70}\n")

    for dim_name, dim_key in [
        ("1D Functions", "1D"), ("2D Functions", "2D"), ("3D Functions", "3D"),
        ("4D Functions", "4D"), ("5D Functions", "5D"), ("6D Functions", "6D"),
        ("7D Functions", "7D"), ("8D Functions", "8D"), ("9D Functions", "9D"),
        ("10D Functions", "10D"), ("11D Functions", "11D"), ("12D Functions", "12D"),
        ("13D Functions", "13D"), ("14D Functions", "14D"), ("15D Functions", "15D")
    ]:
        print(f"\n[DIMENSION] {dim_name}")
        print(f"{'-'*70}")
        print_results(results_by_dim[dim_key], f"{dim_name} (runs = {n_runs})")
        run_statistical_tests(results_by_dim[dim_key], approaches, n_runs, alpha=alpha)
        sys.stdout.flush()
        
    t_stats_end = time.time()
    global_timings["statistical_tests"] = t_stats_end - t_stats_start

    # ====================
    # Performance Profiling Report
    # ====================
    if debug_timing_arg:
        print(f"\n\n{'='*70}")
        print(f"EPISTEMIC UQ BENCHMARKS - PERFORMANCE PROFILING REPORT")
        print(f"{'='*70}")
        print(f"{'Component':<35} | {'Total Time (s)':<14} | {'% of Total':<10}")
        print(f"{'-'*70}")
        
        total_tracked = (
            global_timings["data_generation"] +
            global_timings["rf_fitting"] +
            global_timings["rf_predicting"] +
            global_timings["base_aleatoric"] +
            global_timings["metrics_calc"] +
            global_timings["statistical_tests"]
        )
        for app in approaches:
            total_tracked += global_timings[f"app_uq_{app}"]
            
        def print_row(label, val):
            pct = (val / total_tracked) * 100 if total_tracked > 0 else 0.0
            print(f"{label:<35} | {val:>10.2f} s    | {pct:>7.1f}%")
            
        print_row("Data Generation", global_timings["data_generation"])
        print_row("Random Forest Fitting", global_timings["rf_fitting"])
        print_row("RF Inference (Predicting)", global_timings["rf_predicting"])
        print_row("Base Aleatoric Extraction", global_timings["base_aleatoric"])
        print_row("Metrics Calculation", global_timings["metrics_calc"])
        print_row("Statistical Validation Tests", global_timings["statistical_tests"])
        print(f"{'-'*70}")
        print("Epistemic UQ Calculations:")
        for app in approaches:
            print_row(f"  - {app}", global_timings[f"app_uq_{app}"])
        print(f"{'-'*70}")
        print_row("Total Profiler Tracked Time", total_tracked)
        print(f"Total Suite Execution Time          | {total_time:>10.2f} s    | 100.0%")
        print(f"{'='*70}\n")
        sys.stdout.flush()

    # ====================
    # Final Summary & Auto-Save
    # ====================
    print(f"\n\n{'='*70}")
    print(f"[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Total Runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Evaluations: {len(all_functions) * n_runs} tests (single-pass optimization)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    sys.stdout.flush()

    # Print comprehensive report summary to terminal
    print_comprehensive_summary(results_all, results_by_dim, approaches, n_runs, alpha=alpha)
    
    # Executing file generator to dump everything into a clean timestamped report txt file
    suffix_str = f"rf{rf_config_arg}_k{k_neighbors_arg}_{gap_type_arg}_m{sparse_multiplier_arg}_{scaling_law_arg}"
    if args.function is not None:
        suffix_str += f"_{args.function}"
    if ood_type_arg != "hypercube":
        suffix_str += f"_{ood_type_arg}"
    if topological_decay_lambda_arg is not None:
        suffix_str += f"_lambda{topological_decay_lambda_arg}"
    if use_density_scaling_arg:
        suffix_str += "_ds"
        
    save_results_to_file(
        results_all, results_by_dim, approaches, n_runs, alpha=alpha,
        suffix=suffix_str, use_density_scaling=use_density_scaling_arg,
        output_dir=output_dir_arg
    )
    sys.stdout.flush()



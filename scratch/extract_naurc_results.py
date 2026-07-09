import os
import re
import json
import numpy as np

def parse_filename(filename):
    # Example: uncertainty_quantification_results_rf1_k500_empty_m12_linear_20260707_223759.json
    pattern = r"uncertainty_quantification_results_rf(\d+)_k([a-zA-Z0-9]+)_([a-z]+)_m(\d+)_([a-z]+)_(\d{8})_(\d{6})\.json"
    match = re.search(pattern, filename)
    if match:
        rf_config = int(match.group(1))
        k = match.group(2)
        gap_type = match.group(3)
        mult = int(match.group(4))
        law = match.group(5)
        return rf_config, k, gap_type, mult, law
    return None

def main():
    results_dir = "results"
    all_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and "2026070" in f]
    
    # Nested storage: approach -> gap_type -> dim -> config -> list of NAURC values
    data = {}
    
    for filename in all_files:
        config_info = parse_filename(filename)
        if not config_info:
            continue
        rf_config, k, gap_type, mult, law = config_info
        
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r") as f:
                d = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        approaches = d.get("approaches", [])
        results_by_dim = d.get("results_by_dim", {})
        
        for app in approaches:
            if app not in data:
                data[app] = {}
            if gap_type not in data[app]:
                data[app][gap_type] = {}
                
            for dim, app_data in results_by_dim.items():
                if dim not in data[app][gap_type]:
                    data[app][gap_type][dim] = {}
                
                # Fetch NAURC values
                naurcs = app_data.get(app, {}).get("naurc", [])
                if not naurcs:
                    continue
                
                config_key = (rf_config, k, mult, law)
                if config_key not in data[app][gap_type][dim]:
                    data[app][gap_type][dim][config_key] = []
                data[app][gap_type][dim][config_key].extend(naurcs)

    # Now, for each approach, gap_type, and dimension, find the best tuned configuration
    # (i.e. the one with the lowest mean NAURC)
    best_results = {}
    
    target_non_tuned = ["Standard", "Chen", "Shaker_GMM_Entropy"]
    likelihood_variants = [
        "Shaker_Likelihood_GL_Bisect",
        "Shaker_Likelihood_GL_Newton",
        "Shaker_Likelihood_Trapz_Bisect",
        "Shaker_Likelihood_Trapz_Newton"
    ]
    
    target_tuned = ["Proximity_Baseline", "Proximity_Method_A", "Proximity_Method_B", "Proximity_Method_C", "Proximity_Method_B_C"]
    
    # Map raw approaches to presentation names
    app_mapping = {
        "Standard": "Standard RF",
        "Chen": "Chen Paired",
        "Shaker_GMM_Entropy": "Shaker GMM",
        "Shaker_Likelihood": "Credal Likelihood",
        "Proximity_Baseline": "Std Prox",
        "Proximity_Method_A": "Method A",
        "Proximity_Method_B": "Method B",
        "Proximity_Method_C": "Method C",
        "Proximity_Method_B_C": "Method B+C"
    }
    
    dimensions = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D"]
    gap_types = ["empty", "sparse"]
    
    # Helper to resolve best configuration
    def get_best_stats(app_name, gap_type, dim):
        if app_name not in data or gap_type not in data[app_name] or dim not in data[app_name][gap_type]:
            return None, None
        
        configs = data[app_name][gap_type][dim]
        best_mean = float("inf")
        best_std = 0.0
        best_cfg = None
        
        for cfg, vals in configs.items():
            if not vals:
                continue
            mean = np.mean(vals)
            if mean < best_mean:
                best_mean = mean
                best_std = np.std(vals)
                best_cfg = cfg
                
        if best_cfg is None:
            return None, None
        return best_mean, best_std

    # Process likelihood variants as one "Shaker_Likelihood" approach
    data["Shaker_Likelihood"] = {}
    for gap_type in gap_types:
        data["Shaker_Likelihood"][gap_type] = {}
        for dim in dimensions:
            data["Shaker_Likelihood"][gap_type][dim] = {}
            for variant in likelihood_variants:
                if variant in data and gap_type in data[variant] and dim in data[variant][gap_type]:
                    for cfg, vals in data[variant][gap_type][dim].items():
                        if cfg not in data["Shaker_Likelihood"][gap_type][dim]:
                            data["Shaker_Likelihood"][gap_type][dim][cfg] = []
                        data["Shaker_Likelihood"][gap_type][dim][cfg].extend(vals)
                        
    # Compile the final table data
    final_table = {}
    for gap_type in gap_types:
        final_table[gap_type] = {}
        for dim in dimensions:
            final_table[gap_type][dim] = {}
            for app in list(app_mapping.keys()):
                mean, std = get_best_stats(app, gap_type, dim)
                if mean is not None:
                    final_table[gap_type][dim][app_mapping[app]] = f"{mean:.4f}±{std:.4f}"
                else:
                    final_table[gap_type][dim][app_mapping[app]] = "N/A"
                    
    # Print Markdown tables
    for gap_type in gap_types:
        print(f"\n### GAP TYPE: {gap_type.upper()}")
        header = "| Dim | " + " | ".join(app_mapping.values()) + " |"
        sep = "|---|" + "|".join(["---" for _ in app_mapping]) + "|"
        print(header)
        print(sep)
        for dim in dimensions:
            row = f"| {dim} | "
            row_vals = []
            for app_pres in app_mapping.values():
                val = final_table[gap_type][dim].get(app_pres, "N/A")
                row_vals.append(val)
            row += " | ".join(row_vals) + " |"
            print(row)

if __name__ == "__main__":
    main()

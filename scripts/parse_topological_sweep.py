import os
import re
import json
import numpy as np
import pandas as pd

def parse_sweep_results():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return

    # Pattern to match results filenames and extract metadata
    # e.g., uncertainty_quantification_results_rf5_k20_sparse_m50_leaf_lambda1.0_ds_20260705_065714.json
    pattern = re.compile(
        r"uncertainty_quantification_results_rf(?P<rf>\d)_k(?P<k>\w+?)_(?P<gap>empty|sparse)"
        r"(?:_m(?P<mult>\d+))?(?:_(?P<law>linear|fractional|leaf))?"
        r"(?:_lambda(?P<lambda>[\d\.]+))?(?:_(?P<ds>ds))?_\d{8}_\d{6}\.json"
    )

    records = []

    for root, dirs, files in os.walk(results_dir):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            match = pattern.match(filename)
            if not match:
                continue

            meta = match.groupdict()
            filepath = os.path.join(root, filename)
        
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                # Format metadata
                meta["rf"] = int(meta["rf"])
                try:
                    meta["k"] = int(meta["k"])
                except ValueError:
                    meta["k"] = str(meta["k"]) # 'auto'
                
                if meta["gap"] == "sparse":
                    meta["mult"] = int(meta["mult"]) if meta["mult"] else 12
                    meta["law"] = meta["law"] if meta["law"] else "linear"
                else:
                    meta["mult"] = 0
                    meta["law"] = "none"

                meta["lambda"] = float(meta["lambda"]) if meta["lambda"] else 0.0
                meta["ds"] = True if meta["ds"] else False
                
                # Identify Method label:
                # - lambda = 0, ds = False: Proximity UQ (Standard Proximity UQ)
                # - lambda > 0, ds = False: Topological Proximity UQ (Method A/B depending on other configs)
                # - lambda > 0, ds = True: Topological Proximity UQ + Density Scaling (Method C or A+C / B+C)
                if meta["lambda"] == 0.0 and not meta["ds"]:
                    meta["method_type"] = "Standard Proximity (Baseline)"
                elif meta["lambda"] > 0.0 and not meta["ds"]:
                    meta["method_type"] = f"Topological Proximity (lambda={meta['lambda']})"
                elif meta["lambda"] > 0.0 and meta["ds"]:
                    meta["method_type"] = f"Topological Proximity + DS (lambda={meta['lambda']})"
                else:
                    meta["method_type"] = "Unknown"

                # Extract metrics
                for app in ["Standard", "Proximity"]:
                    meta[f"{app}_auroc"] = np.nanmean(content["results_all"][app]["auroc"])
                    meta[f"{app}_spearman"] = np.nanmean(content["results_all"][app]["spearman"])
                    meta[f"{app}_brier"] = np.nanmean(content["results_all"][app]["brier"])
                    meta[f"{app}_mi"] = np.nanmean(content["results_all"][app]["mi"])
                    meta[f"{app}_jsd"] = np.nanmean(content["results_all"][app]["jsd"])
                
                # Store dimensional details
                meta["content"] = content
                records.append(meta)
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    if not records:
        print("No valid results found.")
        return

    print(f"Parsed {len(records)} benchmark configurations.\n")
    df = pd.DataFrame(records)

    # 1. Show overall top configurations by Proximity AUROC
    df_sorted = df.sort_values(by="Proximity_auroc", ascending=False)
    print("=== TOP 10 CONFIGURATIONS BY PROXIMITY AUROC ===")
    cols = ["rf", "k", "gap", "mult", "law", "lambda", "ds", "Proximity_auroc", "Standard_auroc", "Proximity_brier", "Standard_brier"]
    print(df_sorted[cols].head(10).to_string(index=False))
    print("\n" + "="*50 + "\n")

    # 2. Show overall top configurations by Proximity Brier Score (lower is better)
    df_sorted_brier = df.sort_values(by="Proximity_brier", ascending=True)
    print("=== TOP 10 CONFIGURATIONS BY PROXIMITY BRIER SCORE ===")
    print(df_sorted_brier[cols].head(10).to_string(index=False))
    print("\n" + "="*50 + "\n")

    # 3. Group by method type and check average performance across all configurations
    print("=== AVERAGE PERFORMANCE BY METHOD TYPE ===")
    summary_by_method = df.groupby("method_type").agg({
        "Proximity_auroc": "mean",
        "Standard_auroc": "mean",
        "Proximity_brier": "mean",
        "Standard_brier": "mean",
        "Proximity_spearman": "mean",
        "Standard_spearman": "mean"
    }).reset_index()
    print(summary_by_method.to_string(index=False))
    print("\n" + "="*50 + "\n")

    # 4. Dimensional breakdown of the best overall config (by AUROC)
    best_config = df_sorted.iloc[0]
    print(f"=== DIMENSIONAL BREAKDOWN FOR BEST CONFIG ===")
    print(f"RF={best_config['rf']}, K={best_config['k']}, gap={best_config['gap']} ({best_config['law']}, mult={best_config['mult']}), lambda={best_config['lambda']}, ds={best_config['ds']}")
    print(f"Overall Proximity AUROC: {best_config['Proximity_auroc']:.4f} vs Standard: {best_config['Standard_auroc']:.4f}")
    
    results_by_dim = best_config["content"]["results_by_dim"]
    dim_data = []
    for dim in sorted(results_by_dim.keys(), key=lambda x: int(x[:-1])):
        prox_dim_auroc = np.nanmean(results_by_dim[dim]["Proximity"]["auroc"])
        std_dim_auroc = np.nanmean(results_by_dim[dim]["Standard"]["auroc"])
        prox_dim_brier = np.nanmean(results_by_dim[dim]["Proximity"]["brier"])
        std_dim_brier = np.nanmean(results_by_dim[dim]["Standard"]["brier"])
        prox_dim_spear = np.nanmean(results_by_dim[dim]["Proximity"]["spearman"])
        std_dim_spear = np.nanmean(results_by_dim[dim]["Standard"]["spearman"])
        
        dim_data.append({
            "Dimension": dim,
            "Prox_AUROC": prox_dim_auroc,
            "Std_AUROC": std_dim_auroc,
            "AUROC_Diff": prox_dim_auroc - std_dim_auroc,
            "Prox_Brier": prox_dim_brier,
            "Std_Brier": std_dim_brier,
            "Brier_Diff": prox_dim_brier - std_dim_brier,
            "Prox_Spear": prox_dim_spear,
            "Std_Spear": std_dim_spear,
            "Spear_Diff": prox_dim_spear - std_dim_spear
        })
    df_dim = pd.DataFrame(dim_data)
    print(df_dim.to_string(index=False))

if __name__ == "__main__":
    parse_sweep_results()

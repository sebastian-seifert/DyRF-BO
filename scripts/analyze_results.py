#!/usr/bin/env python3
import os
import re
import json
import numpy as np

def analyze():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        return

    pattern = re.compile(
        r"uncertainty_quantification_results_rf(?P<rf>\d)_k(?P<k>\w+?)_(?P<gap>empty|sparse)(?:_m(?P<mult>\d+))?(?:_(?P<law>linear|fractional|leaf))?(?:_(?P<ds>ds))?_\d{8}_\d{6}\.json"
    )

    records = []

    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        match = pattern.match(filename)
        if not match:
            continue

        meta = match.groupdict()
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            meta["rf"] = int(meta["rf"])
            try:
                meta["k"] = int(meta["k"])
            except ValueError:
                pass
            
            if meta["gap"] == "sparse":
                meta["mult"] = int(meta["mult"]) if meta["mult"] else 12
                meta["law"] = meta["law"] if meta["law"] else "linear"
            
            # Extract metrics
            for app in ["Standard", "Proximity"]:
                meta[f"{app}_auroc"] = np.nanmean(content["results_all"][app]["auroc"])
                meta[f"{app}_spearman"] = np.nanmean(content["results_all"][app]["spearman"])
                meta[f"{app}_brier"] = np.nanmean(content["results_all"][app]["brier"])
                meta[f"{app}_mi"] = np.nanmean(content["results_all"][app]["mi"])
                meta[f"{app}_jsd"] = np.nanmean(content["results_all"][app]["jsd"])
            
            meta["content"] = content
            records.append(meta)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

    if not records:
        print("No valid results found.")
        return

    print(f"Parsed {len(records)} benchmark configurations.\n")

    # 1. Best Proximity Configurations by AUROC
    sorted_by_auroc = sorted(records, key=lambda x: x["Proximity_auroc"], reverse=True)
    
    print("=== TOP 5 PROXIMITY CONFIGURATIONS BY AUROC ===")
    for i, r in enumerate(sorted_by_auroc[:5]):
        gap_info = f"gap={r['gap']}"
        if r["gap"] == "sparse":
            gap_info += f" ({r['law']}, mult={r['mult']})"
        print(f"{i+1}. RF={r['rf']}, K={r['k']}, {gap_info}")
        print(f"   Proximity AUROC: {r['Proximity_auroc']:.4f} | Standard AUROC: {r['Standard_auroc']:.4f}")
        print(f"   Proximity Brier: {r['Proximity_brier']:.4f} | Standard Brier: {r['Standard_brier']:.4f}")
        print(f"   Proximity Spearman: {r['Proximity_spearman']:.4f} | Standard Spearman: {r['Standard_spearman']:.4f}")
        print()

    # 2. Best Proximity Configurations by Brier Score (lower is better)
    sorted_by_brier = sorted(records, key=lambda x: x["Proximity_brier"])
    print("=== TOP 5 PROXIMITY CONFIGURATIONS BY BRIER SCORE ===")
    for i, r in enumerate(sorted_by_brier[:5]):
        gap_info = f"gap={r['gap']}"
        if r["gap"] == "sparse":
            gap_info += f" ({r['law']}, mult={r['mult']})"
        print(f"{i+1}. RF={r['rf']}, K={r['k']}, {gap_info}")
        print(f"   Proximity Brier: {r['Proximity_brier']:.4f} | Standard Brier: {r['Standard_brier']:.4f}")
        print(f"   Proximity AUROC: {r['Proximity_auroc']:.4f} | Standard AUROC: {r['Standard_auroc']:.4f}")
        print()

    # 3. Overall average comparison
    mean_prox_auroc = np.mean([r["Proximity_auroc"] for r in records])
    mean_std_auroc = np.mean([r["Standard_auroc"] for r in records])
    mean_prox_brier = np.mean([r["Proximity_brier"] for r in records])
    mean_std_brier = np.mean([r["Standard_brier"] for r in records])
    mean_prox_spear = np.mean([r["Proximity_spearman"] for r in records])
    mean_std_spear = np.mean([r["Standard_spearman"] for r in records])

    print("=== OVERALL SWEEP MEAN COMPARISON (Across all 63 configs) ===")
    print(f"Metric      | Proximity UQ | Standard UQ | Diff")
    print(f"------------|--------------|-------------|------")
    print(f"AUROC       | {mean_prox_auroc:.4f}       | {mean_std_auroc:.4f}      | {mean_prox_auroc - mean_std_auroc:+.4f}")
    print(f"Brier Score | {mean_prox_brier:.4f}       | {mean_std_brier:.4f}      | {mean_prox_brier - mean_std_brier:+.4f}")
    print(f"Spearman    | {mean_prox_spear:.4f}       | {mean_std_spear:.4f}      | {mean_prox_spear - mean_std_spear:+.4f}")
    print()

    # 4. Dimensional breakdown of the best overall config (by AUROC)
    best_config = sorted_by_auroc[0]
    print(f"=== DIMENSIONAL BREAKDOWN FOR BEST CONFIG (RF={best_config['rf']}, K={best_config['k']}, {best_config['gap']}) ===")
    results_by_dim = best_config["content"]["results_by_dim"]
    for dim in sorted(results_by_dim.keys()):
        prox_dim_auroc = np.nanmean(results_by_dim[dim]["Proximity"]["auroc"])
        std_dim_auroc = np.nanmean(results_by_dim[dim]["Standard"]["auroc"])
        prox_dim_brier = np.nanmean(results_by_dim[dim]["Proximity"]["brier"])
        std_dim_brier = np.nanmean(results_by_dim[dim]["Standard"]["brier"])
        print(f"Dimension {dim}:")
        print(f"  AUROC       -> Proximity: {prox_dim_auroc:.4f} | Standard: {std_dim_auroc:.4f} | Diff: {prox_dim_auroc - std_dim_auroc:+.4f}")
        print(f"  Brier Score -> Proximity: {prox_dim_brier:.4f} | Standard: {std_dim_brier:.4f} | Diff: {prox_dim_brier - std_dim_brier:+.4f}")
        print()

if __name__ == "__main__":
    analyze()

import os
import glob
import json
import numpy as np

def parse_all_results():
    json_files = glob.glob("results/uncertainty_quantification_results_*.json")
    json_files.sort()
    
    print("=========================================================================")
    # Format a table header
    header = f"{'Config File':<40} | {'Method':<20} | {'AUROC':<7} | {'AURC':<7} | {'Excess':<7}"
    print(header)
    print("=" * 89)
    
    # Get files matching the specific timestamp of last night's sweep:
    # 20260705 or 20260706
    json_files = [f for f in glob.glob("results/uncertainty_quantification_results_*.json") 
                  if "20260705" in f or "20260706" in f]
    json_files.sort()
    
    for fpath in json_files:
        filename = os.path.basename(fpath)
        parts = filename.split("_")
        rf_config = parts[3]
        gap_type = parts[5]
        
        config_name = f"{rf_config} | gap={gap_type}"
        
        with open(fpath, "r") as f:
            data = json.load(f)
            
        results_all = data["results_all"]
        
        # Verify if aurc is in keys
        first_app = list(results_all.keys())[0]
        if "aurc" not in results_all[first_app]:
            continue
            
        print(f"\nResults for: {filename}")
        print("-" * 80)
        
        for app in results_all.keys():
            # Skip old names if we encounter them
            if app not in results_all:
                continue
                
            auroc_vals = results_all[app]["auroc"]
            aurc_vals = results_all[app]["aurc"]
            excess_vals = results_all[app]["excess_aurc"]
            
            mean_auroc = np.nanmean(auroc_vals)
            mean_aurc = np.nanmean(aurc_vals)
            mean_excess = np.nanmean(excess_vals)
            
            print(f"  {app:35s} | AUROC = {mean_auroc:.4f} | AURC = {mean_aurc:.4f} | Excess AURC = {mean_excess:.4f}")
        print("=" * 80)

if __name__ == "__main__":
    parse_all_results()

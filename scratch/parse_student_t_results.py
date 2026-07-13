import os
import json
import numpy as np

def main():
    results_dir = "results/student_t_sweep_20260712_155550"
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} not found.")
        return

    # Find all json files
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    approaches = ["Shaker_Likelihood_Normal", "Shaker_Likelihood_StudentT", "Shaker_Likelihood_StudentT_Corrected"]
    metrics = ["auroc", "fpr95", "aupr", "spearman", "brier", "mi", "jsd", "naurc", "nlpd"]

    print("Parsed JSON files:")
    all_data = []
    for jf in sorted(json_files):
        filepath = os.path.join(results_dir, jf)
        print(f"  * Reading: {jf}")
        with open(filepath, "r") as f:
            data = json.load(f)
            # data structure: {"results_all": {"approach": {"metric": [values_per_eval]}}, "results_by_dim": ...}
            if "results_all" in data:
                all_data.append(data["results_all"])

    if not all_data:
        print("No results_all data found.")
        return

    # Compute overall statistics (mean +/- std) per approach per metric
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: NORMAL VS STUDENT-T VS CORRECTED STUDENT-T LIKELIHOOD")
    print("="*80)

    # Print markdown table header
    header = f"| Metric | " + " | ".join(f"{app}" for app in approaches) + " |"
    sep = "| --- | " + " | ".join("---" for _ in approaches) + " |"
    print(header)
    print(sep)

    for metric in metrics:
        row = f"| **{metric.upper()}** |"
        for app in approaches:
            vals = []
            for d in all_data:
                if app in d and metric in d[app]:
                    vals.extend([v for v in d[app][metric] if v is not None and not np.isnan(v)])
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                row += f" {mean_val:.4f} &plusmn; {std_val:.4f} |"
            else:
                row += " N/A |"
        print(row)

if __name__ == "__main__":
    main()
